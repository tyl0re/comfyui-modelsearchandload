"""HTTP API routes for the model downloader, registered with ComfyUI's aiohttp server."""

from __future__ import annotations

import json
from aiohttp import web

try:
    from server import PromptServer  # ComfyUI's server
except ImportError:  # pragma: no cover
    PromptServer = None

from .scanner import scan_workflow
from .sources import find_candidates
from .downloader import manager
from .config import load_config, save_config, DEFAULT_CONFIG


_REGISTERED = False


def register_routes() -> None:
    global _REGISTERED
    if _REGISTERED or PromptServer is None:
        return
    app = PromptServer.instance.app
    routes = web.RouteTableDef()

    @routes.post("/model_downloader/scan")
    async def _scan(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        workflow = payload.get("workflow") or payload
        missing = scan_workflow(workflow)
        return web.json_response({"missing": missing})

    @routes.post("/model_downloader/search")
    async def _search(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        filename = (payload.get("filename") or "").strip()
        folder_hint = payload.get("folder")
        if not filename:
            return web.json_response({"error": "filename required"}, status=400)
        candidates = find_candidates(filename, folder_hint=folder_hint)
        return web.json_response({"filename": filename, "candidates": candidates})

    @routes.post("/model_downloader/download")
    async def _download(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        url = payload.get("url")
        folder = payload.get("folder")
        filename = payload.get("filename")
        subfolder = payload.get("subfolder") or ""
        source = payload.get("source", "manual")
        size = payload.get("size")
        try:
            expected_size = int(size) if size else None
        except (TypeError, ValueError):
            expected_size = None
        if not url or not folder or not filename:
            return web.json_response(
                {"error": "url, folder and filename are required"}, status=400
            )
        job = manager.enqueue(
            url=url, folder=folder, filename=filename,
            subfolder=subfolder, source=source, expected_size=expected_size,
        )
        # If enqueue() returned a job that was already running for the same
        # filename + destination, we tell the client so the UI can show
        # "already in progress" instead of "queued".
        return web.json_response({
            "job": job.to_dict(),
            "duplicate": bool(getattr(job, "_is_duplicate_of_active", False)),
        })

    @routes.get("/model_downloader/jobs")
    async def _jobs(request: web.Request):
        return web.json_response({"jobs": manager.list_jobs()})

    @routes.post("/model_downloader/cancel")
    async def _cancel(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        jid = payload.get("id")
        if not jid:
            return web.json_response({"error": "id required"}, status=400)
        ok = manager.cancel(jid)
        return web.json_response({"cancelled": ok})

    @routes.post("/model_downloader/clear")
    async def _clear(request: web.Request):
        n = manager.clear_finished()
        return web.json_response({"removed": n})

    @routes.post("/model_downloader/relocate")
    async def _relocate(request: web.Request):
        """Try to move files that are already on disk but in the wrong place
        (or with the wrong subfolder) to where ComfyUI actually looks for them.

        Body: { "items": [ {"name": "foo.safetensors", "folder": "loras",
                            "subfolder": "Wan2_2"}, ... ] }
        """
        import os
        import shutil
        from .scanner import _build_local_index, get_target_directory

        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        items = payload.get("items") or []

        # Build a basename -> [absolute paths] map by walking models tree
        try:
            from .scanner import _models_root_dirs, _LOCAL_INDEX_EXTS
        except Exception:
            return web.json_response({"error": "scanner import failed"}, status=500)

        basename_to_paths: dict[str, list[str]] = {}
        for root in _models_root_dirs():
            for dirpath, _dirs, files in os.walk(root, followlinks=False):
                for fn in files:
                    if fn.lower().endswith(_LOCAL_INDEX_EXTS):
                        basename_to_paths.setdefault(fn.lower(), []).append(
                            os.path.join(dirpath, fn)
                        )

        results = []
        for item in items:
            name = (item.get("name") or "").strip()
            folder = item.get("folder")
            subfolder = (item.get("subfolder") or "").replace("\\", "/").strip("/")
            if not name or not folder:
                results.append({"name": name, "status": "skipped",
                                "reason": "name and folder required"})
                continue

            existing = basename_to_paths.get(name.lower(), [])
            if not existing:
                results.append({"name": name, "status": "not_found"})
                continue

            target_dir = get_target_directory(folder)
            if subfolder:
                parts = [p for p in subfolder.split("/") if p not in ("", ".", "..")]
                target_dir = os.path.join(target_dir, *parts)
            target_path = os.path.join(target_dir, name)

            # Already at the right place?
            if any(os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(target_path))
                   for p in existing):
                results.append({"name": name, "status": "already_correct",
                                "path": target_path})
                continue

            # Move the first existing copy. We don't overwrite if something
            # is there already (would lose data).
            src = existing[0]
            if os.path.exists(target_path):
                results.append({"name": name, "status": "target_exists",
                                "src": src, "target": target_path})
                continue

            try:
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(src, target_path)
                results.append({
                    "name": name,
                    "status": "moved",
                    "from": src,
                    "to": target_path,
                })
            except Exception as e:
                results.append({"name": name, "status": "error", "reason": str(e)})

        moved = sum(1 for r in results if r["status"] == "moved")
        return web.json_response({"results": results, "moved": moved, "total": len(items)})

    @routes.post("/model_downloader/download_all")
    async def _download_all(request: web.Request):
        """
        Bulk download: takes a list of {filename, folder} entries (typically the
        output of /scan), auto-picks the best candidate for each, and queues
        downloads. Returns per-item status.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        items = payload.get("items") or []
        if not isinstance(items, list):
            return web.json_response({"error": "items must be a list"}, status=400)

        results = []
        for item in items:
            name = (item.get("name") or item.get("filename") or "").strip()
            folder = item.get("folder")
            subfolder = (item.get("subfolder") or "").strip()
            if not name:
                results.append({"name": name, "status": "skipped", "reason": "empty filename"})
                continue
            try:
                candidates = find_candidates(name, folder_hint=folder)
            except Exception as e:
                results.append({"name": name, "status": "error", "reason": f"search failed: {e}"})
                continue
            if not candidates:
                results.append({"name": name, "status": "no_source", "reason": "no candidate found"})
                continue

            # Pick the best candidate:
            # 1. Anything marked "preferred" (came from the curated DB)
            # 2. Otherwise highest download count
            preferred = [c for c in candidates if c.get("preferred")]
            chosen = preferred[0] if preferred else candidates[0]

            try:
                size_hint = chosen.get("size")
                try:
                    size_hint = int(size_hint) if size_hint else None
                except (TypeError, ValueError):
                    size_hint = None
                job = manager.enqueue(
                    url=chosen["url"],
                    folder=chosen.get("folder") or folder or "checkpoints",
                    filename=chosen.get("filename") or name,
                    subfolder=subfolder,
                    source=chosen.get("source", "auto"),
                    expected_size=size_hint,
                )
                # If the manager returned an already-running job, classify
                # it differently so the UI can say "already downloading"
                # instead of "queued".
                is_dup = bool(getattr(job, "_is_duplicate_of_active", False))
                results.append({
                    "name": name,
                    "status": "already_active" if is_dup else "queued",
                    "job_id": job.id,
                    "source": chosen.get("source"),
                    "title": chosen.get("title"),
                    "gated": chosen.get("gated", False),
                })
            except Exception as e:
                results.append({"name": name, "status": "error", "reason": f"enqueue failed: {e}"})

        queued = sum(1 for r in results if r["status"] == "queued")
        already = sum(1 for r in results if r["status"] == "already_active")
        return web.json_response({
            "results": results,
            "queued": queued,
            "already_active": already,
            "total": len(items),
        })

    def _mask_token(v: str) -> str:
        """Return a masked preview like 'hf_xx••••••••wxyz' that never exposes the full token."""
        if not v:
            return ""
        n = len(v)
        if n <= 8:
            # Token too short to safely show any part of it.
            return "•" * n
        # Show first 4 and last 4 chars (or fewer for shorter tokens), mask the rest.
        prefix_len = 4 if n >= 12 else 2
        suffix_len = 4 if n >= 12 else 2
        middle = max(4, n - prefix_len - suffix_len)
        # Cap the middle dot count so the UI doesn't get a giant string for huge tokens.
        middle = min(middle, 12)
        return f"{v[:prefix_len]}{'•' * middle}{v[-suffix_len:]}"

    @routes.get("/model_downloader/config")
    async def _get_config(request: web.Request):
        cfg = load_config()
        # Mask tokens in responses
        masked = dict(cfg)
        for k in ("huggingface_token", "civitai_token"):
            v = masked.get(k) or ""
            if v:
                masked[k + "_set"] = True
                masked[k + "_masked"] = _mask_token(v)
                masked[k + "_length"] = len(v)
                masked[k] = ""  # never expose the real value
            else:
                masked[k + "_set"] = False
                masked[k + "_masked"] = ""
                masked[k + "_length"] = 0
        return web.json_response(masked)

    @routes.post("/model_downloader/config")
    async def _set_config(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        cfg = load_config()
        ignored: list[str] = []
        for k in DEFAULT_CONFIG.keys():
            if k not in payload:
                continue
            value = payload[k]
            if k.endswith("_token"):
                # Empty string = leave unchanged, unless explicit clear flag set
                if value == "" and not payload.get("clear_" + k):
                    continue
                # Reject masked previews so an accidentally-copied placeholder
                # like "hf_xx••••••••wxyz" never overwrites the real token.
                if isinstance(value, str) and "•" in value:
                    ignored.append(k)
                    continue
            cfg[k] = value
        save_config(cfg)
        return web.json_response({"ok": True, "ignored": ignored})

    app.add_routes(routes)
    _REGISTERED = True
    print("[ModelDownloader] HTTP routes registered.")
