"""HTTP API routes for the model downloader, registered with ComfyUI's aiohttp server."""

from __future__ import annotations

import json
from aiohttp import web

try:
    from server import PromptServer  # ComfyUI's server
except ImportError:  # pragma: no cover
    PromptServer = None

from .scanner import scan_workflow
from .sources import find_candidates, search_web_for_huggingface
from .downloader import manager
from .config import load_config, save_config, DEFAULT_CONFIG, save_user_known_model


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
        # Augment each missing entry with a quick source-URL hint when
        # the curated DB or pattern engine knows the file. This is done
        # synchronously and instantly (no network calls) so the user
        # sees 'comes from huggingface.co/...' right after Scan workflow.
        # Remote search (HF / CivitAI) is still deferred to 'Find sources'
        # on demand because it costs several seconds per file.
        try:
            from .sources import lookup_known_or_pattern
        except Exception:
            lookup_known_or_pattern = None
        if lookup_known_or_pattern is not None:
            for m in missing:
                try:
                    pinned = lookup_known_or_pattern(m["name"], m.get("folder"))
                except Exception:
                    pinned = None
                if pinned:
                    m["source_url"] = pinned.get("url")
                    m["source_kind"] = pinned.get("_via") or pinned.get("source")
        return web.json_response({"missing": missing})

    @routes.post("/model_downloader/search")
    async def _search(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        filename = (payload.get("filename") or "").strip()
        folder_hint = payload.get("folder")
        source_hint = payload.get("source_hint") or payload.get("raw")
        if not filename:
            return web.json_response({"error": "filename required"}, status=400)
        candidates = find_candidates(filename, folder_hint=folder_hint, source_hint=source_hint)
        return web.json_response({"filename": filename, "candidates": candidates})

    @routes.post("/model_downloader/web_search")
    async def _web_search(request: web.Request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        filename = (payload.get("filename") or "").strip()
        folder_hint = payload.get("folder")
        if not filename:
            return web.json_response({"error": "filename required"}, status=400)
        candidates = search_web_for_huggingface(filename)
        if folder_hint:
            for c in candidates:
                c.setdefault("folder", folder_hint)
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
        requested_filename = payload.get("requested_filename") or filename
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
        try:
            if requested_filename and url:
                save_user_known_model(requested_filename, {
                    "source": source or "manual",
                    "title": payload.get("title") or requested_filename,
                    "filename": requested_filename,
                    "folder": folder,
                    "url": url,
                    "size": expected_size,
                    "_reason": "learned from manual user download",
                })
        except Exception:
            pass
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

    @routes.post("/model_downloader/retry")
    async def _retry(request: web.Request):
        """Re-run a finished/errored download (same URL + destination).

        Body: {
            "id": "<job_id>",
            "huggingface_token": "<optional new token>",
        }

        If a token is supplied, it is persisted to the plugin config first so
        the retried download (and any future ones) use it for authentication.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        jid = payload.get("id")
        if not jid:
            return web.json_response({"error": "id required"}, status=400)
        new_token = (payload.get("huggingface_token") or "").strip()
        if new_token:
            from .config import load_config, save_config
            cfg = load_config()
            cfg["huggingface_token"] = new_token
            save_config(cfg)
        job = manager.requeue(jid)
        if not job:
            return web.json_response({"error": "job not found"}, status=404)
        return web.json_response({
            "job": job.to_dict(),
            "token_saved": bool(new_token),
        })

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

        # followlinks=True so symlinked model directories (common on Linux,
        # e.g. NAS mounts) are included. Inode tracking prevents infinite
        # loops from circular symlinks.
        basename_to_paths: dict[str, list[str]] = {}
        _seen_inodes: set[tuple[int, int]] = set()
        for root in _models_root_dirs():
            for dirpath, _dirs, files in os.walk(root, followlinks=True):
                try:
                    _st = os.stat(dirpath)
                    _ikey = (_st.st_dev, _st.st_ino)
                    if _ikey in _seen_inodes:
                        _dirs[:] = []
                        continue
                    _seen_inodes.add(_ikey)
                except OSError:
                    pass
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

    @routes.post("/model_downloader/dedupe_scan")
    async def _dedupe_scan(request: web.Request):
        """Walk the models tree and find duplicate files.

        Body (all optional):
            { "method": "hash" | "size_name",
              "skip_singletons": true }

        Strategy:
          1. Group all model files by (basename_lc, size_bytes). Files with
             unique (name, size) are never duplicates regardless of method.
          2. If method == "size_name": every group with 2+ files (and
             different inodes) is reported as a duplicate group. Fast
             (no file reads beyond os.stat).
             If method == "hash": SHA-256 hash every file in each
             multi-file group. Only files with the same hash are reported.
             Slow but 100% safe.
          3. Skip files that are already symlinks/hardlinks of each other
             (same dev+inode) - they are already deduped.

        Returns:
            { "groups": [
                { "hash": "abc123..."|null, "size": 12345,
                  "name": "foo.safetensors",
                  "paths": ["/a/foo.safetensors", "/b/foo.safetensors"],
                  "saving_bytes": 12345 },
                ... ],
              "method": "hash"|"size_name",
              "total_files_scanned": 1234,
              "potential_savings_bytes": 9999999 }
        """
        import os
        import hashlib
        import asyncio
        from .scanner import _models_root_dirs, _LOCAL_INDEX_EXTS

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        cfg = load_config()
        method = (payload.get("method") or cfg.get("dedupe_method") or "hash").lower()
        if method not in ("hash", "size_name"):
            method = "hash"

        # Walk the tree, group by (basename_lc, size). followlinks=True
        # but inode tracking ensures we don't traverse the same dir twice.
        groups_by_key: dict[tuple[str, int], list[str]] = {}
        seen_inodes: set[tuple[int, int]] = set()
        total_files = 0
        for root in _models_root_dirs():
            for dirpath, _dirs, files in os.walk(root, followlinks=True):
                try:
                    st = os.stat(dirpath)
                    key = (st.st_dev, st.st_ino)
                    if key in seen_inodes:
                        _dirs[:] = []
                        continue
                    seen_inodes.add(key)
                except OSError:
                    pass
                for fn in files:
                    if not fn.lower().endswith(_LOCAL_INDEX_EXTS):
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        continue
                    groups_by_key.setdefault((fn.lower(), sz), []).append(full)
                    total_files += 1

        # Filter: keep only groups with 2+ files
        candidate_groups = {k: v for k, v in groups_by_key.items() if len(v) >= 2}

        def _filter_unique_inodes(paths: list[str]) -> list[str]:
            seen: set[tuple[int, int]] = set()
            out: list[str] = []
            for p in paths:
                try:
                    s = os.stat(p)
                    ikey = (s.st_dev, s.st_ino)
                    if ikey in seen:
                        continue
                    seen.add(ikey)
                    out.append(p)
                except OSError:
                    pass
            return out

        result_groups: list[dict] = []
        total_savings = 0

        if method == "size_name":
            # Fast path: trust (name, size) match.
            for (name, size), paths in candidate_groups.items():
                deduped = _filter_unique_inodes(paths)
                if len(deduped) < 2:
                    continue
                saving = (len(deduped) - 1) * size
                total_savings += saving
                result_groups.append({
                    "hash": None,
                    "size": size,
                    "name": name,
                    "paths": deduped,
                    "saving_bytes": saving,
                })
        else:
            # Slow path: SHA-256 every candidate file in a thread executor.
            # Use the persistent hash cache (mtime+size keyed) so repeat
            # scans only re-hash files whose content actually changed.
            from . import dedupe_cache as _dc
            use_cache = bool(cfg.get("use_hash_cache", True))

            def _hash_file_with_cache(path: str) -> str | None:
                try:
                    st = os.stat(path)
                except OSError:
                    return None
                if use_cache:
                    cached = _dc.get(path, st.st_size, st.st_mtime)
                    if cached:
                        return cached
                try:
                    h = hashlib.sha256()
                    with open(path, "rb") as f:
                        while True:
                            buf = f.read(8 * 1024 * 1024)
                            if not buf:
                                break
                            h.update(buf)
                    digest = h.hexdigest()
                except Exception:
                    return None
                if use_cache and digest:
                    _dc.put(path, st.st_size, st.st_mtime, digest)
                return digest

            cache_hits = 0
            cache_misses = 0
            loop = asyncio.get_event_loop()
            for (name, size), paths in candidate_groups.items():
                deduped = _filter_unique_inodes(paths)
                if len(deduped) < 2:
                    continue

                hashes: dict[str, list[str]] = {}
                for p in deduped:
                    # Check cache before dispatching the slow hash.
                    pre_cached = None
                    if use_cache:
                        try:
                            st = os.stat(p)
                            pre_cached = _dc.get(p, st.st_size, st.st_mtime)
                        except OSError:
                            pre_cached = None
                    if pre_cached:
                        cache_hits += 1
                        digest = pre_cached
                    else:
                        cache_misses += 1
                        digest = await loop.run_in_executor(
                            None, _hash_file_with_cache, p)
                    if digest is None:
                        continue
                    hashes.setdefault(digest, []).append(p)

                for digest, hpaths in hashes.items():
                    hpaths = _filter_unique_inodes(hpaths)
                    if len(hpaths) < 2:
                        continue
                    saving = (len(hpaths) - 1) * size
                    total_savings += saving
                    result_groups.append({
                        "hash": digest,
                        "size": size,
                        "name": name,
                        "paths": hpaths,
                        "saving_bytes": saving,
                    })

            # Persist any new cache entries to disk now that the scan is done.
            if use_cache:
                _dc.flush()

        # Sort groups by savings (largest first) so the user sees high-impact
        # entries up top.
        result_groups.sort(key=lambda g: g["saving_bytes"], reverse=True)

        response_payload = {
            "groups": result_groups,
            "method": method,
            "total_files_scanned": total_files,
            "potential_savings_bytes": total_savings,
        }
        if method == "hash":
            response_payload["cache_hits"] = cache_hits
            response_payload["cache_misses"] = cache_misses
        return web.json_response(response_payload)

    @routes.post("/model_downloader/dedupe_apply")
    async def _dedupe_apply(request: web.Request):
        """Replace duplicate files with hardlinks pointing to a chosen master.

        Body:
            { "groups": [
                { "keep": "/path/to/master.safetensors",
                  "remove": ["/path/to/dup1.safetensors", ...] },
                ... ] }

        For each group:
          1. Verify keep + remove paths still exist and have the same size
             (cheap sanity check; full hash already done in dedupe_scan).
          2. For each path in `remove`:
             a. Skip if same inode as keep (already linked).
             b. Delete the file.
             c. Hardlink keep -> path.
             d. On hardlink failure (cross-fs etc.), fall back to symlink.
             e. On total failure, restore from .dedupe_backup if we can.
          3. Return per-path status.
        """
        import os
        from .linker import _try_hardlink, _try_symlink

        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        groups = payload.get("groups") or []
        results = []
        total_freed = 0
        success_count = 0

        for grp in groups:
            keep = grp.get("keep")
            remove = grp.get("remove") or []
            if not keep or not isinstance(remove, list):
                results.append({"keep": keep, "status": "skipped",
                                "reason": "missing keep or remove"})
                continue
            if not os.path.isfile(keep):
                results.append({"keep": keep, "status": "skipped",
                                "reason": "keep path not a file"})
                continue
            try:
                keep_size = os.path.getsize(keep)
                keep_st = os.stat(keep)
                keep_inode = (keep_st.st_dev, keep_st.st_ino)
            except OSError as e:
                results.append({"keep": keep, "status": "skipped",
                                "reason": f"stat keep failed: {e}"})
                continue

            for path in remove:
                if not os.path.isfile(path):
                    results.append({"keep": keep, "remove": path,
                                    "status": "skipped", "reason": "not a file"})
                    continue
                try:
                    st = os.stat(path)
                except OSError as e:
                    results.append({"keep": keep, "remove": path,
                                    "status": "skipped", "reason": f"stat: {e}"})
                    continue
                if (st.st_dev, st.st_ino) == keep_inode:
                    results.append({"keep": keep, "remove": path,
                                    "status": "already_linked"})
                    continue
                if st.st_size != keep_size:
                    results.append({"keep": keep, "remove": path,
                                    "status": "skipped",
                                    "reason": "size mismatch (file changed?)"})
                    continue

                # Delete the duplicate, then hardlink. Drop any cached
                # hash for the removed path; after the hardlink it shares
                # the keep file's content so the keep entry is what
                # matters going forward.
                try:
                    os.remove(path)
                    try:
                        from . import dedupe_cache as _dc
                        _dc.remove(path)
                    except Exception:
                        pass
                except OSError as e:
                    results.append({"keep": keep, "remove": path,
                                    "status": "error", "reason": f"remove: {e}"})
                    continue

                # Try hardlink first
                ok, err = _try_hardlink(keep, path)
                method = "hardlink"
                if not ok:
                    # Cross-filesystem? Fall back to symlink.
                    ok, err2 = _try_symlink(keep, path)
                    method = "symlink"
                    if not ok:
                        # Both failed - we already deleted the file. Try to
                        # at least copy it back so we don't leave the user
                        # with missing models.
                        try:
                            import shutil
                            shutil.copy2(keep, path)
                            results.append({"keep": keep, "remove": path,
                                            "status": "error",
                                            "reason": f"link failed; restored from copy. hardlink: {err}; symlink: {err2}"})
                        except Exception as ce:
                            results.append({"keep": keep, "remove": path,
                                            "status": "error",
                                            "reason": f"link failed AND restore failed: {err}; {err2}; copy: {ce}"})
                        continue

                results.append({"keep": keep, "remove": path,
                                "status": "linked", "method": method,
                                "freed_bytes": keep_size})
                total_freed += keep_size
                success_count += 1

        # Flush any cache mutations from the per-file _dc.remove() calls
        # so the next page reload sees a clean cache without stale entries.
        try:
            from . import dedupe_cache as _dc
            _dc.flush()
        except Exception:
            pass

        return web.json_response({
            "results": results,
            "linked_count": success_count,
            "freed_bytes": total_freed,
        })

    @routes.post("/model_downloader/clear_dedupe_cache")
    async def _clear_dedupe_cache(request: web.Request):
        """Wipe the persistent SHA-256 cache. The next 'Free Space Via
        Link' run will rehash from scratch.

        Body (optional): { "prune_only": true } - removes only entries
        whose underlying file no longer exists, keeps the rest.
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        from . import dedupe_cache as _dc
        if payload.get("prune_only"):
            removed = _dc.prune_missing()
            return web.json_response({"pruned": removed, "cleared": False})
        n = _dc.clear()
        return web.json_response({"pruned": 0, "cleared": True, "removed": n})

    @routes.get("/model_downloader/dedupe_cache_stats")
    async def _dedupe_cache_stats(request: web.Request):
        from . import dedupe_cache as _dc
        return web.json_response(_dc.stats())

    @routes.post("/model_downloader/check_path")
    async def _check_path(request: web.Request):
        """Cheap path-existence probe used by the Settings UI when the
        user adds an extra_model_paths entry. Returns whether the path
        exists, is a directory, and a rough file count for the user
        feedback line.

        Body: { "path": "C:/Other/ComfyUI/models" }
        """
        import os
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        path = (payload.get("path") or "").strip()
        if not path:
            return web.json_response({"exists": False, "is_dir": False,
                                       "abs": "", "file_count": 0,
                                       "reason": "empty"})
        try:
            ap = os.path.abspath(path)
        except Exception as e:
            return web.json_response({"exists": False, "is_dir": False,
                                       "abs": path, "file_count": 0,
                                       "reason": f"abspath: {e}"})
        exists = os.path.exists(ap)
        is_dir = os.path.isdir(ap)
        file_count = 0
        if is_dir:
            # Cheap probe: just count entries at top level (don't recurse).
            try:
                file_count = sum(1 for _ in os.scandir(ap))
            except Exception:
                file_count = 0
        return web.json_response({
            "exists": exists,
            "is_dir": is_dir,
            "abs": ap,
            "file_count": file_count,
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
            # extra_model_paths must be a list of non-empty strings.
            # Silently drop garbage entries (None, ints, empty strings)
            # so a single malformed UI input doesn't poison the config.
            if k == "extra_model_paths":
                if not isinstance(value, list):
                    ignored.append(k)
                    continue
                value = [
                    s.strip() for s in value
                    if isinstance(s, str) and s.strip()
                ]
            cfg[k] = value
        save_config(cfg)
        return web.json_response({"ok": True, "ignored": ignored})

    app.add_routes(routes)
    _REGISTERED = True
    print("[ModelDownloader] HTTP routes registered.")
