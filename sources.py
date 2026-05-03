"""HuggingFace + CivitAI search providers."""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import urllib.error
import json as _json
from typing import Any

from .config import load_config, load_known_models

USER_AGENT = "ComfyUI-ModelDownloader/1.0"

# Tokens that show up in almost every model filename and tell HF nothing.
# We avoid using these as standalone search queries.
_STOPWORDS = {
    "model", "models", "lora", "loras", "ckpt", "checkpoint", "vae",
    "controlnet", "control", "net", "clip", "unet", "diffusion",
    "embedding", "embeddings", "upscale", "upscaler", "ema", "fp8", "fp16",
    "fp32", "bf16", "int8", "int4", "q4", "q5", "q6", "q8", "gguf",
    "safetensors", "pruned", "pt", "pth", "bin", "rank", "step", "steps",
    "low", "high", "noise", "base", "refiner", "edit", "main", "final",
    "version", "v1", "v2", "v3", "test", "release", "merged",
}

# In-memory cache: filename -> (timestamp, results)
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL_S = 300  # 5 minutes


def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def _hf_headers() -> dict[str, str]:
    cfg = load_config()
    headers = {}
    if cfg.get("huggingface_token"):
        headers["Authorization"] = f"Bearer {cfg['huggingface_token']}"
    return headers


# ---------- Known DB ----------

def lookup_known(filename: str) -> dict | None:
    db = load_known_models()
    if filename in db:
        entry = dict(db[filename])
        entry["filename"] = filename
        entry.setdefault("title", filename)
        return entry
    # Case-insensitive match
    lc = filename.lower()
    for k, v in db.items():
        if k.lower() == lc:
            entry = dict(v)
            entry["filename"] = k
            entry.setdefault("title", k)
            return entry
    return None


# ---------- Filename tokenization ----------

def _tokenize_filename(filename: str) -> list[str]:
    """Extract distinctive search tokens from a model filename.

    Removes file extension, splits on common separators, filters out pure
    numbers and stopwords, and orders by 'distinctiveness' (length + presence
    of letters+digits) so the most useful queries come first.
    """
    base = filename.rsplit(".", 1)[0]
    raw = re.split(r"[_\-.\s/]+", base)
    keep: list[str] = []
    for t in raw:
        if not t:
            continue
        if len(t) < 3:
            continue
        if t.isdigit():
            continue
        if re.fullmatch(r"\d+\w?", t):  # things like "4step" -> drop trailing-digit-only? No, keep if has letters
            # Re-check: this regex matches "4step" because of \w. Only drop if all-digit prefix + 1 letter.
            # Actually allow it; it might be useful.
            pass
        if t.lower() in _STOPWORDS:
            continue
        keep.append(t)

    def score(t: str) -> tuple[int, int, str]:
        # Higher = better. Prefer longer tokens with both letters AND digits
        # (those are usually distinctive identifiers like "lightx2v", "A14b").
        has_letters = any(c.isalpha() for c in t)
        has_digits = any(c.isdigit() for c in t)
        mix_bonus = 1 if (has_letters and has_digits) else 0
        return (-len(t), -mix_bonus, t.lower())

    return sorted(keep, key=score)


def _build_hf_queries(filename: str) -> list[str]:
    """Return a ranked list of HF search queries to try for this filename."""
    tokens = _tokenize_filename(filename)
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str):
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    # Always try the full filename first (works for famous files)
    add(filename)
    # Filename without extension
    add(filename.rsplit(".", 1)[0])
    # Top single tokens
    for t in tokens[:3]:
        add(t)
    # Pairs of distinctive tokens
    for i in range(min(3, len(tokens))):
        for j in range(i + 1, min(5, len(tokens))):
            add(f"{tokens[i]} {tokens[j]}")
    return queries[:8]  # cap to keep latency reasonable


def _build_hf_fulltext_queries(filename: str) -> list[str]:
    """Queries optimised for HF's full-text README search.

    The full-text index splits on word boundaries, so passing exact tokens
    works better than passing the dashed/underscored full filename. Hyphens
    and underscores are converted to spaces.
    """
    base = filename.rsplit(".", 1)[0]
    tokens = _tokenize_filename(filename)

    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str):
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    # 1. Just the basename without extension - this catches files mentioned
    #    verbatim in READMEs (e.g. "film_net_fp16" -> Comfy-Org/frame_interpolation)
    add(base)
    # 2. Same but with separators turned into spaces
    add(base.replace("_", " ").replace("-", " ").replace(".", " "))
    # 3. The full filename including extension
    add(filename)
    # 4. Top tokens joined by spaces (good for partial matches)
    if len(tokens) >= 2:
        add(" ".join(tokens[:3]))
    if len(tokens) >= 3:
        add(" ".join(tokens[:4]))
    return queries[:5]


def _hf_fulltext_search(query: str, limit: int = 20) -> list[dict]:
    """Hit HF's full-text README search and return list of candidate repo IDs.

    Returns a list of dicts: {"repo": "owner/name", "score": <int>}.
    Higher 'score' means the repo appeared higher in HF's relevance ranking.
    """
    url = (
        f"https://huggingface.co/api/search/full-text"
        f"?q={urllib.parse.quote(query)}&type=model&limit={limit}"
    )
    try:
        data = _http_get_json(url, headers=_hf_headers(), timeout=12)
    except Exception:
        return []
    hits = data.get("hits") or []
    out = []
    for i, h in enumerate(hits):
        owner = h.get("repoOwner")
        name = h.get("repoName")
        if not owner or not name:
            continue
        out.append({
            "repo": f"{owner}/{name}",
            "score": len(hits) - i,  # earlier = higher
        })
    return out


def _hf_get_repo_meta(repo_id: str) -> dict | None:
    """Fetch repo metadata (downloads, gated, ...) - small JSON, fast."""
    try:
        return _http_get_json(
            f"https://huggingface.co/api/models/{repo_id}",
            headers=_hf_headers(),
            timeout=8,
        )
    except Exception:
        return None


# Well-known umbrella repos that host many ComfyUI models, often nested in
# subfolders, where the filename does NOT match the repo name AND the README
# doesn't list every file. These are checked as a last-resort fallback.
# Token-hint = a substring of the filename that has to appear before we
# bother probing this repo (cheap heuristic to avoid blasting the API).
_HF_FALLBACK_REPOS: list[tuple[str, tuple[str, ...]]] = [
    # (repo_id, hint-tokens that must match somewhere in the filename)
    ("Kijai/WanVideo_comfy",         ("wan", "rcm")),
    ("Kijai/WanVideo_comfy_fp8_scaled", ("wan",)),
    ("Kijai/HunyuanVideo_comfy",     ("hunyuan",)),
    ("Kijai/LTX2.3_comfy",           ("ltx",)),
    ("Kijai/LTX2-IC-LoRAs",          ("ltx",)),
    ("Kijai/Z-Image_comfy_fp8_scaled", ("z-image", "zimage")),
    ("Comfy-Org/frame_interpolation", ("film", "rife", "frame")),
    ("Comfy-Org/flux1-dev",          ("flux",)),
    ("comfyanonymous/flux_text_encoders", ("clip_l", "t5xxl")),
    ("lightx2v/Wan2.2-Distill-Loras", ("wan", "lightx2v", "distill")),
    ("lightx2v/Wan2.2-Distill-Models", ("wan", "lightx2v", "distill")),
    ("lllyasviel/ControlNet-v1-1",   ("control_",)),
]


def _hf_check_fallback_repos(filename: str) -> list[tuple[str, str, int | None]]:
    """Probe the well-known umbrella repos for a file with this exact basename.

    Returns a list of (repo_id, file_path, file_size). Cheap because each
    relevant repo only needs a single tree-API call, and we filter by hint
    tokens before doing even that.
    """
    fn_lc = filename.lower()
    found: list[tuple[str, str, int | None]] = []
    for repo_id, hints in _HF_FALLBACK_REPOS:
        if hints and not any(h in fn_lc for h in hints):
            continue
        hit = _hf_find_file_in_repo(repo_id, filename)
        if hit:
            found.append((repo_id, hit["path"], hit.get("size")))
    return found


# ---------- HuggingFace ----------

def _hf_list_repo_files(repo_id: str) -> list[dict]:
    """Return the list of files in the main branch of a HF repo."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
    try:
        return _http_get_json(url, headers=_hf_headers(), timeout=10)
    except Exception:
        return []


def _hf_find_file_in_repo(repo_id: str, filename: str) -> dict | None:
    """Look for a file matching `filename` (basename) inside a HF repo's tree."""
    target = filename.lower()
    files = _hf_list_repo_files(repo_id)
    for f in files:
        path = (f.get("path") or "").replace("\\", "/")
        if not path:
            continue
        bn = path.rsplit("/", 1)[-1]
        if bn.lower() == target or path.lower() == target:
            return {
                "path": path,
                "size": f.get("size"),
            }
    return None


def _repo_seems_relevant(repo_id: str, filename_tokens: set[str]) -> bool:
    """
    Cheap heuristic to decide if a repo is worth a tree-lookup. A repo is
    'relevant' if its id (case-insensitive) shares at least one distinctive
    token with the filename. This filters out unrelated repos that share an
    author name (e.g. all of `lightx2v/Qwen-*` when searching for a Wan2.2 lora).
    """
    if not filename_tokens:
        return True
    repo_lc = repo_id.lower().replace("-", " ").replace("_", " ").replace("/", " ").replace(".", " ")
    repo_words = set(repo_lc.split())
    return any(t.lower() in repo_words or t.lower() in repo_lc for t in filename_tokens)


def search_huggingface(filename: str, limit_per_query: int = 20) -> list[dict]:
    """Search HuggingFace for the given filename, trying multiple strategies.

    Strategy (most to least specific):
      Phase 1: Repo-name search via /api/models?search=...
               (HF only matches repo names + tags; works for famous files)
      Phase 2: Full-text README search via /api/search/full-text
               (catches repos that REFERENCE the filename in their README,
                even when the repo name is unrelated - e.g. "film_net_fp16"
                -> Comfy-Org/frame_interpolation)
      Phase 3: Tree-lookups on the candidate repos collected above. The
               tree response confirms the file actually exists and gives
               us the exact path inside the repo.
    """
    tokens = set(_tokenize_filename(filename))
    target = filename.lower()
    headers = _hf_headers()

    # Collect repo candidates from both endpoints.
    # Each entry: {"repo": str, "via": "search"|"fulltext", "siblings": [...] | None,
    #              "downloads": int|None, "gated": bool, "_query": str}
    candidates: list[dict] = []
    seen_repos: set[str] = set()

    def add_candidate(repo_id, via, query, siblings=None, downloads=None, gated=False):
        if not repo_id or repo_id in seen_repos:
            return
        seen_repos.add(repo_id)
        candidates.append({
            "repo": repo_id,
            "via": via,
            "_query": query,
            "siblings": siblings,
            "downloads": downloads,
            "gated": gated,
        })

    # ----- Phase 1: repo-name search -----
    for q in _build_hf_queries(filename):
        url = (
            f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}"
            f"&limit={limit_per_query}&full=true"
        )
        try:
            data = _http_get_json(url, headers=headers, timeout=12)
        except Exception:
            continue
        for repo in data:
            add_candidate(
                repo.get("id") or repo.get("modelId"),
                via="search",
                query=q,
                siblings=repo.get("siblings"),
                downloads=repo.get("downloads", 0),
                gated=bool(repo.get("gated")),
            )

    # ----- Phase 2: full-text README search -----
    # This is what catches files like film_net_fp16 (mentioned in README of
    # an unrelated-looking repo) and rCM-style files (cross-referenced).
    for q in _build_hf_fulltext_queries(filename):
        for hit in _hf_fulltext_search(q, limit=20):
            add_candidate(hit["repo"], via="fulltext", query=q)

    # ----- Phase 3: confirm via siblings / tree lookups -----
    # Order matters here. We process in two passes:
    #   3a. Cheap pass: every candidate's `siblings` array (already in memory)
    #   3b. Tree pass: only for candidates where siblings didn't match.
    # Within 3b, full-text hits are tried FIRST because they came from a
    # README that literally mentions the filename - very high signal. Then
    # we fall back to search-based hits whose repo name shares a token with
    # the filename. Repos that are clearly unrelated (e.g. all the "film"
    # name-search hits when looking for "film_net_fp16") are skipped to keep
    # the API budget for promising candidates.
    results: list[dict] = []

    # 3a. Siblings pass
    confirmed: list[tuple[dict, str, int | None]] = []  # (cand, file_path, file_size)
    unconfirmed: list[dict] = []
    for c in candidates:
        file_path: str | None = None
        file_size: int | None = None
        for s in c.get("siblings") or []:
            rfn = (s.get("rfilename") or "").replace("\\", "/")
            if rfn.lower() == target or rfn.lower().endswith("/" + target):
                file_path = rfn
                file_size = s.get("size")
                break
        if file_path is not None:
            confirmed.append((c, file_path, file_size))
        else:
            unconfirmed.append(c)

    # 3b. Tree-lookup pass, ordered by trust:
    #     full-text hits (highest), then relevant search hits, then nothing.
    def tree_priority(c):
        if c["via"] == "fulltext":
            return 0  # try first
        if _repo_seems_relevant(c["repo"], tokens):
            return 1
        return 2  # skip (won't be tried)

    unconfirmed.sort(key=tree_priority)

    MAX_TREE_LOOKUPS = 12
    tree_lookups_done = 0
    for c in unconfirmed:
        if len(confirmed) >= 5:
            break
        if tree_lookups_done >= MAX_TREE_LOOKUPS:
            break
        if tree_priority(c) >= 2:
            continue  # not worth trying
        tree_lookups_done += 1
        hit = _hf_find_file_in_repo(c["repo"], filename)
        if hit:
            confirmed.append((c, hit["path"], hit.get("size")))

    # ----- Phase 4: well-known fallback repos -----
    # If neither repo-search nor README-fulltext found the file, probe a
    # short list of umbrella repos that we know host many ComfyUI files in
    # subdirectories with no README listing. This catches files like
    # "Wan22-I2V-A14B-LOW-rCM1_0_lora_rank_64_bf16.safetensors" living in
    # Kijai/WanVideo_comfy/LoRAs/rCM/ - the filename has zero match against
    # the repo name and isn't mentioned in any README.
    if not confirmed:
        for repo_id, file_path, file_size in _hf_check_fallback_repos(filename):
            if repo_id in seen_repos:
                continue
            seen_repos.add(repo_id)
            synthetic = {
                "repo": repo_id,
                "via": "fallback",
                "_query": "(known repo)",
                "siblings": None,
                "downloads": None,
                "gated": False,
            }
            confirmed.append((synthetic, file_path, file_size))
            if len(confirmed) >= 5:
                break

    # Build result entries from `confirmed`
    for c, file_path, file_size in confirmed:
        if len(results) >= 5:
            break
        repo_id = c["repo"]

        # Fill in download count for fulltext candidates by hitting the meta
        # endpoint once. Cheap but gives us proper sorting.
        downloads = c.get("downloads")
        gated = c.get("gated")
        if downloads is None:
            meta = _hf_get_repo_meta(repo_id)
            if meta:
                downloads = meta.get("downloads", 0)
                gated = bool(meta.get("gated"))

        results.append({
            "source": "huggingface",
            "repo": repo_id,
            "filename": file_path.rsplit("/", 1)[-1],
            "title": f"{repo_id}/{file_path}",
            "url": f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(file_path)}",
            "size": file_size,
            "downloads": downloads or 0,
            "gated": bool(gated),
            "_query": c.get("_query"),
            "_via": c["via"],
        })

    # Deduplicate (same repo+path)
    deduped: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for r in results:
        k = (r["repo"], r["filename"])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(r)

    deduped.sort(key=lambda r: r.get("downloads", 0), reverse=True)
    return deduped


# ---------- CivitAI ----------

def search_civitai(filename: str, limit: int = 5) -> list[dict]:
    """Search CivitAI by filename. CivitAI's search uses 'query' on names."""
    cfg = load_config()
    headers = {}
    token = cfg.get("civitai_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base = filename.rsplit(".", 1)[0]
    # Try several queries: full base, then top distinctive tokens
    tokens = _tokenize_filename(filename)
    queries = [base]
    for t in tokens[:2]:
        if t.lower() not in (q.lower() for q in queries):
            queries.append(t)

    results: list[dict] = []
    target_lc = filename.lower()
    base_lc = base.lower()
    seen_files: set[tuple[str, str]] = set()

    for q in queries:
        if len(results) >= 5:
            break
        url = f"https://civitai.com/api/v1/models?limit={limit}&query={urllib.parse.quote(q)}"
        try:
            data = _http_get_json(url, headers=headers, timeout=12)
        except Exception:
            continue

        items = data.get("items") if isinstance(data, dict) else None
        if not items:
            continue

        for m in items:
            name = m.get("name", "")
            m_type = (m.get("type") or "").lower()
            for v in m.get("modelVersions", []) or []:
                for f in v.get("files", []) or []:
                    fname = f.get("name", "")
                    if not fname:
                        continue
                    match = (
                        fname.lower() == target_lc
                        or fname.lower().endswith("/" + target_lc)
                        or base_lc in fname.lower()
                    )
                    if not match:
                        continue
                    key = (str(m.get("id")), fname)
                    if key in seen_files:
                        continue
                    seen_files.add(key)

                    download_url = f.get("downloadUrl") or v.get("downloadUrl")
                    if not download_url:
                        continue
                    folder = {
                        "checkpoint": "checkpoints",
                        "lora": "loras",
                        "locon": "loras",
                        "lycoris": "loras",
                        "textualinversion": "embeddings",
                        "hypernetwork": "hypernetworks",
                        "vae": "vae",
                        "controlnet": "controlnet",
                        "upscaler": "upscale_models",
                    }.get(m_type, "checkpoints")

                    results.append({
                        "source": "civitai",
                        "title": f"{name} - {v.get('name', '')}",
                        "filename": fname,
                        "folder": folder,
                        "url": download_url,
                        "size": int((f.get("sizeKB") or 0) * 1024),
                        "downloads": (m.get("stats") or {}).get("downloadCount", 0),
                        "needs_token": True,
                    })

    results.sort(key=lambda r: r.get("downloads", 0), reverse=True)
    return results


# ---------- Combined ----------

def find_candidates(filename: str, folder_hint: str | None = None) -> list[dict]:
    """Return ranked download candidates for a filename, with caching."""
    # Cache lookup
    cached = _search_cache.get(filename)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_S:
        out = list(cached[1])
        if folder_hint:
            for c in out:
                c.setdefault("folder", folder_hint)
        return out

    out: list[dict] = []
    known = lookup_known(filename)
    if known:
        out.append({
            "source": known.get("source", "known"),
            "title": known.get("title", filename),
            "filename": filename,
            "folder": known.get("folder", folder_hint or "checkpoints"),
            "url": known["url"],
            "size": known.get("size"),
            "gated": known.get("gated", False),
            "preferred": True,
        })

    try:
        out.extend(search_huggingface(filename))
    except Exception:
        pass
    try:
        out.extend(search_civitai(filename))
    except Exception:
        pass

    if folder_hint:
        for c in out:
            c.setdefault("folder", folder_hint)

    _search_cache[filename] = (time.time(), [dict(c) for c in out])
    return out
