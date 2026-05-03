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
from .patterns import (
    filename_aliases,
    is_upstream_alias,
    lookup_pattern,
    normalise_filename,
)

USER_AGENT = "ComfyUI-ModelDownloader/1.0"

# Tokens we filter out as STANDALONE search queries because they're far
# too generic to land us in the right repo on their own. We deliberately
# keep them in token COMBINATIONS though - 'stable-diffusion-xl-base'
# (which contains stopwords like 'base') is a perfectly good query.
_STOPWORDS_ALONE = {
    # File-extension-noise and quantisation suffixes
    "safetensors", "ckpt", "pt", "pth", "bin", "gguf", "sft",
    "fp8", "fp16", "fp32", "bf16", "int8", "int4",
    "q4", "q5", "q6", "q8",
    "ema", "pruned", "merged", "final",
    # Single-letter / two-letter noise
    "v1", "v2", "v3", "v4", "v5",
}

# HuggingFace authors / orgs that publish canonical model files. Repos
# from these accounts are heavily preferred over community mirrors when
# the same file appears in multiple repos. The bonus is added to a
# repo's effective download count for ranking purposes - an org with
# 100 declared downloads from this list outranks an unknown repo with
# 100k downloads, because we trust the canonical source more.
_TRUSTED_HF_AUTHORS: set[str] = {
    "stabilityai",
    "black-forest-labs",
    "runwayml",
    "openai",
    "laion",
    "comfyanonymous",
    "Comfy-Org",
    "lllyasviel",
    "h94",
    "latent-consistency",
    "ai-forever",
    "Kijai",
    "wangfuyun",
    "guoyww",
    "depth-anything",
    "LiheYoung",
    "Kim2091",
    "google",
    "facebook",
    "meta-llama",
    "mistralai",
    "PixArt-alpha",
    "TencentARC",
    "InstantX",
    "xinsir",
    "diffusers",
}


def _trusted_author(repo_id: str) -> bool:
    """True if the repo is hosted by an org we treat as canonical."""
    if not repo_id or "/" not in repo_id:
        return False
    return repo_id.split("/", 1)[0] in _TRUSTED_HF_AUTHORS

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
    """Look up `filename` in the curated DB.

    Match priority (first hit wins):
      1. Exact case-sensitive match against an entry's primary key
      2. Exact case-sensitive match against an entry's 'aliases' list
      3. Normalised match (lower-cased, hyphen<->underscore equivalent)
         against either the primary key or any alias

    Underscore _section_/_comment_ keys are skipped automatically.

    The returned entry's "filename" field is set to the user's exact
    spelling so the file ends up on disk under the workflow's name -
    not the canonical key from the DB.
    """
    db = load_known_models()
    if not db:
        return None

    # Build two indexes: case-sensitive and normalised. Each maps the
    # match-key to (primary_key, raw_entry). Skip metadata keys.
    cs_index: dict[str, tuple[str, dict]] = {}
    norm_index: dict[str, tuple[str, dict]] = {}
    for k, v in db.items():
        if not isinstance(v, dict) or k.startswith("_"):
            continue
        cs_index.setdefault(k, (k, v))
        norm_index.setdefault(normalise_filename(k), (k, v))
        for alias in v.get("aliases") or []:
            cs_index.setdefault(alias, (k, v))
            norm_index.setdefault(normalise_filename(alias), (k, v))

    # Pass 1: case-sensitive
    if filename in cs_index:
        primary, raw = cs_index[filename]
        entry = dict(raw)
        entry.pop("aliases", None)
        entry["filename"] = filename
        entry.setdefault("title", primary)
        return entry

    # Pass 2: normalised (handles case + hyphen/underscore)
    for candidate in filename_aliases(filename):
        norm = normalise_filename(candidate)
        if norm in norm_index:
            primary, raw = norm_index[norm]
            entry = dict(raw)
            entry.pop("aliases", None)
            entry["filename"] = filename
            entry.setdefault("title", primary)
            return entry
    return None


def lookup_known_or_pattern(filename: str, folder_hint: str | None = None) -> dict | None:
    """Combined lookup: curated DB first, then pattern rules.

    Returns a candidate dict ready to be inserted into find_candidates'
    output, or None if no rule matched.
    """
    # 1. Curated DB - hand-picked, always wins
    known = lookup_known(filename)
    if known:
        return {
            "source":   known.get("source", "known"),
            "title":    known.get("title", filename),
            "filename": filename,
            "folder":   known.get("folder", folder_hint or "checkpoints"),
            "url":      known["url"],
            "size":     known.get("size"),
            "gated":    known.get("gated", False),
            "preferred": True,
            "downloads": known.get("downloads", 0),
            "_via":     "known",
        }
    # 2. Pattern rules - generic, covers families of filenames
    pat = lookup_pattern(filename)
    if pat:
        if folder_hint and not pat.get("folder"):
            pat["folder"] = folder_hint
        return pat
    return None


# ---------- Filename tokenization ----------

def _split_tokens(filename: str) -> list[str]:
    """Split a filename into ordered tokens, preserving original order
    and case. Splits on common filename separators.

    Unlike _tokenize_filename, this keeps EVERY token (no stopword
    filter) because some queries need consecutive token spans like
    'sd_xl_base' or 'stable-diffusion-xl-base'.
    """
    base = filename.rsplit(".", 1)[0]
    raw = re.split(r"[_\-.\s/]+", base)
    return [t for t in raw if t]


def _tokenize_filename(filename: str) -> list[str]:
    """Extract distinctive search tokens for use as STANDALONE queries.

    Filters out short / pure-digit / safetensors-noise tokens and orders
    by distinctiveness (length + letter+digit mix).
    """
    keep: list[str] = []
    for t in _split_tokens(filename):
        if len(t) < 3:
            continue
        if t.isdigit():
            continue
        if t.lower() in _STOPWORDS_ALONE:
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
    """Return a ranked list of HF search queries to try for this filename.

    Strategy:
      1. The full filename (with and without extension) - catches files
         hosted in repos named after the filename verbatim.
      2. ALL consecutive token spans of length >= 2, joined with hyphen.
         For 'sd_xl_base_1.0.safetensors' this yields:
           'sd-xl-base-1-0', 'sd-xl-base-1', 'sd-xl-base',
           'sd-xl', 'xl-base-1-0', ..., 'base-1-0',
         which means 'stable-diffusion-xl-base' style repo names get
         hit by 'sd-xl-base' or 'xl-base'.
      3. Same spans joined with underscore (for repos that follow the
         filename's separator style).
      4. Distinctive standalone tokens as a fallback.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str):
        q = q.strip()
        if not q:
            return
        # Same query in different separators counts as the same
        norm = q.lower().replace("_", "-").replace(" ", "-")
        if norm in seen:
            return
        seen.add(norm)
        queries.append(q)

    add(filename)
    add(filename.rsplit(".", 1)[0])

    # Consecutive token spans, longest first (more distinctive)
    parts = _split_tokens(filename)
    spans: list[list[str]] = []
    for span_len in range(min(len(parts), 5), 1, -1):  # 5..2 inclusive
        for start in range(0, len(parts) - span_len + 1):
            spans.append(parts[start:start + span_len])
    for span in spans:
        # Skip spans that are entirely stopwords / noise
        if all(t.lower() in _STOPWORDS_ALONE or t.isdigit() for t in span):
            continue
        add("-".join(span))
        add("_".join(span))

    # Distinctive standalone tokens as last resort
    for t in _tokenize_filename(filename)[:3]:
        add(t)

    # Cap to keep latency reasonable. We do 2 HTTP calls per query
    # (with + without sort), so 8 queries = 16 round-trips worst case.
    return queries[:10]


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


# Per-author hint tokens. When the filename contains one of these
# substrings (case-insensitive), the matching trusted author is probed
# directly via /api/models?author=...&full=true. This is what reaches
# canonical repos that HF's filename search can't surface.
#
# Keys are case-sensitive HF org names; values are lowercase substrings
# we look for in the filename.
_AUTHOR_FILENAME_HINTS: dict[str, tuple[str, ...]] = {
    "stabilityai":       ("sd_xl", "sdxl_", "sd_xl_", "stable_diffusion", "stable-diffusion", "sd-vae", "sdvae", "sd-turbo", "vae-ft-mse", "vae_ft_mse"),
    "black-forest-labs": ("flux", "ae.safe", "ae.bin"),
    "runwayml":          ("v1-5-pruned", "v1_5_pruned", "stable-diffusion-v1-5"),
    "comfyanonymous":    ("clip_l.", "clip_g.", "t5xxl", "flux_text"),
    "Comfy-Org":         ("sigclip", "frame_interp", "rife", "v1-5-pruned"),
    "lllyasviel":        ("control_", "controlnet", "annotator", "realesrgan_x4plus.pth"),
    "h94":               ("ip-adapter", "ip_adapter", "image_encoder", "clip-vit-h-14-laion", "clip-vit-bigg"),
    "latent-consistency":("lcm-lora", "lcm_lora"),
    "ai-forever":        ("realesrgan_x2", "realesrgan_x4", "realesrgan_x8", "real-esrgan"),
    "wangfuyun":         ("animatelcm",),
    "guoyww":            ("mm_sd_v", "mm_sdxl", "v3_sd15", "v2_lora_"),
    "depth-anything":    ("depth_anything_v2", "depth-anything-v2"),
    "LiheYoung":         ("depth_anything_vit",),
    "Kim2091":           ("ultrasharp",),
    "Kijai":             ("kijai", "wanvideo", "ltx2", "z-image", "hunyuanvideo"),
}


def _authors_for_filename(filename: str, max_authors: int = 3) -> list[str]:
    """Return up to `max_authors` trusted-author names whose hint tokens
    appear in the filename. Returned in descending hint-match-count order
    so the most specific author is tried first."""
    fn_lc = filename.lower()
    scored: list[tuple[int, str]] = []
    for author, hints in _AUTHOR_FILENAME_HINTS.items():
        score = sum(1 for h in hints if h in fn_lc)
        if score > 0:
            scored.append((score, author))
    scored.sort(reverse=True)
    return [a for _, a in scored[:max_authors]]


# Cache of {author: (timestamp, [repos_with_siblings])}. Listing all of
# stabilityai's 100 repos is expensive enough to keep around for a while.
_author_repo_cache: dict[str, tuple[float, list[dict]]] = {}
_AUTHOR_CACHE_TTL_S = 600  # 10 minutes


def _hf_list_author_repos(author: str, limit: int = 100) -> list[dict]:
    """List up to `limit` repos by an author, sorted by downloads desc.
    Each repo dict carries its `siblings` array so we can match files
    without a follow-up request. Cached for _AUTHOR_CACHE_TTL_S seconds."""
    cached = _author_repo_cache.get(author)
    if cached and (time.time() - cached[0]) < _AUTHOR_CACHE_TTL_S:
        return cached[1]
    url = (
        f"https://huggingface.co/api/models?author={urllib.parse.quote(author)}"
        f"&sort=downloads&direction=-1&limit={limit}&full=true"
    )
    try:
        repos = _http_get_json(url, headers=_hf_headers(), timeout=15)
    except Exception:
        return []
    _author_repo_cache[author] = (time.time(), repos)
    return repos


def _hf_probe_trusted_authors(filename: str, max_authors: int = 3) -> list[dict]:
    """For each trusted author whose hint tokens match the filename, list
    their top repos and check whether any has the file in its `siblings`.

    This is what catches canonical repos that HF's normal /api/models
    ?search= endpoint never returns - filename-based search returns only
    repos whose NAME contains the filename, but stabilityai's repos are
    named 'stable-diffusion-xl-base-1.0', not 'sd_xl_base_1.0.safetensors'.

    Returns a list of candidate dicts ready to be fed into the same
    confirmed-list pipeline as repo-search results. Each carries the
    sibling array so the calling code's siblings-pass picks them up
    without an extra round-trip.
    """
    target = filename.lower()
    out: list[dict] = []
    for author in _authors_for_filename(filename, max_authors=max_authors):
        repos = _hf_list_author_repos(author, limit=100)
        for repo in repos:
            repo_id = repo.get("id") or repo.get("modelId")
            if not repo_id:
                continue
            siblings = repo.get("siblings") or []
            # Look for the filename anywhere in the sibling list (top
            # level OR in a subfolder like models/image_encoder/).
            for s in siblings:
                rfn = (s.get("rfilename") or "").replace("\\", "/")
                rfn_base = rfn.rsplit("/", 1)[-1]
                if rfn.lower() == target or rfn_base.lower() == target:
                    out.append({
                        "repo": repo_id,
                        "via": "trusted-author",
                        "_query": f"(author:{author})",
                        "siblings": siblings,
                        "downloads": repo.get("downloads", 0),
                        "gated": bool(repo.get("gated")),
                    })
                    break  # one match per repo is enough
    return out


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
    tokens before doing even that. Upstream-alias matching is enabled
    because we know the repo is canonically the right place for this
    type of model.
    """
    fn_lc = filename.lower()
    found: list[tuple[str, str, int | None]] = []
    for repo_id, hints in _HF_FALLBACK_REPOS:
        if hints and not any(h in fn_lc for h in hints):
            continue
        hit = _hf_find_file_in_repo(repo_id, filename, accept_upstream_alias=True)
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


def _hf_find_file_in_repo(
    repo_id: str,
    filename: str,
    accept_upstream_alias: bool = False,
) -> dict | None:
    """Look for a file matching `filename` (basename) inside a HF repo's tree.

    Two passes:

      1. Strict pass: look for an exact filename match (case-insensitive,
         normalised so hyphens and underscores are interchangeable).
      2. Optional alias pass: if `accept_upstream_alias` is True and the
         strict pass found nothing, accept any of the well-known
         "single weights file" filenames (model.safetensors,
         pytorch_lora_weights.safetensors, ...) as the answer. This is
         what lets us resolve workflow names like
         ``LCM_LoRA_Weights_SD15.safetensors`` to a HF repo whose only
         file is ``pytorch_lora_weights.safetensors``.

    The returned dict's ``path`` is the path inside the repo, ``size``
    the byte count, and ``alias`` is True iff we matched via the alias
    pass. Callers can use ``alias`` to decide whether to rename the
    file on download.
    """
    norm_target = normalise_filename(filename)
    files = _hf_list_repo_files(repo_id)

    # Strict pass
    for f in files:
        path = (f.get("path") or "").replace("\\", "/")
        if not path:
            continue
        bn = path.rsplit("/", 1)[-1]
        if normalise_filename(bn) == norm_target or path.lower() == filename.lower():
            return {
                "path": path,
                "size": f.get("size"),
                "alias": False,
            }

    # Alias pass - only if the caller opted in. We require that the
    # repo contains EXACTLY ONE matching alias file at the top level (or
    # in the conventional `models/`-like subdir) so we don't pick the
    # wrong one when a repo bundles many.
    if accept_upstream_alias:
        candidates = []
        for f in files:
            path = (f.get("path") or "").replace("\\", "/")
            if not path:
                continue
            bn = path.rsplit("/", 1)[-1]
            if is_upstream_alias(bn):
                # Prefer top-level files (no slashes in path)
                depth = path.count("/")
                candidates.append((depth, path, f.get("size")))
        if candidates:
            # Pick the shallowest-depth candidate; ties broken by path.
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, path, size = candidates[0]
            return {
                "path": path,
                "size": size,
                "alias": True,
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


def search_huggingface(filename: str, limit_per_query: int = 1000) -> list[dict]:
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
    # One HTTP call per query is enough. Empirically:
    #   limit=1000   = HF's per-query result cap
    #   sort=downloads&direction=-1
    #                = applies even when search= is given, returns the
    #                  same set of repos but ordered most-popular first
    #   full=true    = include siblings array (the file listing per
    #                  repo) so the siblings-pass below can match
    #                  without a follow-up tree call. Response is
    #                  ~0.65 MB / 0.2-0.5s for typical model queries.
    # Combining all three means a single round-trip per query gives us
    # popularity-ordered repos with file lists ready to scan.
    for q in _build_hf_queries(filename):
        url = (
            f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}"
            f"&limit={limit_per_query}&sort=downloads&direction=-1&full=true"
        )
        try:
            data = _http_get_json(url, headers=headers, timeout=20)
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

    # ----- Phase 1.5: trusted-author probe -----
    # HF's filename search only returns repos whose NAME contains the
    # filename string. For canonical files like 'sd_xl_base_1.0.safetensors'
    # the only matches are tiny hobby mirrors (youneeds/sd_xl_base_1.0...
    # etc.) - the real stabilityai repo isn't there because its name is
    # 'stable-diffusion-xl-base-1.0'. So when the filename hints at a
    # known org (stabilityai, black-forest-labs, comfyanonymous, ...),
    # we list ALL their repos directly and check each repo's siblings
    # for the file. Cached for 10 minutes.
    for trusted in _hf_probe_trusted_authors(filename, max_authors=3):
        # Inject directly into candidates - they already have their
        # siblings arrays so Phase 3a will accept them immediately.
        if trusted["repo"] not in seen_repos:
            seen_repos.add(trusted["repo"])
            candidates.append(trusted)

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
    #
    # IMPORTANT: HF's /api/models?search= endpoint returns repos sorted
    # by name-match quality, NOT by popularity. We pre-sort the
    # candidate list by:
    #   1. Trusted-author bonus (stabilityai, black-forest-labs, ...)
    #      Repos from canonical orgs come first regardless of download
    #      count, so the official source beats popular community mirrors.
    #   2. Download count, descending. Highest downloads first within
    #      the same author tier.
    # Only the first 5 confirmed hits are emitted, so this dramatically
    # reduces the chance of returning a wrong-but-plausible top result.
    def _candidate_priority(c):
        repo_id = c.get("repo") or ""
        is_trusted = 0 if _trusted_author(repo_id) else 1
        downloads = -int(c.get("downloads") or 0)
        return (is_trusted, downloads)

    candidates.sort(key=_candidate_priority)

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
        # Allow upstream-alias matching only for full-text hits. A
        # full-text hit means the README mentions our filename - very
        # high signal that this repo IS the right one - so if its
        # actual weights file has a generic name like
        # "pytorch_lora_weights.safetensors" we accept that as the
        # answer and rename on download.
        accept_alias = (c["via"] == "fulltext")
        hit = _hf_find_file_in_repo(c["repo"], filename, accept_upstream_alias=accept_alias)
        if hit:
            confirmed.append((c, hit["path"], hit.get("size")))
            if hit.get("alias"):
                # Tag the candidate so callers know the repo file has a
                # different name than the workflow asked for. The download
                # path should still be the workflow's filename - the user's
                # workflow is the source of truth for naming.
                c["_alias_match"] = True

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

def _candidate_sort_key(c: dict) -> tuple:
    """Sort key for the merged candidate list. Lower tuple sorts first.

    Order of precedence:
      1. Curated DB entries (`preferred=True`) always come first - they're
         hand-picked for the file in question.
      2. Trusted-author candidates. Whether the candidate came from the
         dedicated trusted-author probe (_via='trusted-author') OR was
         pulled in by the normal search but its repo happens to be from
         a canonical org (stabilityai, black-forest-labs, comfyanonymous,
         lllyasviel, h94, latent-consistency, ai-forever, ...). Either
         way: very high confidence.
      3. By download count, descending. The official Wan2.2 LoRA repo
         with 1.2M downloads outranks a random fork with 0 downloads.
      4. HuggingFace before CivitAI when downloads tie - HF is more
         reliably accessible (no token usually needed).
      5. Stable: by title for deterministic ordering when everything else
         is equal.
    """
    preferred = 0 if c.get("preferred") else 1
    # Promote any candidate whose REPO ID is from a trusted org, not just
    # the ones the trusted-author probe found explicitly. The 'repo' key
    # is set on dict-shaped Phase-1 candidates; lookup_known/pattern
    # entries might use 'title' instead. Try both.
    repo_id = c.get("repo") or ""
    if not repo_id:
        # Try to extract from title 'owner/repo/path/file'
        title = c.get("title") or ""
        if title.count("/") >= 1:
            repo_id = "/".join(title.split("/")[:2])
    is_trusted = (
        c.get("_via") == "trusted-author"
        or _trusted_author(repo_id)
    )
    via_rank = 0 if is_trusted else 1
    # Negative because Python sorts ascending; we want highest downloads first.
    downloads = -int(c.get("downloads") or 0)
    src = c.get("source", "")
    src_rank = {"known": 0, "huggingface": 1, "civitai": 2}.get(src, 3)
    title = (c.get("title") or "").lower()
    return (preferred, via_rank, downloads, src_rank, title)


def find_candidates(filename: str, folder_hint: str | None = None) -> list[dict]:
    """Return ranked download candidates for a filename, with caching.

    The returned list is sorted with the most-likely-correct entries
    first: hand-curated database entries, then real candidates ranked by
    download count (popularity), with HuggingFace ranking above CivitAI
    when downloads tie.
    """
    # Cache lookup
    cached = _search_cache.get(filename)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_S:
        out = list(cached[1])
        if folder_hint:
            for c in out:
                c.setdefault("folder", folder_hint)
        return out

    out: list[dict] = []
    pinned = lookup_known_or_pattern(filename, folder_hint)
    if pinned:
        out.append(pinned)
        # Short-circuit: a curated DB entry or pattern-rule hit is
        # hand-picked / generated for this exact filename, so there's
        # no need to hit HuggingFace + CivitAI in addition. Skipping
        # those saves ~6 seconds per lookup, which compounds noticeably
        # during "Download all" of many files.
        if folder_hint:
            for c in out:
                c.setdefault("folder", folder_hint)
        _search_cache[filename] = (time.time(), [dict(c) for c in out])
        return out

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

    # Final ranking across all sources combined.
    out.sort(key=_candidate_sort_key)

    _search_cache[filename] = (time.time(), [dict(c) for c in out])
    return out
