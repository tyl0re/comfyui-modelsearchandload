"""Persistent SHA-256 hash cache for the dedupe scanner.

Hashing multi-GB safetensors files takes minutes per file. Re-running the
'Free Space Via Link' scan would re-hash everything from scratch, which is
unfriendly. This cache persists `path -> {size, mtime, hash}` so that we
only re-hash files whose content actually changed (mtime + size mismatch).

Cache layout (JSON, written next to config.json):

    {
        "version": 2,
        "entries": {
            "/home/user/ComfyUI/models/loras/foo.safetensors": {
                "size": 12345678,
                "mtime": 1735689600.0,
                "hash": "abc123...sha256..."
            },
            ...
        }
    }

Path keys are absolute and lowercased uniformly on every platform.
That way the same file looked up under spellings that differ only in
case (e.g. drive-letter case on Windows, or a workflow that quoted
the path with mixed case) hits the same cache entry.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

from .config import PLUGIN_DIR

CACHE_FILE = PLUGIN_DIR / "dedupe_cache.json"
# Bumped from 1 -> 2 when _key() switched from os.path.normcase to
# .lower(). On Linux the old cache had case-sensitive keys; the new
# code expects lowercased keys. Old caches are simply discarded on
# load (which is harmless - they get rebuilt on the next scan).
_CACHE_VERSION = 2

# In-memory copy of the cache; written back to disk lazily.
# Keyed by os.path.normcase(os.path.abspath(path)).
_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_dirty = False


def _key(path: str) -> str:
    """Canonical lookup key for a filesystem path.

    We use absolute path + .lower() (NOT os.path.normcase) so that the
    cache behaves the same on Windows and Linux: keys are always
    case-folded. On Linux this is technically more aggressive than the
    filesystem requires (ext4 is case-sensitive), but that only causes
    the same cache entry to serve two paths that differ only in case -
    no false hits, no data corruption.

    Rationale: matches the case-insensitive strategy in downloader.py's
    duplicate-detection so users get consistent behaviour across both
    code paths regardless of OS.
    """
    try:
        return os.path.abspath(path).lower()
    except Exception:
        return path.lower()


def _load_from_disk() -> dict[str, dict]:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if data.get("version") != _CACHE_VERSION:
            return {}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {}
        # Sanity-filter: every entry must be a dict with the three keys.
        return {
            k: v for k, v in entries.items()
            if isinstance(v, dict) and "hash" in v and "size" in v and "mtime" in v
        }
    except Exception:
        return {}


def _ensure_loaded() -> dict[str, dict]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load_from_disk()
        return _cache


def get(path: str, expected_size: int, expected_mtime: float,
        tolerance: float = 1.0) -> Optional[str]:
    """Return the cached SHA-256 for `path` if size + mtime still match.

    `tolerance` is the mtime comparison fudge in seconds - some filesystems
    (FAT, network shares) round mtime to 2-second precision so an exact
    float-equality check would needlessly invalidate the cache. Default
    of 1.0 second is safe.

    Returns None on cache miss / stale entry / disabled cache.
    """
    cache = _ensure_loaded()
    entry = cache.get(_key(path))
    if not entry:
        return None
    if entry.get("size") != expected_size:
        return None
    cached_mtime = entry.get("mtime")
    try:
        if abs(float(cached_mtime) - float(expected_mtime)) > tolerance:
            return None
    except (TypeError, ValueError):
        return None
    return entry.get("hash")


def put(path: str, size: int, mtime: float, hash_hex: str) -> None:
    """Insert / update an entry. Marks the cache dirty for a later flush."""
    global _dirty
    if not hash_hex:
        return
    cache = _ensure_loaded()
    with _lock:
        cache[_key(path)] = {
            "size": int(size),
            "mtime": float(mtime),
            "hash": hash_hex,
        }
        _dirty = True


def remove(path: str) -> None:
    """Drop a single entry (e.g. after the file was deleted/replaced)."""
    global _dirty
    cache = _ensure_loaded()
    k = _key(path)
    with _lock:
        if k in cache:
            del cache[k]
            _dirty = True


def flush() -> None:
    """Write the in-memory cache to disk. Call this at the end of a scan."""
    global _dirty
    with _lock:
        if not _dirty or _cache is None:
            return
        try:
            tmp = str(CACHE_FILE) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "version": _CACHE_VERSION,
                    "entries": _cache,
                }, f, indent=2, sort_keys=True)
            os.replace(tmp, CACHE_FILE)
            _dirty = False
        except Exception as e:
            # Best-effort; the cache is just an optimisation.
            print(f"[ModelDownloader] dedupe cache flush failed: {e}")


def clear() -> int:
    """Wipe the cache entirely. Returns number of entries removed."""
    global _dirty
    cache = _ensure_loaded()
    with _lock:
        n = len(cache)
        cache.clear()
        _dirty = True
    flush()
    return n


def stats() -> dict:
    """Lightweight summary used by the Settings UI (entry count, disk size)."""
    cache = _ensure_loaded()
    n = len(cache)
    file_size = 0
    if CACHE_FILE.exists():
        try:
            file_size = os.path.getsize(CACHE_FILE)
        except OSError:
            pass
    return {"entries": n, "file_bytes": file_size}


def prune_missing() -> int:
    """Drop entries whose underlying file no longer exists.
    Returns number removed. Useful housekeeping after large file moves."""
    global _dirty
    cache = _ensure_loaded()
    removed = 0
    with _lock:
        # Iterate over a snapshot of keys so we can mutate the dict.
        for k in list(cache.keys()):
            # Keys are lowercased absolute paths. os.path.exists works
            # case-insensitively on Windows (NTFS) and case-sensitively
            # on Linux. On Linux this means a renamed file with a
            # different-case spelling looks "missing" and gets pruned -
            # which is the correct outcome.
            if not os.path.exists(k):
                del cache[k]
                removed += 1
        if removed:
            _dirty = True
    if removed:
        flush()
    return removed
