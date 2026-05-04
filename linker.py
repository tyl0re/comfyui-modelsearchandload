"""Find existing local copies of a model file and link them into the
target location instead of re-downloading. Saves disk space.

Strategy:
  1. Walk the models tree (capped) collecting (basename, size) -> [paths]
  2. When a download is requested, look up the basename. If we find a
     candidate with matching size, link it to the target path.
  3. Try hardlink first (instant, invisible to programs, costs no extra
     space). If that fails (cross-filesystem, permissions, ...), fall
     back to symlink. If symlinks also fail (Windows without dev mode
     or admin), fall back to copy.

Returns a small dict describing what happened so the UI can show
"linked from /existing/path" instead of progress bars.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover
    folder_paths = None


# Same set used by scanner._build_local_index for consistency.
_LINK_INDEX_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".pkl", ".sft",
    ".onnx", ".gguf", ".engine", ".trt", ".msgpack",
)


# A reasonable safety cap to avoid pathological scans on huge trees.
_MAX_INDEXED_FILES = 100_000


def _models_root_dirs() -> list[str]:
    """Mirror of scanner._models_root_dirs but locally to avoid a
    cross-module dependency at import time."""
    roots: list[str] = []
    seen: set[str] = set()

    def add(p: Optional[str]) -> None:
        if not p:
            return
        try:
            ap = os.path.abspath(p)
        except Exception:
            return
        if ap in seen or not os.path.isdir(ap):
            return
        seen.add(ap)
        roots.append(ap)

    if folder_paths is not None:
        try:
            add(folder_paths.models_dir)
        except Exception:
            pass
        # Iterate every registered folder name (covers extra_model_paths.yaml)
        for key in list(getattr(folder_paths, "folder_names_and_paths", {}).keys()):
            try:
                for p in folder_paths.get_folder_paths(key) or []:
                    add(p)
                    add(os.path.dirname(p))
            except Exception:
                continue
    return roots


def _walk_with_symlinks(root: str):
    """os.walk wrapper with followlinks=True and inode-based loop detection.

    On Linux it is common to symlink model directories (e.g. NAS mounts) into
    the models tree. followlinks=False would make those invisible. We follow
    symlinks but track visited (dev, ino) pairs to avoid infinite loops from
    circular symlinks.
    """
    seen_inodes: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
            key = (st.st_dev, st.st_ino)
            if key in seen_inodes:
                dirnames[:] = []
                continue
            seen_inodes.add(key)
        except OSError:
            pass
        yield dirpath, dirnames, filenames


def build_size_index() -> dict[tuple[str, int], list[str]]:
    """Build {(basename_lc, size_bytes): [absolute_path, ...]} index.

    The size disambiguates files that share a name but have different
    content (e.g. fp16 vs fp32 weights). Building this is fast - it's
    an os.walk + os.path.getsize, no file content is read.
    """
    index: dict[tuple[str, int], list[str]] = {}
    count = 0
    for root in _models_root_dirs():
        for dirpath, _dirs, files in _walk_with_symlinks(root):
            for fn in files:
                lc = fn.lower()
                if not lc.endswith(_LINK_INDEX_EXTS):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                index.setdefault((lc, sz), []).append(full)
                count += 1
                if count >= _MAX_INDEXED_FILES:
                    return index
    return index


def find_existing_copy(filename: str, expected_size: Optional[int]) -> Optional[str]:
    """Find a local copy of `filename`. If `expected_size` is provided,
    only return a path with matching size. Returns the absolute path of
    the first match, or None.

    Skips files that are themselves symlinks to keep the link chain flat
    (we want to point at the real backing file, not at another link).
    """
    fn_lc = filename.lower()
    candidates: list[str] = []

    if expected_size is not None and expected_size > 0:
        # Fast path: build a tiny index just for this lookup.
        index = build_size_index()
        candidates = list(index.get((fn_lc, expected_size), []))
    else:
        # No size known yet -> scan and accept any same-name file.
        for root in _models_root_dirs():
            for dirpath, _dirs, files in _walk_with_symlinks(root):
                for fn in files:
                    if fn.lower() == fn_lc:
                        candidates.append(os.path.join(dirpath, fn))
            if len(candidates) >= 20:
                break  # plenty for picking one

    # Filter out symlinks - we want a real backing file.
    real = [p for p in candidates if not os.path.islink(p)]
    if real:
        # Prefer the largest file in case multiple matches; usually all
        # are identical size when we used the size index.
        try:
            real.sort(key=os.path.getsize, reverse=True)
        except OSError:
            pass
        return real[0]
    if candidates:
        return candidates[0]
    return None


def _try_hardlink(src: str, dst: str) -> tuple[bool, Optional[str]]:
    """Returns (success, error). Hardlink fails when src/dst are on
    different filesystems, on FAT32 (no hardlink support), and when the
    user lacks the right permissions."""
    try:
        os.link(src, dst)
        return True, None
    except OSError as e:
        return False, str(e)


def _try_symlink(src: str, dst: str) -> tuple[bool, Optional[str]]:
    """On Windows, symlinks need either Developer Mode or admin rights.
    On Linux/macOS this almost always works."""
    try:
        os.symlink(src, dst)
        return True, None
    except OSError as e:
        return False, str(e)


def link_existing(
    src: str,
    dst: str,
    mode: str = "auto",
) -> dict:
    """Try to link `src` into `dst`.

    mode = "auto"      -> hardlink, fall back to symlink, never copy
    mode = "hardlink"  -> hardlink only
    mode = "symlink"   -> symlink only

    Returns a dict:
      {
        "linked": True/False,
        "method": "hardlink" | "symlink" | None,
        "error":  None | "...",
        "src": src, "dst": dst,
      }
    """
    if not os.path.exists(src):
        return {"linked": False, "method": None, "error": f"source missing: {src}",
                "src": src, "dst": dst}
    # Make sure dest dir exists; do not overwrite existing files.
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        # Already there - if it points at the same inode/path, treat as success.
        try:
            if os.path.samefile(src, dst):
                return {"linked": True, "method": "already-linked",
                        "error": None, "src": src, "dst": dst}
        except OSError:
            pass
        return {"linked": False, "method": None,
                "error": f"target exists: {dst}", "src": src, "dst": dst}

    errors: list[str] = []

    if mode in ("auto", "hardlink"):
        ok, err = _try_hardlink(src, dst)
        if ok:
            return {"linked": True, "method": "hardlink", "error": None,
                    "src": src, "dst": dst}
        errors.append(f"hardlink failed: {err}")

    if mode in ("auto", "symlink"):
        ok, err = _try_symlink(src, dst)
        if ok:
            return {"linked": True, "method": "symlink", "error": None,
                    "src": src, "dst": dst}
        errors.append(f"symlink failed: {err}")
        if sys.platform == "win32" and "privilege" in (err or "").lower():
            errors.append(
                "On Windows, symlinks require Developer Mode "
                "(Settings > For developers > Developer Mode = ON) "
                "or running ComfyUI as administrator."
            )

    return {"linked": False, "method": None, "error": " | ".join(errors),
            "src": src, "dst": dst}
