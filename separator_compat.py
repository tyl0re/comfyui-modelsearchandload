"""Cross-platform separator compatibility for ComfyUI's filename lists.

Problem
-------
On Windows, ``folder_paths.get_filename_list("loras")`` returns entries with
backslashes (``"ltxv\\ltx2\\foo.safetensors"``). Many ComfyUI nodes validate
their widget value against that list. Workflows authored on Linux/macOS use
forward slashes (``"ltxv/ltx2/foo.safetensors"``), so the same model file
shows up as missing on Windows even though ``folder_paths.get_full_path``
can resolve both spellings.

Fix
---
We wrap ``folder_paths.get_filename_list`` so that on Windows every entry
containing ``\\`` is also offered with ``/``. The list still contains the
canonical Windows entry, so existing UIs keep working, but workflows that
ship with forward-slash paths are accepted too.

The wrapper is installed exactly once on import. On non-Windows systems it
is a no-op.
"""

from __future__ import annotations

import os
import sys


_INSTALLED_FLAG = "_msl_separator_compat_installed"


def install() -> None:
    """Install the wrapper on ``folder_paths.get_filename_list`` once."""
    if os.sep == "/":
        # POSIX path separator already matches workflow conventions.
        return
    try:
        import folder_paths  # ComfyUI core
    except Exception:
        return

    if getattr(folder_paths, _INSTALLED_FLAG, False):
        return

    original = folder_paths.get_filename_list

    def _wrapped(folder_name: str):
        files = original(folder_name)
        if not files:
            return files
        # Build an augmented list: keep canonical entries, append a
        # forward-slash twin for any entry containing a backslash that
        # is not already present in its forward-slash form.
        seen: set[str] = set()
        out: list[str] = []
        for entry in files:
            if entry not in seen:
                seen.add(entry)
                out.append(entry)
        for entry in list(out):
            if "\\" in entry:
                twin = entry.replace("\\", "/")
                if twin not in seen:
                    seen.add(twin)
                    out.append(twin)
        return out

    folder_paths.get_filename_list = _wrapped
    setattr(folder_paths, _INSTALLED_FLAG, True)
    print(
        "[ModelDownloader] separator_compat: folder_paths.get_filename_list now "
        "accepts both backslash and forward-slash subpaths on Windows.",
        file=sys.stderr,
    )
