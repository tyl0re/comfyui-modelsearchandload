"""ComfyUI node that injects LoRA trigger words into a prompt.

Design
------
Reading LoRA metadata at workflow-execution time is too late: you cannot
choose which trigger to use because the dropdown does not exist yet.

So the flow is split:

* The sidebar (``web/model_downloader.js``) has a *Read LoRA tags*
  button. It calls ``GET /model_downloader/lora_meta``, lets the user
  pick a LoRA, and writes the chosen triggers into a target
  ``LoRA Tag Selector`` node in the canvas. The node's display text is
  updated immediately - no restart required.

* This file provides the actual node. It is intentionally minimal: it
  takes a prompt, a triggers field that the sidebar fills in, and emits
  one combined prompt string.

Helpers for parsing LoRA metadata also live here so the sidebar's HTTP
endpoint and the node share the same code path.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - importable outside ComfyUI for tests
    folder_paths = None  # type: ignore


_META_CACHE: dict[tuple[str, float], dict[str, Any]] = {}


def lora_path(lora_name: str) -> str | None:
    """Resolve a LoRA filename to an absolute path via ComfyUI's folder_paths."""
    if not lora_name:
        return None
    if folder_paths is not None:
        try:
            p = folder_paths.get_full_path("loras", lora_name)
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    if os.path.isfile(lora_name):
        return lora_name
    return None


def _read_metadata(path: str) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise RuntimeError(
            "safetensors is required to read LoRA metadata. Install with: pip install safetensors"
        ) from e
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})


def _parse_lora_metadata(meta: dict[str, str], top_n: int = 30) -> dict[str, Any]:
    info: dict[str, Any] = {
        "title": meta.get("modelspec.title") or meta.get("ss_output_name"),
        "architecture": meta.get("modelspec.architecture")
        or meta.get("ss_base_model_version"),
        "resolution": meta.get("modelspec.resolution"),
        "num_train_images": meta.get("ss_num_train_images"),
        "num_epochs": meta.get("ss_num_epochs"),
        "steps": meta.get("ss_steps"),
        "triggers": [],
        "dataset_dirs": [],
        "top_tags": [],
        "has_metadata": bool(meta),
    }
    raw_dirs = meta.get("ss_dataset_dirs")
    if raw_dirs:
        try:
            data = json.loads(raw_dirs)
            for key in data.keys():
                info["dataset_dirs"].append(key)
                parts = key.split(" ", 1)
                if len(parts) == 2 and parts[1].strip():
                    info["triggers"].append(parts[1].strip())
        except Exception:
            pass

    bucket: Counter = Counter()
    raw_tags = meta.get("ss_tag_frequency")
    if raw_tags:
        try:
            data = json.loads(raw_tags)
            for _ds, freqs in data.items():
                for tag, n in freqs.items():
                    bucket[tag] += int(n)
        except Exception:
            pass
    info["top_tags"] = [
        {"tag": t, "count": n} for t, n in bucket.most_common(top_n)
    ]
    return info


def info_for(lora_name: str, top_n: int = 30) -> dict[str, Any]:
    """Cached LoRA metadata lookup. Public so the HTTP route can use it."""
    path = lora_path(lora_name)
    if not path:
        return {
            "lora_name": lora_name,
            "path": None,
            "title": None, "architecture": None, "resolution": None,
            "num_train_images": None, "num_epochs": None, "steps": None,
            "triggers": [], "dataset_dirs": [], "top_tags": [],
            "has_metadata": False,
            "error": f"LoRA file not found: {lora_name}",
        }
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = (path, mtime)
    cached = _META_CACHE.get(key)
    if cached and cached.get("_top_n") == top_n:
        return cached
    try:
        meta = _read_metadata(path)
    except Exception as e:
        info = {
            "lora_name": lora_name,
            "path": path,
            "title": None, "architecture": None, "resolution": None,
            "num_train_images": None, "num_epochs": None, "steps": None,
            "triggers": [], "dataset_dirs": [], "top_tags": [],
            "has_metadata": False, "error": str(e),
            "_top_n": top_n,
        }
        _META_CACHE[key] = info
        return info
    info = _parse_lora_metadata(meta, top_n=top_n)
    info["lora_name"] = lora_name
    info["path"] = path
    info["_top_n"] = top_n
    _META_CACHE[key] = info
    return info


# --------------------------------------------------------------------------
# Node: LoRA Tag Selector + Prompt Merger
# --------------------------------------------------------------------------

class LoraTagSelector:
    """Hold pre-picked LoRA tags and append them to a prompt.

    The ``selected_tags`` field is populated by the sidebar's
    "Read LoRA tags" workflow. Each line is one tag; optional weight after
    ``::`` or ``|`` (e.g. ``low lighting :: 1.1``). The node combines the
    incoming prompt with the selected tags and returns a single string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "selected_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# Click 'Read LoRA tags' in the ModelDownloader\n"
                            "# sidebar to fill this in. One tag per line.\n"
                            "# Optional weight after :: or |\n"
                            "# example:  low lighting :: 1.1\n"
                        ),
                    },
                ),
                "position": (["append", "prepend"], {"default": "append"}),
                "separator": ("STRING", {"default": ", "}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "preview")
    FUNCTION = "select"
    CATEGORY = "l0re/LoRA"

    @staticmethod
    def _parse_lines(text: str) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            weight = 1.0
            for sep in ("::", "|"):
                if sep in line:
                    left, right = line.rsplit(sep, 1)
                    try:
                        weight = float(right.strip())
                        line = left.strip()
                        break
                    except ValueError:
                        pass
            if line:
                out.append((line, weight))
        return out

    @staticmethod
    def _weighted(tag: str, weight: float) -> str:
        if abs(weight - 1.0) < 1e-3:
            return tag
        return f"({tag}:{weight:.2f})"

    def select(self, prompt, selected_tags, position, separator):
        tag_parts = [
            self._weighted(t, w)
            for t, w in self._parse_lines(selected_tags)
        ]
        prompt_clean = (prompt or "").strip()
        triggers_str = (separator or " ").join(tag_parts).strip(" ,")
        if not prompt_clean and not triggers_str:
            return ("", "")
        if not triggers_str:
            return (prompt_clean, prompt_clean)
        if not prompt_clean:
            return (triggers_str, triggers_str)
        if position == "prepend":
            combined = (separator or " ").join([triggers_str, prompt_clean])
        else:
            combined = (separator or " ").join([prompt_clean, triggers_str])
        combined = ", ".join(seg.strip() for seg in combined.split(",") if seg.strip())
        return (combined, combined)


NODE_CLASS_MAPPINGS = {
    "LoraTagSelector": LoraTagSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraTagSelector": "LoRA Tag Selector",
}
