"""ComfyUI nodes for extracting LoRA trigger words and merging them into a prompt.

Three small read-only nodes are registered:

* ``LoRA Trigger Inspector`` reads a LoRA's ``.safetensors`` metadata
  (kohya / sd-scripts convention) and returns:
    - ``trigger``: best-guess trigger phrase (from ``ss_dataset_dirs``)
    - ``top_tags``: comma-separated list of the most frequent training tags
    - ``info``: short multi-line summary (title, base model, train counts)

* ``LoRA Tag Selector`` lets the user pick which tag(s) to use from the
  inspected LoRA, with optional ComfyUI-style ``(tag:1.2)`` weighting. The
  output is a single string of comma-separated weighted tags.

* ``Append LoRA Triggers`` merges any text (e.g. a base prompt) with one or
  more trigger strings into a single prompt, handling spacing / commas so
  the result is safe to feed into ``CLIPTextEncode``.

Nothing in here mutates the LoRA file; reads are cached per ``(path, mtime)``
so re-evaluating a workflow doesn't re-parse the file each frame.
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


def _lora_path(lora_name: str) -> str | None:
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
    """Distil safetensors metadata into a small dict usable by the nodes."""
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
        "all_tag_counts": {},
        "has_metadata": bool(meta),
    }
    raw_dirs = meta.get("ss_dataset_dirs")
    if raw_dirs:
        try:
            data = json.loads(raw_dirs)
            for key in data.keys():
                info["dataset_dirs"].append(key)
                # Kohya convention: "<repeats>_<id> [trigger phrase]"
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
    info["all_tag_counts"] = dict(bucket)
    info["top_tags"] = [
        {"tag": t, "count": n} for t, n in bucket.most_common(top_n)
    ]
    return info


def _info_for(lora_name: str, top_n: int = 30) -> dict[str, Any]:
    """Cached metadata fetch keyed by (path, mtime)."""
    path = _lora_path(lora_name)
    if not path:
        return {
            "title": None, "architecture": None, "resolution": None,
            "num_train_images": None, "num_epochs": None, "steps": None,
            "triggers": [], "dataset_dirs": [], "top_tags": [],
            "all_tag_counts": {}, "has_metadata": False,
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
            "title": None, "architecture": None, "resolution": None,
            "num_train_images": None, "num_epochs": None, "steps": None,
            "triggers": [], "dataset_dirs": [], "top_tags": [],
            "all_tag_counts": {}, "has_metadata": False, "error": str(e),
            "_top_n": top_n,
        }
        _META_CACHE[key] = info
        return info
    info = _parse_lora_metadata(meta, top_n=top_n)
    info["_top_n"] = top_n
    _META_CACHE[key] = info
    return info


def _format_summary(name: str, info: dict[str, Any]) -> str:
    if info.get("error"):
        return f"{name}\nERROR: {info['error']}"
    if not info["has_metadata"]:
        return f"{name}\n(no metadata - cannot infer trigger; check the model card)"
    lines = [name]
    if info.get("title"):
        lines.append(f"title: {info['title']}")
    if info.get("architecture"):
        lines.append(f"arch:  {info['architecture']}")
    if info.get("resolution"):
        lines.append(f"res:   {info['resolution']}")
    train_bits = []
    if info.get("num_train_images"):
        train_bits.append(f"{info['num_train_images']} imgs")
    if info.get("num_epochs"):
        train_bits.append(f"{info['num_epochs']} ep")
    if info.get("steps"):
        train_bits.append(f"{info['steps']} steps")
    if train_bits:
        lines.append("train: " + ", ".join(train_bits))
    if info["triggers"]:
        lines.append("triggers (from dataset dirs):")
        for t in info["triggers"]:
            lines.append(f"  - {t}")
    else:
        lines.append("triggers: none found in dataset_dirs")
    if info["top_tags"]:
        lines.append("top training tags:")
        for t in info["top_tags"][:10]:
            lines.append(f"  - {t['tag']} ({t['count']})")
    return "\n".join(lines)


def _split_user_selection(text: str) -> list[tuple[str, float]]:
    """Parse the multiline selection field.

    Each non-empty line is treated as one tag. Optional trailing weight syntax
    ``tag :: 1.2`` or ``tag | 1.2`` sets the weight; otherwise weight 1.0.
    """
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


def _format_weighted(tag: str, weight: float) -> str:
    if abs(weight - 1.0) < 1e-3:
        return tag
    # ComfyUI / A1111 syntax: (tag:1.2)
    return f"({tag}:{weight:.2f})"


# --------------------------------------------------------------------------
# Node 1: Inspector
# --------------------------------------------------------------------------

class LoRATriggerInspector:
    """Read trigger / top-tag metadata from a LoRA file.

    Outputs are pure strings so the node can be chained into any text-encode
    setup (CLIPTextEncode, ShowText, FluxGuidance prompt fields, ...).
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = [""]
        if folder_paths is not None:
            try:
                loras = folder_paths.get_filename_list("loras") or [""]
            except Exception:
                loras = [""]
        return {
            "required": {
                "lora_name": (loras, ),
                "top_n": ("INT", {"default": 15, "min": 1, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("trigger", "top_tags", "info")
    FUNCTION = "inspect"
    CATEGORY = "ModelDownloader/LoRA"

    @classmethod
    def IS_CHANGED(cls, lora_name, top_n):  # pragma: no cover - ComfyUI hook
        path = _lora_path(lora_name)
        if not path:
            return f"missing:{lora_name}"
        try:
            return f"{path}:{os.path.getmtime(path)}:{top_n}"
        except OSError:
            return f"{path}:0:{top_n}"

    def inspect(self, lora_name: str, top_n: int):
        info = _info_for(lora_name, top_n=top_n)
        trigger = info["triggers"][0] if info["triggers"] else ""
        top_tags = ", ".join(t["tag"] for t in info["top_tags"][:top_n])
        summary = _format_summary(lora_name, info)
        return (trigger, top_tags, summary)


# --------------------------------------------------------------------------
# Node 2: Tag selector / weighting
# --------------------------------------------------------------------------

class LoRATagSelector:
    """Pick tags from a LoRA and emit a comma-separated weighted string.

    Workflow:
      1. Hit ``Inspect`` on the LoRA - the ``info`` socket of the inspector
         lists the top training tags.
      2. Copy the tags you want into ``selected_tags`` here, one per line.
      3. Optionally add a weight with ``::`` (``tag :: 1.2``) or ``|``.
      4. Connect ``triggers`` into ``Append LoRA Triggers``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = [""]
        if folder_paths is not None:
            try:
                loras = folder_paths.get_filename_list("loras") or [""]
            except Exception:
                loras = [""]
        return {
            "required": {
                "lora_name": (loras, ),
                "include_dataset_trigger": ("BOOLEAN", {"default": True}),
                "dataset_trigger_weight": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "selected_tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# One tag per line. Optional weight after :: or |\n"
                            "# example:\n"
                            "# young woman :: 1.1\n"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("triggers", "preview")
    FUNCTION = "select"
    CATEGORY = "ModelDownloader/LoRA"

    @classmethod
    def IS_CHANGED(cls, lora_name, include_dataset_trigger, dataset_trigger_weight, selected_tags):
        path = _lora_path(lora_name)
        mtime = 0.0
        if path:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                pass
        return f"{lora_name}:{mtime}:{include_dataset_trigger}:{dataset_trigger_weight}:{hash(selected_tags)}"

    def select(self, lora_name, include_dataset_trigger, dataset_trigger_weight, selected_tags):
        info = _info_for(lora_name, top_n=30)
        parts: list[str] = []
        if include_dataset_trigger:
            for trig in info["triggers"]:
                parts.append(_format_weighted(trig, dataset_trigger_weight))
        for tag, weight in _split_user_selection(selected_tags):
            parts.append(_format_weighted(tag, weight))
        out = ", ".join(parts)
        preview = out or "(no triggers / tags selected)"
        return (out, preview)


# --------------------------------------------------------------------------
# Node 3: Merge with base prompt
# --------------------------------------------------------------------------

class AppendLoRATriggers:
    """Merge a base prompt and one or more trigger strings into one prompt.

    Empty inputs are skipped; whitespace and trailing commas are cleaned up.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "triggers": ("STRING", {"multiline": False, "default": ""}),
                "position": (["append", "prepend"], {"default": "append"}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "extra_triggers": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "merge"
    CATEGORY = "ModelDownloader/LoRA"

    def merge(self, prompt, triggers, position, separator, extra_triggers: str = ""):
        bits: list[str] = []
        for s in (prompt, triggers, extra_triggers):
            s = (s or "").strip()
            if s:
                bits.append(s)
        if not bits:
            return ("",)

        base = bits[0]
        rest = bits[1:]
        joined_rest = (separator or " ").join(rest).strip(" ,")
        if not joined_rest:
            return (base,)
        if position == "prepend":
            text = (separator or " ").join([joined_rest, base])
        else:
            text = (separator or " ").join([base, joined_rest])
        # Normalise duplicate commas and surrounding whitespace.
        text = ", ".join(seg.strip() for seg in text.split(",") if seg.strip())
        return (text,)


NODE_CLASS_MAPPINGS = {
    "LoRATriggerInspector": LoRATriggerInspector,
    "LoRATagSelector": LoRATagSelector,
    "AppendLoRATriggers": AppendLoRATriggers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRATriggerInspector": "LoRA Trigger Inspector",
    "LoRATagSelector": "LoRA Tag Selector",
    "AppendLoRATriggers": "Append LoRA Triggers",
}
