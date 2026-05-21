"""Pattern-based filename resolution.

This module replaces dozens of one-off entries in ``known_models.json``
with a small set of generic rules. The goal is that when a NEW file
name appears that follows an existing pattern (e.g. a new LCM LoRA, a
new ControlNet 1.1 variant, a new IP-Adapter encoder), the resolver
finds it without anyone touching the curated DB.

There are three engines:

  1. Alias normalisation
     - Treat ``foo-bar.safetensors`` and ``foo_bar.safetensors`` as the
       same file when looking things up.
     - Drop common decorative prefixes / suffixes (``v2_``, ``_fp16``,
       case differences) when matching against patterns.

  2. Pattern rules
     A pattern rule says "if the filename matches this regex / contains
     these tokens, the file lives at this URL". The URL can include
     captured groups from the regex so a single rule maps a whole
     family of filenames.

  3. Upstream-alias resolution during HuggingFace tree lookups
     Many HF repos name their single weights file ``model.safetensors``
     or ``pytorch_lora_weights.safetensors``. When we have probed a
     repo via the umbrella-repo or full-text path and find such a file,
     we treat it as the answer for the workflow's filename - the
     downloaded file is renamed locally to whatever the workflow asked
     for.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Engine 1 - Filename normalisation
# ---------------------------------------------------------------------------

def normalise_filename(name: str) -> str:
    """Return a comparable form of `name`.

    - Strips path components
    - Lower-cases
    - Replaces - with _ so 'clip_vision-sd15' and 'clip_vision_sd15'
      hash-equal
    - Strips a leading ``./`` or ``/``
    """
    if not name:
        return ""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.lower().strip()
    name = name.replace("-", "_")
    return name


def filename_aliases(name: str) -> list[str]:
    """Return the list of alternative spellings worth trying.

    Used when consulting ``known_models.json`` and pattern rules - a DB
    keyed by ``clip_vision_sd15.safetensors`` should also serve a
    workflow asking for ``clip_vision-sd15.safetensors``.
    """
    if not name:
        return []
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    out: list[str] = [base]

    # Some workflow/plugin authors add local-only decorative prefixes to
    # filenames even when the upstream file omits them, e.g.
    # comfy_gemma_3_12B_it.safetensors -> gemma_3_12B_it.safetensors.
    # Keep this conservative: only strip clearly Comfy-related prefixes.
    decorative_prefixes = (
        "comfy_",
        "comfy-",
        "comfyui_",
        "comfyui-",
    )
    base_lc = base.lower()
    for prefix in decorative_prefixes:
        if base_lc.startswith(prefix) and len(base) > len(prefix):
            out.append(base[len(prefix):])
    # Hyphen <-> underscore swap
    if "-" in base:
        out.append(base.replace("-", "_"))
    if "_" in base:
        out.append(base.replace("_", "-"))
    # Also apply separator swaps to generated aliases.
    for v in list(out):
        if "-" in v:
            out.append(v.replace("-", "_"))
        if "_" in v:
            out.append(v.replace("_", "-"))
    # Lower-case variants
    out += [v.lower() for v in list(out) if v != v.lower()]
    # Deduplicate, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


# ---------------------------------------------------------------------------
# Engine 2 - Pattern rules
# ---------------------------------------------------------------------------
# Each rule is a dict:
#   {
#     "pattern": <compiled regex matched against normalised filename>,
#     "url":     <URL template, can use \\1 \\2 backrefs from the regex>,
#     "folder":  ComfyUI folder name,
#     "size":    optional expected size in bytes (used for link-dedupe),
#     "source":  "huggingface" / "civitai" (informational),
#     "title":   human-readable (can use backrefs)
#   }
#
# Rules are tried in order. The first match wins. A rule is matched
# against the NORMALISED filename (normalise_filename), so use lowercase
# and underscores in the regex.

_RULES: list[dict] = [
    # -----------------------------------------------------------------
    # ControlNet v1.1 (lllyasviel) - one rule covers ALL task variants
    # -----------------------------------------------------------------
    # The repo holds 14+ controlnets named control_v11p_sd15_<task>.pth.
    # Keep the task list explicit so decorative suffixes like _fp16 do
    # not produce URLs for files that do not exist upstream.
    {
        "pattern": re.compile(
            r"^control[_]?v11(?:[fp][1ep]?)?p?[_]?sd15[_]?"
            r"(canny|openpose|depth|normalbae|lineart|mlsd|scribble|seg|tile|inpaint|ip2p|softedge|shuffle)"
            r"\.(?:pth|safetensors)$"
        ),
        "url":    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_\\g<1>.pth",
        "folder": "controlnet",
        "source": "huggingface",
        "title":  "lllyasviel/ControlNet-v1-1 \\g<1>",
    },

    # -----------------------------------------------------------------
    # Depth-Anything V1 - one rule covers vitl / vitb / vits
    # -----------------------------------------------------------------
    # LiheYoung/depth_anything_<size>14 - the actual file at HF is
    # 'pytorch_model.bin' but workflows reference it as
    # 'depth_anything_<size>14.pth'. The download manager preserves the
    # workflow's filename on disk via the upstream-alias mechanism.
    {
        "pattern": re.compile(r"^depth[_]?anything[_]?(vitl|vitb|vits)14\.(?:pth|bin|safetensors)$"),
        "url":    "https://huggingface.co/LiheYoung/depth_anything_\\g<1>14/resolve/main/pytorch_model.bin",
        "folder": "controlnet_aux",
        "source": "huggingface",
        "title":  "LiheYoung/depth_anything_\\g<1>14",
    },

    # -----------------------------------------------------------------
    # Depth-Anything V2 - one rule covers vitl / vitb / vits
    # -----------------------------------------------------------------
    # Three repos with capitalised size words. _size_map expands the
    # captured short name into Large/Base/Small for the repo path.
    {
        "pattern": re.compile(r"^depth[_]?anything[_]?v2[_]?(vitl|vitb|vits)\.pth$"),
        "url":    "https://huggingface.co/depth-anything/Depth-Anything-V2-{SIZE}/resolve/main/depth_anything_v2_\\g<1>.pth",
        "folder": "controlnet_aux",
        "source": "huggingface",
        "title":  "depth-anything/Depth-Anything-V2-\\g<1>",
        "_size_map": {"vitl": "Large", "vitb": "Base", "vits": "Small"},
    },

    # -----------------------------------------------------------------
    # Wan2.1 control LoRAs - spacepxl/Wan2.1-control-loras
    # -----------------------------------------------------------------
    # Files live in nested subfolders: 1.3b/<task>/wan2.1-1.3b-control-lora-
    # <task>-v<X.Y>_comfy.safetensors. HF search splits 'wan2.1' on the dot
    # so the regular search never returns this repo. Tasks released so far:
    # tile (v0.2/v1.0/v1.1), depth (v0.1). Pattern catches future versions.
    # normalise_filename converts dots to _ inside numbers? No - dots are
    # left alone. So we match the literal filename here.
    {
        "pattern": re.compile(
            r"^wan2\.1[_]?1\.3b[_]?control[_]?lora[_]?(tile|depth)[_]?v(\d+\.\d+)[_]?comfy\.safetensors$"
        ),
        "url":    "https://huggingface.co/spacepxl/Wan2.1-control-loras/resolve/main/1.3b/\\g<1>/wan2.1-1.3b-control-lora-\\g<1>-v\\g<2>_comfy.safetensors",
        "folder": "loras",
        "source": "huggingface",
        "size":   378400000,
        "title":  "spacepxl/Wan2.1-control-loras 1.3b/\\g<1> v\\g<2>",
    },
]


def _render(template: str, match: re.Match, rule: dict) -> str:
    """Apply regex backrefs in `template` and resolve {SIZE}-style tokens
    using the rule's optional `_size_map`."""
    out = match.expand(template)
    smap = rule.get("_size_map")
    if smap and "{SIZE}" in out:
        # Use the first captured group as the lookup key
        try:
            key = match.group(1).lower()
        except IndexError:
            key = ""
        out = out.replace("{SIZE}", smap.get(key, ""))
    return out


def lookup_pattern(filename: str) -> Optional[dict]:
    """Return a candidate dict if `filename` matches a pattern rule.

    The candidate dict has the same shape as a curated DB entry so the
    rest of the pipeline doesn't care where it came from.
    """
    norm = normalise_filename(filename)
    if not norm:
        return None
    for rule in _RULES:
        m = rule["pattern"].match(norm)
        if not m:
            continue
        return {
            "source":   rule.get("source", "huggingface"),
            "title":    _render(rule.get("title", filename), m, rule),
            "filename": filename,  # preserve the workflow's original spelling
            "folder":   rule["folder"],
            "url":      _render(rule["url"], m, rule),
            "size":     rule.get("size"),
            "gated":    rule.get("gated", False),
            "preferred": True,
            "_via":     "pattern",
        }
    return None


# ---------------------------------------------------------------------------
# Engine 3 - Upstream alias filenames
# ---------------------------------------------------------------------------
# When walking a HF repo's tree (umbrella-repo lookup or full-text hit),
# accept these "anonymous" filenames as a match for ANY workflow filename
# we're looking for. These are the conventions HF/diffusers use for
# single-file releases.

_UPSTREAM_ALIAS_FILES: tuple[str, ...] = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.safetensors",
)


def is_upstream_alias(repo_filename: str) -> bool:
    """True if a file at this path inside a HF repo is a generic
    'single weights file' alias that we can treat as the answer for
    whatever filename the workflow used."""
    if not repo_filename:
        return False
    base = repo_filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base in _UPSTREAM_ALIAS_FILES
