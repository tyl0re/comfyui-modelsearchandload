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
    # Hyphen <-> underscore swap
    if "-" in base:
        out.append(base.replace("-", "_"))
    if "_" in base:
        out.append(base.replace("_", "-"))
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
    # LCM LoRAs - latent-consistency Org on HuggingFace
    # -----------------------------------------------------------------
    {
        "pattern": re.compile(
            r"^(?:lcm[_]?lora[_]?weights[_]?sd[_]?1?[_.]?5"
            r"|lcm[_]?lora[_]?sdv?1?[_.]?5"
            r"|lcm[_]?lora[_]?sd15)\.safetensors$"
        ),
        "url":    "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors",
        "folder": "loras",
        "size":   134621556,
        "source": "huggingface",
        "title":  "latent-consistency/lcm-lora-sdv1-5",
    },
    {
        "pattern": re.compile(
            r"^(?:lcm[_]?lora[_]?weights[_]?sdxl"
            r"|lcm[_]?lora[_]?sdxl)\.safetensors$"
        ),
        "url":    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
        "folder": "loras",
        "size":   393854592,
        "source": "huggingface",
        "title":  "latent-consistency/lcm-lora-sdxl",
    },
    {
        "pattern": re.compile(r"^lcm[_]?lora[_]?ssd[_]?1b\.safetensors$"),
        "url":    "https://huggingface.co/latent-consistency/lcm-lora-ssd-1b/resolve/main/pytorch_lora_weights.safetensors",
        "folder": "loras",
        "source": "huggingface",
        "title":  "latent-consistency/lcm-lora-ssd-1b",
    },

    # -----------------------------------------------------------------
    # CLIP Vision encoders used by IP-Adapter
    # -----------------------------------------------------------------
    {
        # SD15 image encoder (CLIP-ViT-H, 2.4 GB) - many alias names
        "pattern": re.compile(
            r"^(?:clip[_]?vision[_]?sd1?[_.]?5"
            r"|ip[_]?adapter[_]?image[_]?encoder[_]?sd15"
            r"|clip[_]?vit[_]?h[_]?14[_]?laion2b.*"
            r"|image[_]?encoder[_]?sd15)\.safetensors$"
        ),
        "url":    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        "folder": "clip_vision",
        "size":   2528373448,
        "source": "huggingface",
        "title":  "h94/IP-Adapter (image_encoder, CLIP-ViT-H for SD1.5 IP-Adapter)",
    },
    {
        # SDXL image encoder (3.5 GB)
        "pattern": re.compile(
            r"^(?:clip[_]?vision[_]?sdxl"
            r"|ip[_]?adapter[_]?image[_]?encoder[_]?sdxl"
            r"|image[_]?encoder[_]?sdxl)\.safetensors$"
        ),
        "url":    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
        "folder": "clip_vision",
        "size":   3689912664,
        "source": "huggingface",
        "title":  "h94/IP-Adapter (sdxl_models/image_encoder)",
    },
    {
        # Wan2.2 / SigCLIP H model, often referenced as clip_vision_h.safetensors
        "pattern": re.compile(r"^(?:sigclip[_]?vision[_]?(?:patch14[_]?)?384"
                               r"|clip[_]?vision[_]?h)\.safetensors$"),
        "url":    "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
        "folder": "clip_vision",
        "size":   856506240,
        "source": "huggingface",
        "title":  "Comfy-Org/sigclip_vision_384",
    },
    {
        # CLIP-G for SDXL refiner / StableCascade
        "pattern": re.compile(r"^clip[_]?vision[_]?g\.safetensors$"),
        "url":    "https://huggingface.co/comfyanonymous/clip_vision_g/resolve/main/clip_vision_g.safetensors",
        "folder": "clip_vision",
        "size":   3689912664,
        "source": "huggingface",
        "title":  "comfyanonymous/clip_vision_g",
    },

    # -----------------------------------------------------------------
    # Depth-Anything family (custom_nodes/comfyui_controlnet_aux/ckpts/)
    # -----------------------------------------------------------------
    # V1 - LiheYoung/depth_anything_<size>14 stored as pytorch_model.bin
    {
        "pattern": re.compile(r"^depth[_]?anything[_]?(vitl|vitb|vits)14\.(?:pth|bin|safetensors)$"),
        "url":    "https://huggingface.co/LiheYoung/depth_anything_\\g<1>14/resolve/main/pytorch_model.bin",
        "folder": "controlnet_aux",
        "source": "huggingface",
        "title":  "LiheYoung/depth_anything_\\g<1>14",
    },
    # V2 - depth-anything/Depth-Anything-V2-{Large,Base,Small}, file already named .pth
    {
        "pattern": re.compile(r"^depth[_]?anything[_]?v2[_]?(vitl|vitb|vits)\.pth$"),
        "url":    "https://huggingface.co/depth-anything/Depth-Anything-V2-{SIZE}/resolve/main/depth_anything_v2_\\g<1>.pth",
        "folder": "controlnet_aux",
        "source": "huggingface",
        "title":  "depth-anything/Depth-Anything-V2-\\g<1>",
        # SIZE depends on the captured short name; resolved at render time:
        "_size_map": {"vitl": "Large", "vitb": "Base", "vits": "Small"},
    },

    # -----------------------------------------------------------------
    # ControlNet v1.1 (lllyasviel) - canonical SD15 controlnet pack
    # -----------------------------------------------------------------
    {
        "pattern": re.compile(r"^control[_]?v11(?:[fp][1ep]?)?p?[_]?sd15[_]?(\w+)\.(?:pth|safetensors)$"),
        "url":    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_\\g<1>.pth",
        "folder": "controlnet",
        "source": "huggingface",
        "title":  "lllyasviel/ControlNet-v1-1 \\g<1>",
    },

    # -----------------------------------------------------------------
    # AnimateLCM by wangfuyun - canonical T2V motion module + LoRA
    # -----------------------------------------------------------------
    {
        "pattern": re.compile(r"^animatelcm[_]?sd15[_]?t2v\.ckpt$"),
        "url":    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt",
        "folder": "animatediff_models",
        "size":   1813021123,
        "source": "huggingface",
        "title":  "wangfuyun/AnimateLCM (T2V motion module)",
    },
    {
        "pattern": re.compile(r"^animatelcm[_]?sd15[_]?t2v[_]?lora\.safetensors$"),
        "url":    "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors",
        "folder": "loras",
        "size":   134579344,
        "source": "huggingface",
        "title":  "wangfuyun/AnimateLCM (T2V LoRA)",
    },
    {
        "pattern": re.compile(r"^animatelcm[_]?i2v\.ckpt$"),
        "url":    "https://huggingface.co/wangfuyun/AnimateLCM-I2V/resolve/main/AnimateLCM_sd15_i2v.ckpt",
        "folder": "animatediff_models",
        "source": "huggingface",
        "title":  "wangfuyun/AnimateLCM-I2V",
    },

    # -----------------------------------------------------------------
    # AnimateDiff motion modules (guoyww/animatediff)
    # mm_sd_v14, mm_sd_v15, mm_sd_v15_v2, v3_sd15_mm, mm_sdxl_v10_beta
    # -----------------------------------------------------------------
    {
        "pattern": re.compile(r"^mm[_]?sd[_]?(v14|v15|v15_v2)\.ckpt$"),
        "url":    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_\\g<1>.ckpt",
        "folder": "animatediff_models",
        "source": "huggingface",
        "title":  "guoyww/animatediff mm_sd_\\g<1>",
    },
    {
        "pattern": re.compile(r"^mm[_]?sdxl[_]?v10[_]?beta\.ckpt$"),
        "url":    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt",
        "folder": "animatediff_models",
        "source": "huggingface",
        "title":  "guoyww/animatediff SDXL v1.0 beta",
    },
    {
        "pattern": re.compile(r"^v3[_]?sd15[_]?(mm|adapter|sparsectrl_rgb|sparsectrl_scribble)\.ckpt$"),
        "url":    "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_\\g<1>.ckpt",
        "folder": "animatediff_models",
        "source": "huggingface",
        "title":  "guoyww/animatediff v3_sd15_\\g<1>",
    },
    # Motion LoRAs (camera control): v2_lora_PanLeft.ckpt, etc. We list
    # them explicitly so the case-sensitive part of the filename
    # (PanLeft, ZoomIn, ...) maps to the exact case HuggingFace expects -
    # capture-group backrefs would always be lower-cased by our
    # normaliser before matching.
    *[
        {
            "pattern": re.compile(rf"^v2[_]?lora[_]?{name.lower()}\.ckpt$"),
            "url":    f"https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_{name}.ckpt",
            "folder": "animatediff_motion_lora",
            "source": "huggingface",
            "title":  f"guoyww/animatediff motion LoRA {name}",
        }
        for name in (
            "PanLeft", "PanRight",
            "ZoomIn", "ZoomOut",
            "TiltUp", "TiltDown",
            "RollingClockwise", "RollingAnticlockwise",
        )
    ],

    # -----------------------------------------------------------------
    # Real-ESRGAN family of upscalers
    # -----------------------------------------------------------------
    {
        "pattern": re.compile(r"^realesrgan[_]?x4plus[_]?anime[_]?6b\.pth$"),
        "url":    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B.pth",
        "folder": "upscale_models",
        "size":   17938799,
        "source": "huggingface",
        "title":  "ai-forever/Real-ESRGAN (anime 6B)",
    },
    {
        "pattern": re.compile(r"^realesrgan[_]?x2plus\.pth$"),
        "url":    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth",
        "folder": "upscale_models",
        "source": "huggingface",
        "title":  "ai-forever/Real-ESRGAN (x2)",
    },
    {
        "pattern": re.compile(r"^realesr[_]?general[_]?x4v3\.pth$"),
        "url":    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/realesr-general-x4v3.pth",
        "folder": "upscale_models",
        "source": "huggingface",
        "title":  "ai-forever/Real-ESRGAN (general x4 v3)",
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
