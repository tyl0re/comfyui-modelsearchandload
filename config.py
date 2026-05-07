"""Configuration management for the model downloader."""

import json
import os
from pathlib import Path

# Plugin directory
PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = PLUGIN_DIR / "config.json"
KNOWN_MODELS_FILE = PLUGIN_DIR / "known_models.json"
USER_KNOWN_MODELS_FILE = PLUGIN_DIR / "user_known_models.json"

DEFAULT_CONFIG = {
    "huggingface_token": "",
    "civitai_token": "",
    "auto_search": True,
    "concurrent_downloads": 1,
    # Reuse existing files via filesystem links instead of downloading.
    # When True, before each download we look for a same-name + same-size
    # copy anywhere in the models tree. If found, we hardlink (preferred)
    # or symlink the file to its expected location. Saves disk space when
    # a model is referenced from multiple workflows / subfolders.
    "enable_linking": False,
    # "auto" = try hardlink, fall back to symlink, fall back to copy.
    # "hardlink" = only hardlink, fail otherwise.
    # "symlink"  = only symlink.
    "linking_mode": "auto",
    # How to identify duplicate model files for both the dedupe scanner
    # and the post-download de-dupe-and-link step.
    #
    #   "hash"      = same SHA-256 hash. Slow (reads every byte) but
    #                 100% safe; we know the content is identical.
    #   "size_name" = same basename + same size in bytes. Fast (no
    #                 file content read) but risks false positives in
    #                 the rare case of two unrelated files coinciding
    #                 in name and size.
    #   "disabled"  = don't dedupe (the button is hidden, post-download
    #                 hook is a no-op).
    "dedupe_method": "hash",
    # When True, after a successful download (not link) we look for
    # OTHER copies of the same file already in the tree using
    # `dedupe_method`. If a duplicate exists, the freshly-downloaded
    # file is replaced with a hardlink. Only takes effect when
    # `enable_linking` is also True.
    "auto_dedupe_after_download": True,
}


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Fill in any missing keys
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_known_models() -> dict:
    """Load the bundled name -> download-info mapping."""
    if not KNOWN_MODELS_FILE.exists():
        return {}
    try:
        with open(KNOWN_MODELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_user_known_models() -> dict:
    """Load locally learned user selections.

    This file is intentionally separate from the bundled DB so users can
    keep personal workflow fixes without making them look curated upstream.
    """
    if not USER_KNOWN_MODELS_FILE.exists():
        return {}
    try:
        with open(USER_KNOWN_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_known_model(filename: str, entry: dict) -> None:
    if not filename or not isinstance(entry, dict):
        return
    data = load_user_known_models()
    data[filename] = entry
    with open(USER_KNOWN_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# Mapping of node-input fields -> ComfyUI model subfolder
# This is a reasonable default; ComfyUI's folder_paths is the source of truth at runtime.
FIELD_TO_FOLDER = {
    "ckpt_name": "checkpoints",
    "checkpoint_name": "checkpoints",
    "lora_name": "loras",
    "lora_01": "loras",
    "lora_02": "loras",
    "lora_03": "loras",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    # ComfyUI's logical "clip" folder maps to models/text_encoders/ (and
    # models/clip/ as a legacy alias). Note: filename-level overrides in
    # scanner._guess_folder_for_field redirect anything matching clip_vision
    # to the proper clip_vision folder.
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "unet_name": "unet",
    "diffusion_model": "diffusion_models",
    "model_name": "upscale_models",
    "upscale_model_name": "upscale_models",
    "embedding_name": "embeddings",
    "style_model_name": "style_models",
    "ipadapter_file": "ipadapter",
    "clip_vision_name": "clip_vision",
    "gligen_name": "gligen",
    "hypernetwork_name": "hypernetworks",
}
