"""Workflow scanner: finds model references that aren't installed locally."""

from __future__ import annotations

import os
from typing import Iterable

try:
    import folder_paths  # provided by ComfyUI at runtime
except ImportError:  # pragma: no cover - allows import outside ComfyUI
    folder_paths = None

from .config import FIELD_TO_FOLDER


# File extensions that we consider "model files"
MODEL_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".onnx", ".gguf", ".sft",
)

# All folder keys we will probe when checking whether a file is *anywhere*
# in ComfyUI's model tree. This avoids false positives where a model is
# saved into a different folder than the field name suggests
# (e.g. a UNet stored under "diffusion_models" but referenced via "unet_name").
_ALL_FOLDER_KEYS = [
    "checkpoints", "loras", "vae", "controlnet", "clip", "clip_vision",
    "unet", "diffusion_models", "upscale_models", "embeddings",
    "style_models", "ipadapter", "gligen", "hypernetworks",
    "vae_approx", "photomaker", "instantid", "insightface",
]


def _looks_like_model_filename(value: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    # Reject things that are obviously not filenames (newlines = prompt text,
    # very long values without an extension at the end of a path component)
    if "\n" in v or len(v) > 500:
        return False
    # Final path component must end with a model extension
    last = v.replace("\\", "/").rstrip("/").split("/")[-1]
    return last.lower().endswith(MODEL_EXTS)


# Filename extensions we consider during the filesystem walk. We use a
# superset of MODEL_EXTS because the local index also has to recognise
# files that ComfyUI's own folder_paths doesn't list (e.g. .onnx files).
_LOCAL_INDEX_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".pkl", ".sft",
    ".onnx", ".gguf", ".engine", ".trt", ".msgpack",
)


def _models_root_dirs() -> list[str]:
    """Return all top-level directories that may contain model files.

    Includes:
      - ComfyUI's main `models/` directory
      - Every directory registered via folder_paths (covers
        extra_model_paths.yaml)
      - `custom_nodes/<pack>/ckpts/` and similar local cache folders
        used by some custom-node packs (notably comfyui_controlnet_aux,
        which keeps its annotator checkpoints inside its own folder).
    """
    roots: list[str] = []
    seen: set[str] = set()

    def add(p: str):
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
        # Main models dir
        try:
            add(folder_paths.models_dir)
        except Exception:
            pass
        # Every registered folder (covers extra_model_paths.yaml too)
        for key in _ALL_FOLDER_KEYS:
            try:
                for p in folder_paths.get_folder_paths(key) or []:
                    # Index the *parent* of the folder so siblings (custom
                    # subfolders like "dwpose", "yolo", "ultralytics") are
                    # also covered. Keep the folder itself too.
                    add(p)
                    add(os.path.dirname(p))
            except Exception:
                continue
        # Custom-node-internal model cache folders. Some packs (e.g.
        # comfyui_controlnet_aux) bundle their annotator checkpoints
        # inside the pack itself rather than under models/. We probe a
        # short list of well-known names per pack so we don't accidentally
        # walk the python source tree.
        try:
            base_dir = getattr(folder_paths, "base_path", None)
            if base_dir:
                custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
                if os.path.isdir(custom_nodes_dir):
                    for entry in os.listdir(custom_nodes_dir):
                        pack_dir = os.path.join(custom_nodes_dir, entry)
                        if not os.path.isdir(pack_dir):
                            continue
                        for cache_name in ("ckpts", "models", "checkpoints"):
                            add(os.path.join(pack_dir, cache_name))
        except Exception:
            pass
    return roots


def _walk_local_files(root: str, max_files: int = 50_000) -> set[str]:
    """Walk a root directory and return lowercased basenames + relative paths
    of every model-extension file found. Capped at max_files to keep the scan
    cheap on huge model libraries."""
    out: set[str] = set()
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for fn in filenames:
            lc = fn.lower()
            if not lc.endswith(_LOCAL_INDEX_EXTS):
                continue
            out.add(lc)
            try:
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/").lower()
                out.add(rel)
            except Exception:
                pass
            count += 1
            if count >= max_files:
                return out
    return out


# Cache of {folder_key: set(filenames-lowercased)} built once per scan.
def _build_local_index() -> dict[str, set[str]]:
    """Return a multi-source index of locally-present model files.

    Three sources are combined:
      1. ComfyUI's folder_paths.get_filename_list for each known folder
         (cheap; populated from registered extensions only).
      2. A direct filesystem walk of every models root directory. This
         catches files with extensions ComfyUI doesn't register (.onnx,
         .gguf, etc.) and files in custom subfolders (e.g. "dwpose",
         "ultralytics") that aren't tied to a known folder key.
      3. Bucketed by folder key so cross-folder lookups still work.
    """
    index: dict[str, set[str]] = {}
    if folder_paths is None:
        return index

    # Source 1: ComfyUI's API
    for key in _ALL_FOLDER_KEYS:
        try:
            files = folder_paths.get_filename_list(key)
        except Exception:
            continue
        s: set[str] = set()
        for f in files:
            f_norm = f.replace("\\", "/")
            s.add(f_norm.lower())
            s.add(os.path.basename(f_norm).lower())
        if s:
            index[key] = s

    # Source 2: filesystem walk over all model roots
    fs_files: set[str] = set()
    for root in _models_root_dirs():
        try:
            fs_files |= _walk_local_files(root)
        except Exception:
            continue
    if fs_files:
        index["_filesystem"] = fs_files

    return index


def _is_locally_present(
    filename: str,
    raw: str,
    index: dict[str, set[str]],
    expected_folder: str | None = None,
) -> bool:
    """True if the file referenced by `raw` exists locally in a way that
    ComfyUI would actually find it.

    Three rules:
      1. If the workflow reference includes a subfolder (e.g.
         "Wan2_2/foo.safetensors"), the file must exist at exactly that
         relative path. ComfyUI's loaders look up the EXACT relative
         string from the workflow.
      2. If we know which folder ComfyUI expects (`expected_folder` =
         "clip_vision", "loras", ...), the file must exist under one of
         the registered paths for that folder. Sitting in the wrong
         folder counts as missing.
      3. Otherwise, any occurrence anywhere is good enough.
    """
    if not index:
        return False
    raw_norm = raw.replace("\\", "/").lstrip("/").lower()
    has_subfolder = "/" in raw_norm
    fn_lc = filename.lower()

    if has_subfolder:
        # Strict relative-path match under the folder ComfyUI will actually
        # query. A file at e.g. models/diffusion_models/ltx2/foo does not
        # satisfy a workflow asking a lora loader for ltx2/foo.
        if expected_folder:
            bucket = index.get(expected_folder)
            if bucket and raw_norm in bucket:
                return True
            if folder_paths is not None:
                try:
                    for d in folder_paths.get_folder_paths(expected_folder) or []:
                        if os.path.isfile(os.path.join(d, *raw_norm.split("/"))):
                            return True
                except Exception:
                    pass
            target_dir = get_target_directory(expected_folder)
            if target_dir and os.path.isfile(os.path.join(target_dir, *raw_norm.split("/"))):
                return True
            return False

        # No expected folder: keep the old broad behaviour.
        for files in index.values():
            if raw_norm in files:
                return True
        return False

    if expected_folder:
        # 1. Bucket keyed by ComfyUI's folder name. Source 1 of the
        #    index only contains files folder_paths surfaces for that
        #    folder (limited by supported_pt_extensions).
        bucket = index.get(expected_folder)
        if bucket and fn_lc in bucket:
            return True
        # 2. Direct filesystem probe of the registered ComfyUI folder
        #    paths for this key. Catches files that folder_paths
        #    doesn't list because of extension filtering (.onnx etc.).
        if folder_paths is not None:
            try:
                for d in folder_paths.get_folder_paths(expected_folder) or []:
                    if os.path.isfile(os.path.join(d, filename)):
                        return True
            except Exception:
                pass
        # 3. The expected folder may be a logical name we map to a
        #    custom-node-internal directory (e.g. 'controlnet_aux' ->
        #    custom_nodes/comfyui_controlnet_aux/ckpts). Probe that
        #    directly. Without this, files downloaded by the manager
        #    would forever appear as missing because the folder isn't
        #    registered with folder_paths.
        target_dir = get_target_directory(expected_folder)
        if target_dir and os.path.isfile(os.path.join(target_dir, filename)):
            return True
        return False

    # No folder hint -> accept any occurrence anywhere.
    for files in index.values():
        if fn_lc in files:
            return True
    return False


def _guess_folder_for_field(field: str, value: str) -> str | None:
    # Filename-based overrides take priority over field-name mapping. This
    # catches cases where a generic field name (e.g. clip_name) is paired
    # with a specifically-named file (e.g. clip_vision_h.safetensors).
    v_lc = (value or "").lower()
    bn_lc = os.path.basename(v_lc.replace("\\", "/"))

    if "clip_vision" in bn_lc or bn_lc.startswith("clip-vision"):
        return "clip_vision"
    if bn_lc.startswith("clip_l") or bn_lc.startswith("clip_g") or "t5xxl" in bn_lc or "umt5" in bn_lc:
        return "text_encoders"
    if "gemma" in v_lc and bn_lc.startswith("model-") and bn_lc.endswith(".safetensors"):
        return "text_encoders"
    if "lora" in bn_lc:
        return "loras"
    if "vae" in bn_lc:
        return "vae"
    if bn_lc.startswith("ltx-2") or bn_lc.startswith("ltx2"):
        return "diffusion_models"

    # Frame interpolation models live in their own folder (registered by
    # ComfyUI core: models/frame_interpolation/). FILM, RIFE, ...
    if bn_lc.startswith("film_") or bn_lc.startswith("film-") or bn_lc.startswith("rife"):
        return "frame_interpolation"

    # Depth Anything checkpoints belong to comfyui_controlnet_aux's local
    # ckpts/ folder. We surface them as 'controlnet_aux' which is just a
    # logical label - the download manager will route them correctly via
    # the known-models DB entry.
    if bn_lc.startswith("depth_anything_") or bn_lc.startswith("depth-anything-"):
        return "controlnet_aux"

    # ControlNet model files frequently start with these prefixes, regardless
    # of which loader node references them.
    if (bn_lc.startswith("control_") or bn_lc.startswith("controlnet")
            or bn_lc.startswith("t2iadapter_") or bn_lc.startswith("t2i-adapter")):
        return "controlnet"

    # Upscaler models. Common naming patterns put these on a clear
    # path - we recognise them regardless of which loader node is in
    # the workflow (Reactor, WAS, multiGPU, Searge, vanilla, etc.).
    if (bn_lc.startswith("realesrgan")
            or bn_lc.startswith("real-esrgan")
            or bn_lc.startswith("realesr_")
            or bn_lc.startswith("real_esrgan")
            or bn_lc.startswith("4x_") or bn_lc.startswith("4x-")
            or bn_lc.startswith("2x_") or bn_lc.startswith("2x-")
            or bn_lc.startswith("8x_") or bn_lc.startswith("8x-")
            or bn_lc.startswith("ultrasharp")
            or "esrgan" in bn_lc
            or "swinir" in bn_lc
            or "ldsr" in bn_lc
            or bn_lc.startswith("nmkd")
            or bn_lc.startswith("anime6b")
            or bn_lc.startswith("4xfacefix")
            or bn_lc.startswith("gfpgan")):
        return "upscale_models"

    # AnimateDiff motion modules vs LoRAs. These are NOT regular
    # checkpoints despite the .ckpt extension. We split here:
    #   *_lora.safetensors -> loras (it's a regular LoRA file applied
    #                          via standard LoraLoader)
    #   everything else    -> animatediff_models
    looks_animatediff_family = (
        bn_lc.startswith("animatelcm")
        or bn_lc.startswith("mm_sd_v")
        or bn_lc.startswith("mm_sdxl_v")
        or bn_lc.startswith("v3_sd15_mm")
        or bn_lc.startswith("v3_sd15_adapter")
        or bn_lc.startswith("v3_sd15_sparsectrl")
        or bn_lc.startswith("temporaldiff-")
        or bn_lc.startswith("hsxl_temporal")
        or "_motion_module" in bn_lc
    )
    if looks_animatediff_family:
        # AnimateLCM_sd15_t2v_lora.safetensors is a regular LoRA, not a
        # motion module. Same for any other ..._lora.* sibling.
        if "_lora" in bn_lc or bn_lc.endswith("_lora.safetensors"):
            return "loras"
        return "animatediff_models"

    # AnimateDiff motion LoRAs (camera control). Live in animatediff_motion_lora.
    if bn_lc.startswith("v2_lora_") or bn_lc.startswith("v3_lora_"):
        return "animatediff_motion_lora"

    if bn_lc.endswith(".onnx"):
        # ONNX files are NEVER checkpoints/loras/etc. Pick a folder based
        # on what the custom-node ecosystem expects:
        #
        # - ComfyUI-WanAnimatePreprocess registers folder "detection" for
        #   BOTH yolo* AND vitpose* / dwpose* models. So we route them
        #   there together. Old DWPose preprocessors used a separate
        #   "dwpose"/"ultralytics" folder; if you only use those, copy
        #   the file from "detection" or symlink.
        if ("yolo" in bn_lc or "ultralytic" in bn_lc
                or "vitpose" in bn_lc or "dwpose" in bn_lc or "rtmpose" in bn_lc):
            return "detection"
        if "rmbg" in bn_lc or "isnet" in bn_lc or "u2net" in bn_lc or "briarmbg" in bn_lc:
            return "rembg"
        if "insightface" in bn_lc or "antelope" in bn_lc or "buffalo" in bn_lc:
            return "insightface"
        if "sam_" in bn_lc or "sam2" in bn_lc or "mobile_sam" in bn_lc:
            return "sams"
        return "onnx"  # generic fallback

    if field in FIELD_TO_FOLDER:
        return FIELD_TO_FOLDER[field]
    f = field.lower()
    if "ckpt" in f or "checkpoint" in f:
        return "checkpoints"
    if "lora" in f:
        return "loras"
    if "vae" in f:
        return "vae"
    if "controlnet" in f or "control_net" in f:
        return "controlnet"
    if "clip_vision" in f:
        return "clip_vision"
    if "clip" in f:
        return "text_encoders"  # ComfyUI's "clip" folder IS text_encoders
    if "unet" in f or "diffusion" in f:
        return "diffusion_models"
    if "upscale" in f:
        return "upscale_models"
    if "embedding" in f:
        return "embeddings"
    if "ipadapter" in f:
        return "ipadapter"
    if "style" in f:
        return "style_models"
    if "gligen" in f:
        return "gligen"
    if "hypernetwork" in f:
        return "hypernetworks"
    return None


def _guess_folder_for_node_type(node_type: str) -> str | None:
    nt = (node_type or "").lower()
    if not nt:
        return None
    if "lora" in nt:
        return "loras"
    if "vae" in nt:
        return "vae"
    if "checkpoint" in nt or "ckpt" in nt:
        return "checkpoints"
    if "controlnet" in nt or "control_net" in nt:
        return "controlnet"
    if "upscale" in nt:
        return "upscale_models"
    if "clip_vision" in nt or "clipvision" in nt:
        return "clip_vision"
    if "clip" in nt or "dualclip" in nt or "tripleclip" in nt:
        return "clip"
    if "unet" in nt or "diffusion" in nt:
        return "diffusion_models"
    if "ipadapter" in nt:
        return "ipadapter"
    if "embedding" in nt:
        return "embeddings"
    if "style" in nt:
        return "style_models"
    if "gligen" in nt:
        return "gligen"
    if "hypernetwork" in nt:
        return "hypernetworks"
    if "photomaker" in nt:
        return "photomaker"
    if "instantid" in nt:
        return "instantid"
    return None


# Node types where we know widgets_values[i] holds a model filename, mapped
# to (index, folder_key). This is how the UI-format workflow tells us what
# a string value means without a field name.
UI_NODE_MODEL_SLOTS: dict[str, list[tuple[int, str]]] = {
    "CheckpointLoaderSimple":        [(0, "checkpoints")],
    "CheckpointLoader":              [(0, "checkpoints")],
    "unCLIPCheckpointLoader":        [(0, "checkpoints")],
    "ImageOnlyCheckpointLoader":     [(0, "checkpoints")],
    "LoraLoader":                    [(0, "loras")],
    "LoraLoaderModelOnly":           [(0, "loras")],
    "VAELoader":                     [(0, "vae")],
    "ControlNetLoader":              [(0, "controlnet")],
    "DiffControlNetLoader":          [(0, "controlnet")],
    # Text encoders. ComfyUI exposes them under the "clip" folder name but
    # the actual on-disk location is models/text_encoders/.
    "CLIPLoader":                    [(0, "text_encoders")],
    "DualCLIPLoader":                [(0, "text_encoders"), (1, "text_encoders")],
    "TripleCLIPLoader":              [(0, "text_encoders"), (1, "text_encoders"), (2, "text_encoders")],
    "CLIPVisionLoader":              [(0, "clip_vision")],
    "UNETLoader":                    [(0, "diffusion_models")],
    "StyleModelLoader":              [(0, "style_models")],
    "UpscaleModelLoader":            [(0, "upscale_models")],
    "GLIGENLoader":                  [(0, "gligen")],
    "HypernetworkLoader":            [(0, "hypernetworks")],
    "PhotoMakerLoader":              [(0, "photomaker")],
    "IPAdapterModelLoader":          [(0, "ipadapter")],
    "IPAdapterUnifiedLoader":        [(0, "ipadapter")],
    # ComfyUI core: frame interpolation (FILM, RIFE, ...). Lives in
    # models/frame_interpolation/.
    "FrameInterpolationModelLoader": [(0, "frame_interpolation")],
    # Kijai's ComfyUI-WanAnimatePreprocess: ViTPose + YOLO ONNX models go
    # into models/detection/ (the node registers this folder itself).
    # Slot 0 = vitpose_model, slot 1 = yolo_model.
    "OnnxDetectionModelLoader":      [(0, "detection"), (1, "detection")],
    # Generic DWPose / OpenPose preprocessors from common custom-node packs.
    # The filename-level override in _guess_folder_for_field refines these
    # further if the value points at a yolo / vitpose / etc. file.
    "DwposeDetector":                [(0, "dwpose"), (1, "ultralytics")],
    "DWPreprocessor":                [(0, "dwpose"), (1, "ultralytics")],
    "OpenposePreprocessor":          [(0, "dwpose")],
    "UltralyticsDetectorProvider":   [(0, "ultralytics")],
    "YOLOWorldModelLoader":          [(0, "ultralytics")],
    # comfyui_controlnet_aux preprocessors that take a checkpoint filename
    # in slot 0. The actual ckpt lives inside the pack at
    # custom_nodes/comfyui_controlnet_aux/ckpts/, mapped via the logical
    # folder name 'controlnet_aux' to that path in get_target_directory().
    "DepthAnythingPreprocessor":     [(0, "controlnet_aux")],
    "Zoe_DepthAnythingPreprocessor": [(0, "controlnet_aux")],
    # AnimateDiff-Evolved loaders. The motion-module ckpt lives in
    # models/animatediff_models/ (registered by the pack).
    "ADE_AnimateDiffLoaderGen1":         [(0, "animatediff_models")],
    "ADE_AnimateDiffLoaderWithContext":  [(0, "animatediff_models")],
    "ADE_LoadAnimateDiffModel":          [(0, "animatediff_models")],
    "ADE_LoadAnimateLCMI2VModel":        [(0, "animatediff_models")],
    "ADE_AnimateDiffLoRALoader":         [(0, "animatediff_motion_lora")],
    "AnimateDiffLoaderV1":                [(0, "animatediff_models")],
    "AnimateDiffModuleLoader":            [(0, "animatediff_models")],
}


def _iter_api_inputs(node: dict) -> Iterable[tuple[str, str]]:
    """API-format node: yield (field_name, value) for string inputs."""
    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, str):
                yield k, v


def _iter_ui_widgets(node: dict) -> Iterable[tuple[str, str, str]]:
    """
    UI-format node: yield (synthetic_field, value, folder_hint).

    Strategy:
      1. If we have a hand-curated slot map for this exact node type,
         use it - that's the most precise.
      2. Otherwise scan EVERY widget value and yield each that looks
         like a model filename. There are too many third-party loader /
         upscaler / preprocessor nodes to whitelist them all
         (Reactor, WAS, multiGPU, Searge, Bjornulf, DTUpscale, ...),
         so we trust the file extension as the primary signal.

    The folder hint is derived from:
      a) node-type heuristic (if the type contains 'upscale', etc.)
      b) filename heuristic (e.g. AnimateLCM_*.ckpt -> animatediff_models),
         which overrides (a) when the filename clearly identifies a
         family.
    """
    node_type = node.get("type") or ""
    wv = node.get("widgets_values")
    if not isinstance(wv, list):
        return
    slots = UI_NODE_MODEL_SLOTS.get(node_type)
    if slots:
        for idx, folder in slots:
            if idx < len(wv) and isinstance(wv[idx], str):
                v = wv[idx]
                # Refine folder via filename-level override (e.g. a value
                # ending in .onnx with "yolo" in it goes to ultralytics
                # regardless of what the slot mapping said).
                refined = _guess_folder_for_field(f"_widget[{idx}]", v) or folder
                yield f"_widget[{idx}]", v, refined
        return

    # Liberal fallback: yield every widget value that looks like a model
    # filename. Filters in _looks_like_model_filename already exclude
    # prompt text, very long strings, and strings with newlines.
    base_hint = _guess_folder_for_node_type(node_type)
    for i, v in enumerate(wv):
        if not isinstance(v, str):
            continue
        if not _looks_like_model_filename(v):
            continue
        # Filename-derived folder always wins - it knows that
        # 'depth_anything_*.pth' is NOT a checkpoint regardless of which
        # node loaded it. node-type heuristic is the fallback when the
        # filename doesn't ring any specific bells. Final fallback is
        # 'checkpoints' so the file at least gets surfaced.
        refined = (
            _guess_folder_for_field(f"_widget[{i}]", v)
            or base_hint
            or "checkpoints"
        )
        yield f"_widget[{i}]", v, refined


def scan_workflow(workflow: dict) -> list[dict]:
    """
    Scan a workflow JSON and return a list of missing models:
        [{ "name": "...", "folder": "checkpoints", "node_type": "...", "field": "..." }, ...]

    Models that exist anywhere in the ComfyUI model tree are excluded from
    the result (cross-folder lookup), so this list is what is *actually* missing.
    """
    if not isinstance(workflow, dict):
        return []

    is_ui_format = "nodes" in workflow and isinstance(workflow["nodes"], list)
    if is_ui_format:
        nodes_iter = workflow["nodes"]
    else:
        nodes_iter = []
        for n in workflow.values():
            if isinstance(n, dict) and ("inputs" in n or "class_type" in n):
                merged = dict(n)
                merged.setdefault("type", n.get("class_type"))
                nodes_iter.append(merged)

    local_index = _build_local_index()
    found: list[dict] = []
    seen: set[str] = set()

    for node in nodes_iter:
        node_type = node.get("type") or node.get("class_type") or ""

        candidates: list[tuple[str, str, str | None]] = []

        if is_ui_format:
            for field, value, folder_hint in _iter_ui_widgets(node):
                candidates.append((field, value, folder_hint))
        else:
            for field, value in _iter_api_inputs(node):
                if not _looks_like_model_filename(value):
                    continue
                folder_hint = _guess_folder_for_field(field, value) \
                    or _guess_folder_for_node_type(node_type)
                candidates.append((field, value, folder_hint))

        for field, value, folder_hint in candidates:
            if not _looks_like_model_filename(value):
                continue
            value_norm = value.replace("\\", "/").strip().lstrip("/")
            basename = os.path.basename(value_norm)
            # Preserve the subfolder portion of the workflow reference. This
            # is critical for files like "Wan2_2/Wan22-...safetensors" where
            # ComfyUI looks at the EXACT relative path under models/<folder>/.
            # If we stripped the subfolder, the file would be downloaded next
            # to it but ComfyUI would still report it as missing.
            subfolder = os.path.dirname(value_norm)  # may be ""
            # Dedupe by full relative path (case-insensitive) so the same
            # file referenced with and without subfolder doesn't appear twice.
            dedupe_key = value_norm.lower() if subfolder else basename.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            folder = folder_hint or "checkpoints"

            # Folder-aware lookup. We pass the expected folder so the check
            # is strict: a file in the wrong folder counts as missing,
            # because ComfyUI's loader only looks in its registered
            # directories. Also matches subfolder-qualified references
            # exactly.
            if _is_locally_present(basename, value_norm, local_index, folder):
                continue

            # Compute the absolute path where the downloader will put
            # this file. Surfacing this lets the user catch wrong-folder
            # routing in the UI before clicking Download.
            try:
                target_dir = get_target_directory(folder)
                if subfolder:
                    parts = [p for p in subfolder.split("/") if p not in ("", ".", "..")]
                    target_dir = os.path.join(target_dir, *parts)
                target_path = os.path.join(target_dir, basename)
            except Exception:
                target_path = None

            found.append({
                "name": basename,
                "raw": value,
                "subfolder": subfolder,
                "folder": folder,
                "node_type": node_type,
                "field": field,
                "target_path": target_path,
            })

    return found


# Logical folder names that don't exist in folder_paths but do correspond
# to a well-known location inside a custom-node pack. Mapped here so the
# download manager routes the file into the right place automatically.
# Each value is a (custom_nodes_subdir, relative_subpath) tuple.
_CUSTOM_NODE_FOLDERS: dict[str, tuple[str, str]] = {
    "controlnet_aux": ("comfyui_controlnet_aux", "ckpts"),
}


def get_target_directory(folder_key: str) -> str:
    """Return the absolute path where a model of the given type should be saved."""
    if folder_paths is not None:
        # 1. Check ComfyUI's registered folders
        try:
            paths = folder_paths.get_folder_paths(folder_key)
            if paths:
                return paths[0]
        except Exception:
            pass

        # 2. Check our custom-node-folder map (for things like
        #    comfyui_controlnet_aux/ckpts which are not registered globally
        #    but ARE the canonical location for a class of models).
        if folder_key in _CUSTOM_NODE_FOLDERS:
            try:
                base_dir = getattr(folder_paths, "base_path", None)
                if base_dir:
                    pack, sub = _CUSTOM_NODE_FOLDERS[folder_key]
                    return os.path.join(base_dir, "custom_nodes", pack, sub)
            except Exception:
                pass

        # 3. Fall back to <models>/<folder_key>/
        try:
            base = folder_paths.models_dir
            return os.path.join(base, folder_key)
        except Exception:
            pass
    # 4. Last resort: relative to CWD
    return os.path.join(os.getcwd(), "models", folder_key)
