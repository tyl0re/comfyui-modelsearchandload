"""
ComfyUI-ModelDownloader
A custom node / extension that scans workflows for missing models
and downloads them automatically from HuggingFace and CivitAI.
"""

from .server import register_routes
from .separator_compat import install as _install_separator_compat
from .lora_trigger_nodes import (
    NODE_CLASS_MAPPINGS as _LORA_NODE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _LORA_NODE_NAMES,
)

# Register HTTP routes on import
register_routes()

# On Windows, expose forward-slash variants for filename lists so workflows
# authored with "/" separators don't appear missing in ComfyUI loader nodes.
_install_separator_compat()

# Tell ComfyUI where the JS frontend lives
WEB_DIRECTORY = "./web"

# Custom nodes: LoRA trigger inspector / selector / merger.
NODE_CLASS_MAPPINGS = dict(_LORA_NODE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS = dict(_LORA_NODE_NAMES)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
