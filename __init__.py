"""
ComfyUI-ModelDownloader
A custom node / extension that scans workflows for missing models
and downloads them automatically from HuggingFace and CivitAI.
"""

from .server import register_routes
from .separator_compat import install as _install_separator_compat

# Register HTTP routes on import
register_routes()

# On Windows, expose forward-slash variants for filename lists so workflows
# authored with "/" separators don't appear missing in ComfyUI loader nodes.
_install_separator_compat()

# Tell ComfyUI where the JS frontend lives
WEB_DIRECTORY = "./web"

# No custom nodes are registered, but ComfyUI expects these symbols.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
