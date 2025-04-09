from .api_server import run_comfyui_api_tools
from .nodes import SimpleGenImageInterface

NODE_CLASS_MAPPINGS = {
    "SimpleGenImageInterface": SimpleGenImageInterface,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleGenImageInterface": "Simple Gen Image Interface",
}

run_comfyui_api_tools()