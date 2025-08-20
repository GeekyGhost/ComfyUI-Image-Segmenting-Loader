"""
Geeky Qwen Edit Segmenting Nodes for ComfyUI
Interactive image segmentation and composition nodes
"""

from .geeky_qwen_segment_loader import GeekyQwenSegmentLoader
from .geeky_qwen_compositor import GeekyQwenCompositor

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "GeekyQwenSegmentLoader": GeekyQwenSegmentLoader,
    "GeekyQwenCompositor": GeekyQwenCompositor,
}

# Display name mappings for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyQwenSegmentLoader": "🎯 Qwen Segment Loader",
    "GeekyQwenCompositor": "🔧 Qwen Compositor", 
}

# Web directory for JavaScript files  
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']