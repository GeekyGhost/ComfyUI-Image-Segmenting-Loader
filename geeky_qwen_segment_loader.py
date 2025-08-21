import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import folder_paths
import os
import json
import hashlib

class GeekyQwenSegmentLoader:
    """
    Image loader with auto-updating visual coordinate grid. 
    Preview automatically updates when you change X, Y, width, or height parameters.
    """
    
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ([""], {"image_upload": True}),
            },
            "optional": {
                "segment_x": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 1,
                    "display": "number"
                }),
                "segment_y": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 1,
                    "display": "number"
                }),
                "segment_width": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 2048, 
                    "step": 8,
                    "display": "number"
                }),
                "segment_height": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 2048, 
                    "step": 8,
                    "display": "number"
                }),
                "target_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "display": "number"
                }),
                "grid_spacing": ("INT", {
                    "default": 100,
                    "min": 25,
                    "max": 200,
                    "step": 25,
                    "display": "number",
                    "tooltip": "Spacing between grid lines in pixels"
                }),
                "show_coordinates": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Show Grid Numbers",
                    "label_off": "Hide Grid Numbers"
                }),
                "enable_auto_preview": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto Update Preview",
                    "label_off": "Manual Update Only"
                }),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("original_image", "segment_image", "segment_metadata", "segment_width", "segment_height", "coordinate_preview")
    OUTPUT_NODE = False
    FUNCTION = "load_and_segment"
    CATEGORY = "🎯 Geeky Qwen Edit"

    @classmethod
    def IS_CHANGED(cls, image, segment_x=0, segment_y=0, segment_width=512, 
                   segment_height=512, grid_spacing=100, show_coordinates=True, 
                   enable_auto_preview=True, **kwargs):
        """
        This method tells ComfyUI when to re-execute the node.
        Returns a hash that changes when preview-relevant parameters change.
        """
        if not enable_auto_preview:
            # If auto-preview is disabled, only update when image changes
            if image:
                image_path = folder_paths.get_annotated_filepath(image)
                if os.path.exists(image_path):
                    return str(os.path.getmtime(image_path))
            return ""
        
        # Create a hash of all parameters that affect the preview
        params_to_hash = f"{image}_{segment_x}_{segment_y}_{segment_width}_{segment_height}_{grid_spacing}_{show_coordinates}"
        
        # Also include image file modification time if it exists
        if image:
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.exists(image_path):
                params_to_hash += f"_{os.path.getmtime(image_path)}"
        
        # Return hash of parameters - when any change, ComfyUI will re-execute
        return hashlib.md5(params_to_hash.encode()).hexdigest()

    def load_and_segment(self, image, segment_x=0, segment_y=0, segment_width=512, 
                        segment_height=512, target_size=1024, grid_spacing=100, 
                        show_coordinates=True, enable_auto_preview=True, 
                        prompt=None, extra_pnginfo=None):
        
        # Load the image
        image_path = folder_paths.get_annotated_filepath(image)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size
        
        # Ensure segment coordinates are within image bounds
        segment_x = max(0, min(segment_x, original_width - 1))
        segment_y = max(0, min(segment_y, original_height - 1))
        segment_width = min(segment_width, original_width - segment_x)
        segment_height = min(segment_height, original_height - segment_y)
        
        # Create coordinate preview image (this updates automatically when parameters change)
        coordinate_preview = self.create_coordinate_preview(
            img, segment_x, segment_y, segment_width, segment_height, 
            grid_spacing, show_coordinates
        )
        
        # Extract the segment - ONLY the cropped section, no padding
        segment_box = (
            segment_x,
            segment_y,
            segment_x + segment_width,
            segment_y + segment_height
        )
        
        segment_img = img.crop(segment_box)
        
        # Optionally resize the segment to target size while maintaining aspect ratio
        # But keep it as the actual cropped content, not padded to a square
        if target_size and (segment_img.width > target_size or segment_img.height > target_size):
            segment_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert images to tensors
        original_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        segment_tensor = torch.from_numpy(np.array(segment_img).astype(np.float32) / 255.0).unsqueeze(0)
        coordinate_tensor = torch.from_numpy(np.array(coordinate_preview).astype(np.float32) / 255.0).unsqueeze(0)
        
        # Create metadata for the compositor
        metadata = {
            "original_size": [original_width, original_height],
            "segment_coords": [segment_x, segment_y, segment_width, segment_height],
            "segment_original_size": [segment_width, segment_height],  # Original cropped size
            "segment_current_size": [segment_img.width, segment_img.height],  # After any resizing
            "target_size": target_size,
            "grid_params": {
                "spacing": grid_spacing,
                "show_coords": show_coordinates
            }
        }
        
        metadata_str = json.dumps(metadata)
        
        # Log updates when auto-preview is enabled
        if enable_auto_preview:
            print(f"🎯 Preview Updated: {segment_width}×{segment_height} at ({segment_x}, {segment_y})")
        else:
            print(f"🎯 Selected: {segment_width}×{segment_height} at ({segment_x}, {segment_y})")
        
        return (original_tensor, segment_tensor, metadata_str, segment_width, segment_height, coordinate_tensor)
    
    def create_coordinate_preview(self, img, sel_x, sel_y, sel_w, sel_h, grid_spacing, show_coords):
        """Create a preview image with coordinate grid overlay"""
        
        # Create a copy for the preview
        preview = img.copy()
        draw = ImageDraw.Draw(preview)
        
        # Get image dimensions
        width, height = img.size
        
        # Calculate font size based on image size
        font_size = max(8, min(width, height) // 80)
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # Linux
                except:
                    font = ImageFont.load_default()
        
        # Draw grid lines
        grid_color = (255, 255, 255, 180)  # Semi-transparent white
        coord_color = (255, 215, 0)  # Gold for coordinates
        
        # Vertical grid lines
        for x in range(0, width + 1, grid_spacing):
            if x <= width:
                draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
                
                # Draw coordinate numbers
                if show_coords and x < width:
                    coord_text = str(x)
                    bbox = draw.textbbox((0, 0), coord_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    
                    # Top coordinate
                    draw.rectangle([x-2, 0, x + text_width + 2, bbox[3] + 4], 
                                 fill=(0, 0, 0, 160))
                    draw.text((x, 2), coord_text, fill=coord_color, font=font)
                    
                    # Bottom coordinate  
                    draw.rectangle([x-2, height - bbox[3] - 4, x + text_width + 2, height], 
                                 fill=(0, 0, 0, 160))
                    draw.text((x, height - bbox[3] - 2), coord_text, fill=coord_color, font=font)
        
        # Horizontal grid lines
        for y in range(0, height + 1, grid_spacing):
            if y <= height:
                draw.line([(0, y), (width, y)], fill=grid_color, width=1)
                
                # Draw coordinate numbers
                if show_coords and y < height:
                    coord_text = str(y)
                    bbox = draw.textbbox((0, 0), coord_text, font=font)
                    text_height = bbox[3] - bbox[1]
                    text_width = bbox[2] - bbox[0]
                    
                    # Left coordinate
                    draw.rectangle([0, y-2, text_width + 4, y + text_height + 2], 
                                 fill=(0, 0, 0, 160))
                    draw.text((2, y), coord_text, fill=coord_color, font=font)
                    
                    # Right coordinate
                    draw.rectangle([width - text_width - 4, y-2, width, y + text_height + 2], 
                                 fill=(0, 0, 0, 160))
                    draw.text((width - text_width - 2, y), coord_text, fill=coord_color, font=font)
        
        # Draw current selection with bright highlight
        selection_color = (255, 0, 100)  # Bright pink/red
        selection_width = max(2, min(width, height) // 500)
        
        # Draw selection rectangle
        for i in range(selection_width):
            draw.rectangle([sel_x - i, sel_y - i, sel_x + sel_w + i, sel_y + sel_h + i], 
                         outline=selection_color, width=1)
        
        # Draw selection corner markers
        corner_size = max(8, min(width, height) // 150)
        corners = [
            (sel_x, sel_y),
            (sel_x + sel_w, sel_y),
            (sel_x, sel_y + sel_h),
            (sel_x + sel_w, sel_y + sel_h)
        ]
        
        for corner_x, corner_y in corners:
            draw.rectangle([corner_x - corner_size//2, corner_y - corner_size//2,
                          corner_x + corner_size//2, corner_y + corner_size//2],
                         fill=selection_color, outline=(255, 255, 255), width=1)
        
        # Draw selection info box
        info_text = f"Selection: {sel_w}×{sel_h} at ({sel_x},{sel_y})"
        info_bbox = draw.textbbox((0, 0), info_text, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        info_height = info_bbox[3] - info_bbox[1]
        
        # Position info box in top-left corner
        info_x = 10
        info_y = 10
        
        # Draw info background
        draw.rectangle([info_x - 5, info_y - 3, info_x + info_width + 5, info_y + info_height + 3], 
                     fill=(0, 0, 0, 200), outline=selection_color, width=2)
        
        # Draw info text
        draw.text((info_x, info_y), info_text, fill=(255, 255, 255), font=font)
        
        # Add grid info
        grid_info = f"Grid: {grid_spacing}px spacing"
        grid_bbox = draw.textbbox((0, 0), grid_info, font=font)
        grid_width = grid_bbox[2] - grid_bbox[0]
        grid_height = grid_bbox[3] - grid_bbox[1]
        
        grid_y = info_y + info_height + 10
        
        draw.rectangle([info_x - 5, grid_y - 3, info_x + grid_width + 5, grid_y + grid_height + 3], 
                     fill=(0, 0, 0, 180), outline=coord_color, width=1)
        draw.text((info_x, grid_y), grid_info, fill=coord_color, font=font)
        
        # Add auto-update status
        if hasattr(self, '_last_update_mode'):
            status_info = "🔄 Auto-updating" 
        else:
            status_info = "📌 Manual mode"
            
        status_bbox = draw.textbbox((0, 0), status_info, font=font)
        status_width = status_bbox[2] - status_bbox[0]
        status_height = status_bbox[3] - status_bbox[1]
        
        status_y = grid_y + grid_height + 10
        
        draw.rectangle([info_x - 5, status_y - 3, info_x + status_width + 5, status_y + status_height + 3], 
                     fill=(0, 0, 0, 180), outline=(100, 255, 100), width=1)
        draw.text((info_x, status_y), status_info, fill=(100, 255, 100), font=font)
        
        return preview

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not image:
            return "No image selected"
        
        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path):
            return f"Image file not found: {image_path}"
            
        return True