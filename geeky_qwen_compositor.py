import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageChops
import json

class GeekyQwenCompositor:
    """
    Composites an edited segment back onto the original image using the metadata
    from the GeekyQwenSegmentLoader. Handles scaling and positioning automatically.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_segment": ("IMAGE",),
                "segment_metadata": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"], {
                    "default": "normal"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "feather_edges": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    OUTPUT_NODE = False
    FUNCTION = "composite_images"
    CATEGORY = "🎯 Geeky Qwen Edit"

    def composite_images(self, original_image, edited_segment, segment_metadata, 
                        blend_mode="normal", opacity=1.0, feather_edges=0):
        
        # Parse metadata
        try:
            metadata = json.loads(segment_metadata)
        except json.JSONDecodeError:
            raise ValueError("Invalid metadata format")
        
        # Convert tensors to PIL Images
        original_pil = self.tensor_to_pil(original_image)
        edited_pil = self.tensor_to_pil(edited_segment)
        
        # Extract metadata
        original_width, original_height = metadata["original_size"]
        segment_x, segment_y, segment_width, segment_height = metadata["segment_coords"]
        
        # Get the original segment size and current edited size
        segment_original_size = metadata.get("segment_original_size", [segment_width, segment_height])
        segment_current_size = metadata.get("segment_current_size", [edited_pil.width, edited_pil.height])
        
        # Ensure original is RGB for proper compositing
        if original_pil.mode == 'RGBA':
            # Convert RGBA original to RGB with white background
            background = Image.new('RGB', original_pil.size, (255, 255, 255))
            background.paste(original_pil, (0, 0), original_pil)
            original_pil = background
        elif original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')
        
        # Handle alpha channel in edited segment
        has_alpha = edited_pil.mode in ('RGBA', 'LA')
        if not has_alpha and edited_pil.mode != 'RGB':
            edited_pil = edited_pil.convert('RGB')
        
        # Resize the edited segment back to original segment size if needed
        if edited_pil.size != (segment_width, segment_height):
            # Use high-quality resampling and preserve alpha if present
            if has_alpha:
                edited_pil = edited_pil.resize((segment_width, segment_height), Image.Resampling.LANCZOS)
            else:
                edited_pil = edited_pil.resize((segment_width, segment_height), Image.Resampling.LANCZOS)
        
        # Apply feathering if requested
        if feather_edges > 0:
            edited_pil = self.apply_feather(edited_pil, feather_edges)
            has_alpha = True  # Feathering creates alpha
        
        # Create a copy of the original image
        result = original_pil.copy()
        
        # Apply blend mode and opacity if needed
        if blend_mode != "normal" or opacity != 1.0:
            # Get the original segment from the background for blending
            original_segment = original_pil.crop((segment_x, segment_y, 
                                                 segment_x + segment_width, 
                                                 segment_y + segment_height))
            edited_pil = self.apply_blend_mode(edited_pil, original_segment, blend_mode, opacity)
        
        # Handle alpha channel compositing
        if has_alpha or edited_pil.mode == 'RGBA':
            # For alpha channel images, use alpha compositing
            result_rgba = result.convert('RGBA')
            if edited_pil.mode != 'RGBA':
                edited_pil = edited_pil.convert('RGBA')
            
            # Create a temporary canvas for the segment
            temp_canvas = Image.new('RGBA', result_rgba.size, (0, 0, 0, 0))
            temp_canvas.paste(edited_pil, (segment_x, segment_y), edited_pil)
            
            # Alpha composite
            result_rgba = Image.alpha_composite(result_rgba, temp_canvas)
            
            # Convert back to RGB
            final_result = Image.new('RGB', result_rgba.size, (255, 255, 255))
            final_result.paste(result_rgba, (0, 0), result_rgba)
            result = final_result
        else:
            # For non-alpha images, use regular paste
            result.paste(edited_pil, (segment_x, segment_y))
        
        # Convert back to tensor
        result_tensor = self.pil_to_tensor(result)
        
        print(f"🔧 Composited: {edited_pil.mode} segment back to {result.mode} image")
        return (result_tensor,)
    
    def tensor_to_pil(self, tensor):
        """Convert a ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from float [0,1] to uint8 [0,255]
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    
    def apply_feather(self, image, feather_radius):
        """Apply edge feathering to soften the transition"""
        from PIL import ImageFilter, ImageDraw
        
        # Create a mask for feathering
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Create a gradient mask from edges
        width, height = image.size
        
        # Fill the center area with full opacity
        inner_rect = [feather_radius, feather_radius, 
                     width - feather_radius, height - feather_radius]
        
        if inner_rect[0] < inner_rect[2] and inner_rect[1] < inner_rect[3]:
            draw.rectangle(inner_rect, fill=255)
        
        # Create gradient edges
        for i in range(feather_radius):
            alpha = int(255 * (i + 1) / feather_radius)
            # Top edge
            if i < height:
                draw.rectangle([feather_radius, i, width - feather_radius, i + 1], fill=alpha)
            # Bottom edge  
            if height - i - 1 >= 0:
                draw.rectangle([feather_radius, height - i - 1, width - feather_radius, height - i], fill=alpha)
            # Left edge
            if i < width:
                draw.rectangle([i, feather_radius, i + 1, height - feather_radius], fill=alpha)
            # Right edge
            if width - i - 1 >= 0:
                draw.rectangle([width - i - 1, feather_radius, width - i, height - feather_radius], fill=alpha)
        
        # Apply Gaussian blur to smooth the mask
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius/3))
        
        # Apply the mask to the image
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        result.paste(image, (0, 0))
        result.putalpha(mask)
        
        return result
    
    def apply_blend_mode(self, top_image, bottom_image, blend_mode, opacity):
        """Apply various blend modes between two images"""
        from PIL import ImageChops
        
        # Ensure both images are the same size
        if top_image.size != bottom_image.size:
            top_image = top_image.resize(bottom_image.size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed for blending
        if top_image.mode == 'RGBA':
            # For RGBA images, we need to handle alpha separately
            top_rgb = Image.new('RGB', top_image.size, (255, 255, 255))
            top_rgb.paste(top_image, (0, 0), top_image)
            top_alpha = top_image.split()[-1]
        else:
            top_rgb = top_image.convert('RGB')
            top_alpha = None
            
        bottom_rgb = bottom_image.convert('RGB')
        
        if blend_mode == "normal":
            result = top_rgb
        elif blend_mode == "multiply":
            result = ImageChops.multiply(bottom_rgb, top_rgb)
        elif blend_mode == "screen":
            result = ImageChops.screen(bottom_rgb, top_rgb)
        elif blend_mode == "overlay":
            result = ImageChops.overlay(bottom_rgb, top_rgb)
        elif blend_mode == "soft_light":
            result = ImageChops.soft_light(bottom_rgb, top_rgb)
        else:
            result = top_rgb
        
        if opacity < 1.0:
            # Blend with original based on opacity
            result = Image.blend(bottom_rgb, result, opacity)
        
        # Restore alpha channel if it existed
        if top_alpha:
            result = result.convert('RGBA')
            result.putalpha(top_alpha)
        
        return result