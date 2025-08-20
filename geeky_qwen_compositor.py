import torch
import numpy as np
from PIL import Image
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
        segment_size = metadata["segment_size"]
        target_size = metadata["target_size"]
        paste_x, paste_y = metadata["paste_offset"]
        
        # Extract the actual edited content from the target-sized square
        # The edited content should be at the paste offset position
        actual_edited = edited_pil.crop((
            paste_x,
            paste_y,
            paste_x + segment_size[0],
            paste_y + segment_size[1]
        ))
        
        # Resize the edited segment back to original segment size
        if actual_edited.size != (segment_width, segment_height):
            actual_edited = actual_edited.resize((segment_width, segment_height), Image.Resampling.LANCZOS)
        
        # Apply feathering if requested
        if feather_edges > 0:
            actual_edited = self.apply_feather(actual_edited, feather_edges)
        
        # Create a copy of the original image
        result = original_pil.copy()
        
        # Apply blend mode and opacity
        if blend_mode != "normal" or opacity != 1.0:
            actual_edited = self.apply_blend_mode(actual_edited, 
                                                 original_pil.crop((segment_x, segment_y, 
                                                                   segment_x + segment_width, 
                                                                   segment_y + segment_height)),
                                                 blend_mode, opacity)
        
        # Paste the edited segment back onto the original
        result.paste(actual_edited, (segment_x, segment_y))
        
        # Convert back to tensor
        result_tensor = self.pil_to_tensor(result)
        
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
        mask = Image.new('L', image.size, 255)
        draw = ImageDraw.Draw(mask)
        
        # Draw a rectangle with feathered edges
        for i in range(feather_radius):
            alpha = int(255 * (i + 1) / feather_radius)
            draw.rectangle([i, i, image.size[0] - i - 1, image.size[1] - i - 1], 
                          outline=alpha, width=1)
        
        # Apply Gaussian blur to smooth the mask
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius/2))
        
        # Apply the mask to the image
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        result.paste(image, (0, 0))
        result.putalpha(mask)
        
        # Convert back to RGB with white background
        final = Image.new('RGB', image.size, (255, 255, 255))
        final.paste(result, (0, 0), result)
        
        return final
    
    def apply_blend_mode(self, top_image, bottom_image, blend_mode, opacity):
        """Apply various blend modes between two images"""
        from PIL import ImageChops
        
        if blend_mode == "normal":
            result = top_image
        elif blend_mode == "multiply":
            result = ImageChops.multiply(bottom_image, top_image)
        elif blend_mode == "screen":
            result = ImageChops.screen(bottom_image, top_image)
        elif blend_mode == "overlay":
            result = ImageChops.overlay(bottom_image, top_image)
        elif blend_mode == "soft_light":
            result = ImageChops.soft_light(bottom_image, top_image)
        else:
            result = top_image
        
        if opacity < 1.0:
            # Blend with original based on opacity
            result = Image.blend(bottom_image, result, opacity)
        
        return result