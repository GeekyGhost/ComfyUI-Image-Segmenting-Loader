import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps, ImageChops
import json
import math

class GeekyQwenEffects:
    """
    Advanced image effects processor optimized for images with alpha channels.
    Perfect for post-processing segments with removed backgrounds.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Drop Shadow
                "drop_shadow": ("BOOLEAN", {"default": False}),
                "shadow_offset_x": ("INT", {"default": 5, "min": -50, "max": 50, "step": 1}),
                "shadow_offset_y": ("INT", {"default": 5, "min": -50, "max": 50, "step": 1}),
                "shadow_blur": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "shadow_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_color": (["black", "white", "red", "green", "blue", "yellow", "purple", "orange"], {"default": "black"}),
                
                # Glow Effects
                "outer_glow": ("BOOLEAN", {"default": False}),
                "glow_radius": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1}),
                "glow_intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "glow_color": (["white", "blue", "red", "green", "yellow", "purple", "orange", "cyan"], {"default": "white"}),
                
                # Stroke/Outline
                "stroke": ("BOOLEAN", {"default": False}),
                "stroke_width": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "stroke_color": (["black", "white", "red", "green", "blue", "yellow", "purple", "orange"], {"default": "black"}),
                
                # Color Adjustments
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "hue_shift": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                
                # Blur and Sharpen
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sharpen": ("BOOLEAN", {"default": False}),
                "sharpen_amount": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                
                # Emboss and Bevel
                "emboss": ("BOOLEAN", {"default": False}),
                "emboss_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                
                # Color Overlay
                "color_overlay": ("BOOLEAN", {"default": False}),
                "overlay_color": (["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"], {"default": "blue"}),
                "overlay_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light"], {"default": "overlay"}),
                
                # Noise and Texture
                "add_noise": ("BOOLEAN", {"default": False}),
                "noise_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Artistic Effects
                "posterize": ("BOOLEAN", {"default": False}),
                "posterize_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                
                "edge_enhance": ("BOOLEAN", {"default": False}),
                "edge_enhance_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                
                # Vignette
                "vignette": ("BOOLEAN", {"default": False}),
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "vignette_radius": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    OUTPUT_NODE = False
    FUNCTION = "apply_effects"
    CATEGORY = "🎯 Geeky Qwen Edit"

    def apply_effects(self, image, drop_shadow=False, shadow_offset_x=5, shadow_offset_y=5, 
                     shadow_blur=10, shadow_opacity=0.5, shadow_color="black",
                     outer_glow=False, glow_radius=15, glow_intensity=0.8, glow_color="white",
                     stroke=False, stroke_width=3, stroke_color="black",
                     brightness=1.0, contrast=1.0, saturation=1.0, hue_shift=0,
                     blur_radius=0.0, sharpen=False, sharpen_amount=1.0,
                     emboss=False, emboss_strength=1.0,
                     color_overlay=False, overlay_color="blue", overlay_opacity=0.3, overlay_mode="overlay",
                     add_noise=False, noise_amount=0.1,
                     posterize=False, posterize_levels=8,
                     edge_enhance=False, edge_enhance_strength=1.0,
                     vignette=False, vignette_strength=0.5, vignette_radius=0.7):
        
        # Convert tensor to PIL
        pil_image = self.tensor_to_pil(image)
        
        # Ensure we have an RGBA image to work with alpha
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        original_alpha = pil_image.split()[-1] if pil_image.mode == 'RGBA' else None
        
        # Create working canvas
        width, height = pil_image.size
        result = Image.new('RGBA', (width + 100, height + 100), (0, 0, 0, 0))  # Larger canvas for effects
        
        # Center the original image
        center_x, center_y = 50, 50
        
        # Apply effects in order
        working_image = pil_image.copy()
        
        # 1. Drop Shadow (applied first, behind everything)
        if drop_shadow:
            shadow = self.create_drop_shadow(working_image, shadow_offset_x, shadow_offset_y, 
                                           shadow_blur, shadow_opacity, shadow_color)
            result.paste(shadow, (center_x + shadow_offset_x, center_y + shadow_offset_y), shadow)
        
        # 2. Outer Glow (behind main image)
        if outer_glow:
            glow = self.create_outer_glow(working_image, glow_radius, glow_intensity, glow_color)
            result.paste(glow, (center_x, center_y), glow)
        
        # 3. Apply color adjustments to main image
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(working_image)
            working_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(working_image)
            working_image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(working_image)
            working_image = enhancer.enhance(saturation)
        
        if hue_shift != 0:
            working_image = self.shift_hue(working_image, hue_shift)
        
        # 4. Blur effects
        if blur_radius > 0:
            # Preserve alpha while blurring
            rgb = working_image.convert('RGB')
            alpha = working_image.split()[-1]
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            working_image = rgb.copy()
            working_image.putalpha(alpha)
        
        # 5. Sharpen
        if sharpen:
            working_image = self.apply_sharpen(working_image, sharpen_amount)
        
        # 6. Emboss
        if emboss:
            working_image = self.apply_emboss(working_image, emboss_strength)
        
        # 7. Edge Enhancement
        if edge_enhance:
            working_image = self.apply_edge_enhance(working_image, edge_enhance_strength)
        
        # 8. Posterize
        if posterize:
            working_image = self.apply_posterize(working_image, posterize_levels)
        
        # 9. Add Noise
        if add_noise:
            working_image = self.add_noise_effect(working_image, noise_amount)
        
        # 10. Color Overlay
        if color_overlay:
            working_image = self.apply_color_overlay(working_image, overlay_color, 
                                                   overlay_opacity, overlay_mode)
        
        # 11. Vignette
        if vignette:
            working_image = self.apply_vignette(working_image, vignette_strength, vignette_radius)
        
        # 12. Stroke (applied last, on top)
        if stroke:
            working_image = self.apply_stroke(working_image, stroke_width, stroke_color)
        
        # Paste the main processed image
        result.paste(working_image, (center_x, center_y), working_image)
        
        # Crop back to original size (removing the padding we added)
        final_result = result.crop((center_x, center_y, center_x + width, center_y + height))
        
        # Convert back to tensor
        return (self.pil_to_tensor(final_result),)
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    
    def get_color_rgb(self, color_name):
        """Convert color name to RGB tuple"""
        colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (255, 0, 255),
            "orange": (255, 165, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255)
        }
        return colors.get(color_name, (255, 255, 255))
    
    def create_drop_shadow(self, image, offset_x, offset_y, blur_radius, opacity, color):
        """Create a drop shadow effect"""
        shadow_color = self.get_color_rgb(color)
        
        # Create shadow mask from alpha channel
        if image.mode == 'RGBA':
            alpha = image.split()[-1]
        else:
            alpha = Image.new('L', image.size, 255)
        
        # Create shadow
        shadow = Image.new('RGBA', image.size, shadow_color + (0,))
        shadow.putalpha(alpha)
        
        # Apply blur
        if blur_radius > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Apply opacity
        if opacity < 1.0:
            shadow_alpha = shadow.split()[-1].point(lambda x: int(x * opacity))
            shadow.putalpha(shadow_alpha)
        
        return shadow
    
    def create_outer_glow(self, image, radius, intensity, color):
        """Create an outer glow effect"""
        glow_color = self.get_color_rgb(color)
        
        # Get alpha channel
        if image.mode == 'RGBA':
            alpha = image.split()[-1]
        else:
            alpha = Image.new('L', image.size, 255)
        
        # Create glow
        glow = Image.new('RGBA', image.size, glow_color + (0,))
        glow.putalpha(alpha)
        
        # Apply blur
        glow = glow.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Enhance intensity
        if intensity != 1.0:
            glow_alpha = glow.split()[-1].point(lambda x: min(255, int(x * intensity)))
            glow.putalpha(glow_alpha)
        
        return glow
    
    def apply_stroke(self, image, width, color):
        """Apply stroke/outline effect"""
        stroke_color = self.get_color_rgb(color)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha channel
        alpha = image.split()[-1]
        
        # Create stroke by expanding the alpha channel
        stroke_alpha = alpha
        for i in range(width):
            stroke_alpha = stroke_alpha.filter(ImageFilter.MaxFilter(3))
        
        # Create stroke image
        stroke = Image.new('RGBA', image.size, stroke_color + (255,))
        stroke.putalpha(stroke_alpha)
        
        # Composite stroke behind original
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        result = Image.alpha_composite(result, stroke)
        result = Image.alpha_composite(result, image)
        
        return result
    
    def shift_hue(self, image, shift_degrees):
        """Shift the hue of the image"""
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image.convert('RGB')
            alpha = None
        
        # Convert to HSV
        hsv = rgb.convert('HSV')
        h, s, v = hsv.split()
        
        # Shift hue
        h_array = np.array(h)
        h_array = (h_array + shift_degrees) % 180
        h = Image.fromarray(h_array.astype(np.uint8))
        
        # Merge back
        result = Image.merge('HSV', (h, s, v)).convert('RGB')
        
        if alpha:
            result = result.convert('RGBA')
            result.putalpha(alpha)
        
        return result
    
    def apply_sharpen(self, image, amount):
        """Apply sharpening effect"""
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image
            alpha = None
        
        # Create sharpen kernel
        kernel = ImageFilter.Kernel((3, 3), 
                                   [-1, -1, -1,
                                    -1, 8 + amount, -1,
                                    -1, -1, -1])
        
        sharpened = rgb.filter(kernel)
        
        if alpha:
            sharpened = sharpened.convert('RGBA')
            sharpened.putalpha(alpha)
        
        return sharpened
    
    def apply_emboss(self, image, strength):
        """Apply emboss effect"""
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image
            alpha = None
        
        embossed = rgb.filter(ImageFilter.EMBOSS)
        
        # Blend with original based on strength
        if strength < 1.0:
            embossed = Image.blend(rgb, embossed, strength)
        
        if alpha:
            embossed = embossed.convert('RGBA')
            embossed.putalpha(alpha)
        
        return embossed
    
    def apply_edge_enhance(self, image, strength):
        """Apply edge enhancement"""
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image
            alpha = None
        
        enhanced = rgb.filter(ImageFilter.EDGE_ENHANCE)
        
        if strength != 1.0:
            enhanced = Image.blend(rgb, enhanced, strength)
        
        if alpha:
            enhanced = enhanced.convert('RGBA')
            enhanced.putalpha(alpha)
        
        return enhanced
    
    def apply_posterize(self, image, levels):
        """Apply posterize effect"""
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image
            alpha = None
        
        posterized = ImageOps.posterize(rgb, levels)
        
        if alpha:
            posterized = posterized.convert('RGBA')
            posterized.putalpha(alpha)
        
        return posterized
    
    def add_noise_effect(self, image, amount):
        """Add noise to the image"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        width, height = image.size
        noise = np.random.randint(0, int(255 * amount), (height, width, 3))
        noise_image = Image.fromarray(noise.astype(np.uint8), 'RGB').convert('RGBA')
        
        # Blend noise with original
        result = Image.blend(image.convert('RGB'), noise_image.convert('RGB'), amount)
        result = result.convert('RGBA')
        result.putalpha(image.split()[-1])
        
        return result
    
    def apply_color_overlay(self, image, color, opacity, mode):
        """Apply color overlay with blend mode"""
        overlay_color = self.get_color_rgb(color)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, overlay_color + (int(255 * opacity),))
        
        # Apply blend mode
        if mode == "multiply":
            result = ImageChops.multiply(image.convert('RGB'), overlay.convert('RGB'))
        elif mode == "screen":
            result = ImageChops.screen(image.convert('RGB'), overlay.convert('RGB'))
        elif mode == "overlay":
            result = ImageChops.overlay(image.convert('RGB'), overlay.convert('RGB'))
        elif mode == "soft_light":
            result = ImageChops.soft_light(image.convert('RGB'), overlay.convert('RGB'))
        elif mode == "hard_light":
            result = ImageChops.hard_light(image.convert('RGB'), overlay.convert('RGB'))
        else:  # normal
            result = Image.alpha_composite(image, overlay)
            return result
        
        result = result.convert('RGBA')
        result.putalpha(image.split()[-1])
        
        return result
    
    def apply_vignette(self, image, strength, radius):
        """Apply vignette effect"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        
        # Create vignette mask
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        max_distance = math.sqrt(center_x**2 + center_y**2) * radius
        
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < max_distance:
                    alpha = 1.0 - (distance / max_distance) * strength
                    alpha = max(0, min(1, alpha))
                    mask.putpixel((x, y), int(255 * alpha))
                else:
                    mask.putpixel((x, y), int(255 * (1 - strength)))
        
        # Apply vignette
        result = image.copy()
        original_alpha = result.split()[-1]
        
        # Combine original alpha with vignette mask
        combined_alpha = ImageChops.multiply(original_alpha, mask)
        result.putalpha(combined_alpha)
        
        return result