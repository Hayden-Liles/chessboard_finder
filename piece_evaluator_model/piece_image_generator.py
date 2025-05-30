from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import os, random, math
import multiprocessing
from tqdm import tqdm
import numpy as np
from functools import lru_cache
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2  # Add opencv-python for faster operations
import warnings
import sys
from contextlib import contextmanager

# Suppress libpng warnings
@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ─── CONFIG ───────────────────────────────────────────────────────────────────
boards_dir   = "boards"
boards_dot_dir = "boards_with_dot"
sprites_dir  = "pieces"
cursors_dir  = "cursors"
out_dir      = "data3/train"

PATCH_SIZE   = 244
N_VARIANTS   = 2500
MAX_CROP     = 8

# ─── ENHANCED AUGMENTATION CONFIG ─────────────────────────────────────────────
MAX_SHIFT = int(PATCH_SIZE * 0.15)
ROTATION_PROBABILITY = 0.7
MAX_ROTATION_DEGREES = 15
SCALE_VARIATION_RANGE = (0.8, 1.3)
SCALE_PROBABILITY = 0.6
PERSPECTIVE_PROBABILITY = 0.3
MAX_PERSPECTIVE_DISTORTION = 0.05
SHADOW_PROBABILITY = 0.5
SHADOW_OFFSET_RANGE = (2, 8)
SHADOW_BLUR_RANGE = (1, 4)
SHADOW_OPACITY_RANGE = (0.2, 0.6)
HIGHLIGHT_PROBABILITY = 0.4
HIGHLIGHT_SIZE_FACTOR_RANGE = (0.1, 0.3)
HIGHLIGHT_INTENSITY_RANGE = (0.3, 0.8)
BLUR_PROBABILITY = 0.2
BLUR_RADIUS_RANGE = (0.5, 2.0)
NOISE_PROBABILITY = 0.3
NOISE_INTENSITY_RANGE = (0.02, 0.08)
JPEG_COMPRESSION_PROBABILITY = 0.2
COMPRESSION_QUALITY_RANGE = (60, 95)
CURSOR_PROBABILITY = 1.0
CURSOR_ACTIVE_PROBABILITY = 0.3
CURSOR_SIZE_RANGE = (0.8, 1.5)
CURSOR_EDGE_PROBABILITY = 0.2
OCCLUSION_PROBABILITY = 0.4
MAX_OCCLUSIONS_PER_IMAGE = 2
MIN_OCCLUSION_SIZE_FACTOR = 0.05
MAX_OCCLUSION_SIZE_FACTOR = 0.20
OCCLUSION_COLOR_RANGES = ((0, 255), (0, 255), (0, 255))
OCCLUSION_ALPHA_RANGE = (70, 180)
COLOR_JITTER_PROBABILITY = 0.6
BRIGHTNESS_FACTOR_RANGE = (0.6, 1.4)
CONTRAST_FACTOR_RANGE = (0.6, 1.4)
SATURATION_FACTOR_RANGE = (0.6, 1.4)

NUM_WORKERS = os.cpu_count()
CHUNK_SIZE = 100  # Process images in chunks for better memory efficiency
# ──────────────────────────────────────────────────────────────────────────────

# Global cache for cursors
_cursor_cache = None

def init_worker(cursor_data):
    """Initialize worker with shared cursor data"""
    global _cursor_cache
    _cursor_cache = cursor_data

def create_default_cursors():
    """Create basic cursor shapes if no cursor directory exists"""
    cursors = {}
    
    # Convert to numpy arrays for faster operations
    # Arrow cursor
    arrow_img = np.zeros((24, 24, 4), dtype=np.uint8)
    arrow_points = np.array([(2, 2), (2, 18), (8, 12), (12, 16), (16, 12), (10, 6), (16, 2)])
    cv2.fillPoly(arrow_img, [arrow_points], (255, 255, 255, 255))
    cv2.polylines(arrow_img, [arrow_points], True, (0, 0, 0, 255), 1)
    cursors['arrow'] = {'normal': arrow_img, 'active': arrow_img}
    
    # Hand cursor
    hand_img = np.zeros((24, 24, 4), dtype=np.uint8)
    cv2.ellipse(hand_img, (12, 14), (6, 6), 0, 0, 360, (255, 220, 180, 255), -1)
    cv2.ellipse(hand_img, (12, 14), (6, 6), 0, 0, 360, (0, 0, 0, 255), 1)
    cursors['hand'] = {'normal': hand_img, 'active': hand_img}
    
    # Crosshair cursor
    cross_img = np.zeros((24, 24, 4), dtype=np.uint8)
    cv2.line(cross_img, (12, 2), (12, 22), (255, 255, 255, 255), 2)
    cv2.line(cross_img, (2, 12), (22, 12), (255, 255, 255, 255), 2)
    cv2.line(cross_img, (12, 2), (12, 22), (0, 0, 0, 255), 1)
    cv2.line(cross_img, (2, 12), (22, 12), (0, 0, 0, 255), 1)
    cursors['crosshair'] = {'normal': cross_img, 'active': cross_img}
    
    return cursors

def load_cursors():
    """Load cursor images and convert to numpy arrays for faster processing"""
    cursors = {}
    
    if os.path.exists(cursors_dir):
        for cursor_file in os.listdir(cursors_dir):
            if cursor_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                cursor_name = os.path.splitext(cursor_file)[0]
                with suppress_stdout_stderr():
                    cursor_img = cv2.imread(os.path.join(cursors_dir, cursor_file), cv2.IMREAD_UNCHANGED)
                if cursor_img is None:
                    continue
                if cursor_img.shape[2] == 3:
                    cursor_img = cv2.cvtColor(cursor_img, cv2.COLOR_BGR2BGRA)
                
                # Create active version
                h, w = cursor_img.shape[:2]
                new_h, new_w = int(h * 1.1), int(w * 1.1)
                active_cursor = cv2.resize(cursor_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                cursors[cursor_name] = {
                    'normal': cursor_img,
                    'active': active_cursor
                }
    
    if not cursors:
        cursors = create_default_cursors()
    
    return cursors

def pil_to_cv2(pil_img):
    """Fast conversion from PIL to OpenCV format"""
    if pil_img.mode == 'RGBA':
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    elif pil_img.mode == 'RGB':
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        return np.array(pil_img)

def cv2_to_pil(cv2_img, mode='RGBA'):
    """Fast conversion from OpenCV to PIL format"""
    if len(cv2_img.shape) == 3:
        if cv2_img.shape[2] == 4:
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2_img
    return Image.fromarray(img)

def apply_perspective_transform_cv2(image, distortion_factor):
    """Apply perspective transform using OpenCV (faster)"""
    if distortion_factor == 0:
        return image
    
    h, w = image.shape[:2]
    dx = distortion_factor * w
    dy = distortion_factor * h
    
    # Source points
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Destination points with perspective
    dst = np.float32([
        [dx, dy],
        [w - dx, dy],
        [w - dx/2, h - dy/2],
        [dx/2, h - dy/2]
    ])
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Apply transform
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def add_shadow_effect_cv2(image):
    """Add shadow effect using OpenCV (faster)"""
    h, w = image.shape[:2]
    
    # Extract alpha channel for shadow shape
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
    else:
        alpha = np.ones((h, w), dtype=np.uint8) * 255
    
    # Shadow parameters
    offset_x = random.randint(-SHADOW_OFFSET_RANGE[1], SHADOW_OFFSET_RANGE[1])
    offset_y = random.randint(SHADOW_OFFSET_RANGE[0], SHADOW_OFFSET_RANGE[1])
    blur_radius = random.uniform(*SHADOW_BLUR_RANGE)
    opacity = random.uniform(*SHADOW_OPACITY_RANGE)
    
    # Create shadow
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow_alpha = cv2.GaussianBlur(alpha, (0, 0), blur_radius)
    
    # Shift shadow
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    shadow_alpha = cv2.warpAffine(shadow_alpha, M, (w, h))
    
    # Apply shadow
    shadow[:, :, 3] = (shadow_alpha * opacity).astype(np.uint8)
    
    # Composite
    result = image.copy()
    alpha_shadow = shadow[:, :, 3:4] / 255.0
    alpha_image = image[:, :, 3:4] / 255.0
    
    for c in range(3):
        result[:, :, c] = (1 - alpha_shadow[:, :, 0]) * shadow[:, :, c] + alpha_shadow[:, :, 0] * image[:, :, c]
    
    result[:, :, 3] = np.maximum(shadow[:, :, 3], image[:, :, 3])
    
    return result

def add_noise_cv2(image):
    """Add noise using OpenCV/NumPy (faster)"""
    noise_intensity = random.uniform(*NOISE_INTENSITY_RANGE)
    h, w = image.shape[:2]
    
    # Generate noise only for RGB channels
    noise = np.random.normal(0, noise_intensity * 255, (h, w, 3))
    
    if image.shape[2] == 4:
        # Preserve alpha channel
        noisy = image.copy()
        noisy[:, :, :3] = np.clip(image[:, :, :3] + noise, 0, 255)
    else:
        noisy = np.clip(image + noise, 0, 255)
    
    return noisy.astype(np.uint8)

def process_batch(batch_tasks):
    """Process a batch of tasks efficiently"""
    results = []
    
    for task_args in batch_tasks:
        patch_type, output_filename, output_dir, base_bg_pil, sprite_info, common_config = task_args
        
        # Convert to CV2 format for faster processing
        base_bg = pil_to_cv2(base_bg_pil)
        
        # Ensure BGRA format for alpha blending
        if base_bg.shape[2] == 3:
            base_bg = cv2.cvtColor(base_bg, cv2.COLOR_BGR2BGRA)
        
        # Unpack common_config
        patch_size = common_config['PATCH_SIZE']
        max_crop = common_config['MAX_CROP']
        max_shift = common_config['MAX_SHIFT']
        
        current_patch = base_bg.copy()
        
        if patch_type == "sprite":
            scaled_sprite_pil, new_w, new_h = sprite_info
            sprite_cv2 = pil_to_cv2(scaled_sprite_pil)
            
            # Scale variation
            if random.random() < SCALE_PROBABILITY:
                scale_factor = random.uniform(*SCALE_VARIATION_RANGE)
                new_w = int(new_w * scale_factor)
                new_h = int(new_h * scale_factor)
                sprite_cv2 = cv2.resize(sprite_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Rotation
            if random.random() < ROTATION_PROBABILITY:
                angle = random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
                center = (new_w // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new bounds
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                rotated_w = int(new_h * sin + new_w * cos)
                rotated_h = int(new_h * cos + new_w * sin)
                
                # Adjust rotation matrix
                M[0, 2] += (rotated_w / 2) - center[0]
                M[1, 2] += (rotated_h / 2) - center[1]
                
                sprite_cv2 = cv2.warpAffine(sprite_cv2, M, (rotated_w, rotated_h), 
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
                new_w, new_h = rotated_w, rotated_h
            
            # Shadow effect
            if random.random() < SHADOW_PROBABILITY:
                sprite_cv2 = add_shadow_effect_cv2(sprite_cv2)
            
            # FIXED: Handle sprites larger than patch size
            # If sprite is too large, resize it to fit within patch bounds
            max_sprite_size = int(patch_size * 0.9)  # Leave some margin
            if new_w > max_sprite_size or new_h > max_sprite_size:
                scale_factor = min(max_sprite_size / new_w, max_sprite_size / new_h)
                new_w = int(new_w * scale_factor)
                new_h = int(new_h * scale_factor)
                sprite_cv2 = cv2.resize(sprite_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Position calculation
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            x_pos = (patch_size - new_w) // 2 + dx
            y_pos = (patch_size - new_h) // 2 + dy
            
            # Ensure sprite fits within patch bounds
            x_pos = max(0, min(x_pos, patch_size - new_w))
            y_pos = max(0, min(y_pos, patch_size - new_h))
            
            # FIXED: Calculate actual paste dimensions to handle edge cases
            paste_w = min(new_w, patch_size - x_pos)
            paste_h = min(new_h, patch_size - y_pos)
            
            # Alpha blending with proper bounds checking
            if sprite_cv2.shape[2] == 4:
                # Ensure background has alpha channel
                if current_patch.shape[2] == 3:
                    current_patch = cv2.cvtColor(current_patch, cv2.COLOR_BGR2BGRA)
                
                # Extract the portion of sprite that will actually be pasted
                sprite_portion = sprite_cv2[:paste_h, :paste_w]
                alpha = sprite_portion[:, :, 3:4] / 255.0
                
                # Apply alpha blending only to the valid region
                for c in range(3):
                    current_patch[y_pos:y_pos+paste_h, x_pos:x_pos+paste_w, c] = \
                        (1 - alpha[:, :, 0]) * current_patch[y_pos:y_pos+paste_h, x_pos:x_pos+paste_w, c] + \
                        alpha[:, :, 0] * sprite_portion[:, :, c]
                        
                current_patch[y_pos:y_pos+paste_h, x_pos:x_pos+paste_w, 3] = \
                    np.maximum(current_patch[y_pos:y_pos+paste_h, x_pos:x_pos+paste_w, 3], 
                              sprite_portion[:, :, 3])
            else:
                # No alpha channel, just paste the portion that fits
                sprite_portion = sprite_cv2[:paste_h, :paste_w]
                current_patch[y_pos:y_pos+paste_h, x_pos:x_pos+paste_w] = sprite_portion
        
        # Random crop
        l = random.randint(0, max_crop)
        t = random.randint(0, max_crop)
        r = random.randint(0, max_crop)
        b = random.randint(0, max_crop)
        
        h, w = current_patch.shape[:2]
        current_patch = current_patch[t:h-b, l:w-r]
        
        # Resize to final size
        current_patch = cv2.resize(current_patch, (patch_size, patch_size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Apply augmentations
        if random.random() < PERSPECTIVE_PROBABILITY:
            distortion = random.uniform(-MAX_PERSPECTIVE_DISTORTION, MAX_PERSPECTIVE_DISTORTION)
            current_patch = apply_perspective_transform_cv2(current_patch, distortion)
        
        if random.random() < BLUR_PROBABILITY:
            blur_radius = random.uniform(*BLUR_RADIUS_RANGE)
            ksize = int(blur_radius * 4) | 1  # Ensure odd kernel size
            current_patch = cv2.GaussianBlur(current_patch, (ksize, ksize), blur_radius)
        
        if random.random() < NOISE_PROBABILITY:
            current_patch = add_noise_cv2(current_patch)
        
        # MISSING: Color jittering (brightness, contrast, saturation)
        if random.random() < COLOR_JITTER_PROBABILITY:
            # Convert to PIL for color adjustments (OpenCV color ops are more complex)
            temp_pil = cv2_to_pil(current_patch, mode='RGBA' if current_patch.shape[2] == 4 else 'RGB')
            
            # Brightness
            brightness_factor = random.uniform(*BRIGHTNESS_FACTOR_RANGE)
            enhancer = ImageEnhance.Brightness(temp_pil)
            temp_pil = enhancer.enhance(brightness_factor)
            
            # Contrast  
            contrast_factor = random.uniform(*CONTRAST_FACTOR_RANGE)
            enhancer = ImageEnhance.Contrast(temp_pil)
            temp_pil = enhancer.enhance(contrast_factor)
            
            # Saturation
            if temp_pil.mode in ('RGB', 'RGBA'):
                saturation_factor = random.uniform(*SATURATION_FACTOR_RANGE)
                enhancer = ImageEnhance.Color(temp_pil)
                temp_pil = enhancer.enhance(saturation_factor)
            
            current_patch = pil_to_cv2(temp_pil)
        
        # MISSING: Highlight effects
        if random.random() < HIGHLIGHT_PROBABILITY:
            h, w = current_patch.shape[:2]
            
            # Create highlight
            highlight_size = random.uniform(*HIGHLIGHT_SIZE_FACTOR_RANGE)
            highlight_radius = int(min(h, w) * highlight_size)
            highlight_intensity = random.uniform(*HIGHLIGHT_INTENSITY_RANGE)
            
            # Random position
            highlight_x = random.randint(highlight_radius, w - highlight_radius)
            highlight_y = random.randint(highlight_radius, h - highlight_radius)
            
            # Create circular highlight
            highlight_overlay = np.zeros((h, w), dtype=np.float32)
            cv2.circle(highlight_overlay, (highlight_x, highlight_y), highlight_radius, 1.0, -1)
            
            # Apply Gaussian blur for soft highlight
            highlight_overlay = cv2.GaussianBlur(highlight_overlay, (0, 0), highlight_radius * 0.3)
            
            # Apply highlight to RGB channels
            for c in range(3):
                current_patch[:, :, c] = np.clip(
                    current_patch[:, :, c] + highlight_overlay * highlight_intensity * 255,
                    0, 255
                ).astype(np.uint8)
        
        # MISSING: Occlusion effects  
        if random.random() < OCCLUSION_PROBABILITY:
            h, w = current_patch.shape[:2]
            num_occlusions = random.randint(1, MAX_OCCLUSIONS_PER_IMAGE)
            
            for _ in range(num_occlusions):
                # Random occlusion size
                size_factor = random.uniform(MIN_OCCLUSION_SIZE_FACTOR, MAX_OCCLUSION_SIZE_FACTOR)
                occlusion_size = int(min(h, w) * size_factor)
                
                # Random position
                occ_x = random.randint(0, max(0, w - occlusion_size))
                occ_y = random.randint(0, max(0, h - occlusion_size))
                
                # Random occlusion shape (rectangle or circle)
                if random.random() < 0.5:
                    # Rectangular occlusion
                    occ_w = random.randint(occlusion_size // 2, occlusion_size)
                    occ_h = random.randint(occlusion_size // 2, occlusion_size)
                    
                    # Random color
                    color = [random.randint(*OCCLUSION_COLOR_RANGES[c]) for c in range(3)]
                    alpha = random.randint(*OCCLUSION_ALPHA_RANGE)
                    
                    # Create occlusion overlay
                    overlay = np.zeros((h, w, 4), dtype=np.uint8)
                    overlay[occ_y:occ_y+occ_h, occ_x:occ_x+occ_w] = color + [alpha]
                    
                else:
                    # Circular occlusion
                    radius = occlusion_size // 2
                    center_x = occ_x + radius
                    center_y = occ_y + radius
                    
                    # Random color
                    color = [random.randint(*OCCLUSION_COLOR_RANGES[c]) for c in range(3)]
                    alpha = random.randint(*OCCLUSION_ALPHA_RANGE)
                    
                    # Create occlusion overlay
                    overlay = np.zeros((h, w, 4), dtype=np.uint8)
                    cv2.circle(overlay, (center_x, center_y), radius, color + [alpha], -1)
                
                # Apply occlusion with alpha blending
                if current_patch.shape[2] == 3:
                    current_patch = cv2.cvtColor(current_patch, cv2.COLOR_BGR2BGRA)
                
                overlay_alpha = overlay[:, :, 3:4] / 255.0
                for c in range(3):
                    current_patch[:, :, c] = (
                        (1 - overlay_alpha[:, :, 0]) * current_patch[:, :, c] + 
                        overlay_alpha[:, :, 0] * overlay[:, :, c]
                    ).astype(np.uint8)
        
        # MISSING CURSOR APPLICATION - Add cursor overlay
        if random.random() < CURSOR_PROBABILITY and _cursor_cache:
            cursor_name = random.choice(list(_cursor_cache.keys()))
            cursor_data = _cursor_cache[cursor_name]
            
            # Choose normal or active cursor
            if random.random() < CURSOR_ACTIVE_PROBABILITY:
                cursor_img = cursor_data['active'].copy()
            else:
                cursor_img = cursor_data['normal'].copy()
            
            # Scale cursor to appropriate size for patch
            cursor_h, cursor_w = cursor_img.shape[:2]
            
            # Target cursor size should be small relative to patch (5-15% of patch size)
            target_size = random.randint(int(patch_size * 0.1), int(patch_size * 0.15))  # 12-37 pixels for 244px patch
            
            # Scale based on larger dimension to maintain aspect ratio
            scale_factor = target_size / max(cursor_w, cursor_h)
            new_cursor_w = int(cursor_w * scale_factor)
            new_cursor_h = int(cursor_h * scale_factor)
            cursor_img = cv2.resize(cursor_img, (new_cursor_w, new_cursor_h), interpolation=cv2.INTER_LINEAR)
            
            # Position cursor (can be on edges or center)
            if random.random() < CURSOR_EDGE_PROBABILITY:
                # Place on edges
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    cursor_x = random.randint(0, max(0, patch_size - new_cursor_w))
                    cursor_y = random.randint(0, patch_size // 4)
                elif edge == 'bottom':
                    cursor_x = random.randint(0, max(0, patch_size - new_cursor_w))
                    cursor_y = random.randint(3 * patch_size // 4, max(3 * patch_size // 4, patch_size - new_cursor_h))
                elif edge == 'left':
                    cursor_x = random.randint(0, patch_size // 4)
                    cursor_y = random.randint(0, max(0, patch_size - new_cursor_h))
                else:  # right
                    cursor_x = random.randint(3 * patch_size // 4, max(3 * patch_size // 4, patch_size - new_cursor_w))
                    cursor_y = random.randint(0, max(0, patch_size - new_cursor_h))
            else:
                # Place anywhere
                cursor_x = random.randint(0, max(0, patch_size - new_cursor_w))
                cursor_y = random.randint(0, max(0, patch_size - new_cursor_h))
            
            # Ensure cursor fits within bounds
            cursor_x = max(0, min(cursor_x, patch_size - new_cursor_w))
            cursor_y = max(0, min(cursor_y, patch_size - new_cursor_h))
            
            # Calculate actual overlay dimensions
            overlay_w = min(new_cursor_w, patch_size - cursor_x)
            overlay_h = min(new_cursor_h, patch_size - cursor_y)
            
            # Apply cursor with alpha blending
            if cursor_img.shape[2] == 4:  # Has alpha channel
                # Ensure current_patch has alpha channel
                if current_patch.shape[2] == 3:
                    current_patch = cv2.cvtColor(current_patch, cv2.COLOR_BGR2BGRA)
                
                # Extract cursor portion that will be overlaid
                cursor_portion = cursor_img[:overlay_h, :overlay_w]
                cursor_alpha = cursor_portion[:, :, 3:4] / 255.0
                
                # Alpha blend cursor
                for c in range(3):
                    current_patch[cursor_y:cursor_y+overlay_h, cursor_x:cursor_x+overlay_w, c] = \
                        (1 - cursor_alpha[:, :, 0]) * current_patch[cursor_y:cursor_y+overlay_h, cursor_x:cursor_x+overlay_w, c] + \
                        cursor_alpha[:, :, 0] * cursor_portion[:, :, c]
                
                # Update alpha channel
                current_patch[cursor_y:cursor_y+overlay_h, cursor_x:cursor_x+overlay_w, 3] = \
                    np.maximum(current_patch[cursor_y:cursor_y+overlay_h, cursor_x:cursor_x+overlay_w, 3], 
                              cursor_portion[:, :, 3])
            else:
                # No alpha channel, direct overlay
                cursor_portion = cursor_img[:overlay_h, :overlay_w]
                current_patch[cursor_y:cursor_y+overlay_h, cursor_x:cursor_x+overlay_w] = cursor_portion
        
        # Convert to RGB for saving
        if current_patch.shape[2] == 4:
            current_patch = cv2.cvtColor(current_patch, cv2.COLOR_BGRA2BGR)
        
        # Save directly with OpenCV (faster than PIL)
        full_output_path = os.path.join(output_dir, output_filename)
        
        # JPEG compression
        if random.random() < JPEG_COMPRESSION_PROBABILITY:
            quality = random.randint(*COMPRESSION_QUALITY_RANGE)
            cv2.imwrite(full_output_path, current_patch, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(full_output_path, current_patch)
        
        results.append(True)
    
    return results

def generate_patch_worker_optimized(batch_tasks):
    """Optimized worker that processes batches"""
    return process_batch(batch_tasks)

# Main execution
if __name__ == "__main__":
    print("🎯 Optimized Chess Piece Training Data Generator")
    print("=" * 50)
    
    # Load cursors
    print("📱 Loading cursors...")
    cursors = load_cursors()
    print(f"   Loaded {len(cursors)} cursor types")
    
    # Load backgrounds efficiently
    print("🏞️ Loading backgrounds...")
    backgrounds = []
    backgrounds_dot = []
    orig_patch = None
    
    # Use OpenCV for faster image loading
    for bfile in os.listdir(boards_dir):
        board_path = os.path.join(boards_dir, bfile)
        with suppress_stdout_stderr():
            board = cv2.imread(board_path, cv2.IMREAD_UNCHANGED)
        if board is None:
            continue
        if board.shape[2] == 3:
            board = cv2.cvtColor(board, cv2.COLOR_BGR2BGRA)
        
        sz = board.shape[1] // 8
        if orig_patch is None:
            orig_patch = sz
        
        for rank in range(8):
            for file in range(8):
                p = board[rank*sz:(rank+1)*sz, file*sz:(file+1)*sz]
                p_resized = cv2.resize(p, (PATCH_SIZE, PATCH_SIZE), 
                                     interpolation=cv2.INTER_LINEAR)
                backgrounds.append(cv2_to_pil(p_resized))
    
    for bfile in os.listdir(boards_dot_dir):
        board_path = os.path.join(boards_dot_dir, bfile)
        with suppress_stdout_stderr():
            board = cv2.imread(board_path, cv2.IMREAD_UNCHANGED)
        if board is None:
            continue
        if board.shape[2] == 3:
            board = cv2.cvtColor(board, cv2.COLOR_BGR2BGRA)
        
        sz = board.shape[1] // 8
        for rank in range(8):
            for file in range(8):
                p = board[rank*sz:(rank+1)*sz, file*sz:(file+1)*sz]
                p_resized = cv2.resize(p, (PATCH_SIZE, PATCH_SIZE), 
                                     interpolation=cv2.INTER_LINEAR)
                backgrounds_dot.append(cv2_to_pil(p_resized))
    
    scale = PATCH_SIZE / orig_patch
    
    # Gather sprite entries
    sprite_entries = []
    for style_folder in os.listdir(sprites_dir):
        style_path = os.path.join(sprites_dir, style_folder)
        if not os.path.isdir(style_path):
            continue
        for sprite_file in os.listdir(style_path):
            if sprite_file.lower().endswith(('.png','.jpg','.jpeg')):
                sprite_entries.append((style_folder, sprite_file))
    
    # Calculate totals
    total_empty = N_VARIANTS * 5
    total_empty_dot = N_VARIANTS * 5
    total_sprites = len(sprite_entries) * N_VARIANTS
    total_tasks = total_empty + total_empty_dot + total_sprites
    
    # Prepare common configuration
    common_config = {
        'PATCH_SIZE': PATCH_SIZE,
        'MAX_CROP': MAX_CROP,
        'MAX_SHIFT': MAX_SHIFT,
        'cursors': cursors
    }
    
    # Prepare all tasks
    all_tasks = []
    
    # Empty patches
    empty_dir = os.path.join(out_dir, "empty")
    empty_dot_dir = os.path.join(out_dir, "empty_dot")
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(empty_dot_dir, exist_ok=True)
    
    print(f"📝 Preparing {total_empty} empty patch tasks...")
    for i in range(total_empty):
        bg = random.choice(backgrounds)
        output_filename = f"empty_{i}.jpg"
        all_tasks.append(
            ("empty", output_filename, empty_dir, bg, None, common_config)
        )
    
    print(f"📝 Preparing {total_empty_dot} empty_dot patch tasks...")
    for i in range(total_empty_dot):
        bg = random.choice(backgrounds_dot)
        output_filename = f"empty_dot_{i}.jpg"
        all_tasks.append(
            ("empty_dot", output_filename, empty_dot_dir, bg, None, common_config)
        )
    
    print(f"🎪 Preparing {total_sprites} sprite patch tasks...")
    
    # Pre-load and resize sprites for better efficiency
    sprite_cache = {}
    for style_folder, sprite_file in sprite_entries:
        style_path = os.path.join(sprites_dir, style_folder)
        sprite_path = os.path.join(style_path, sprite_file)
        
        # Load with OpenCV for consistency
        with suppress_stdout_stderr():
            sprite_cv2 = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if sprite_cv2 is None:
            continue
        if sprite_cv2.shape[2] == 3:
            sprite_cv2 = cv2.cvtColor(sprite_cv2, cv2.COLOR_BGR2BGRA)
        
        # Convert to PIL for compatibility
        sprite_pil = cv2_to_pil(sprite_cv2)
        
        new_w = int(sprite_pil.width * scale)
        new_h = int(sprite_pil.height * scale)
        sprite_resized = sprite_pil.resize((new_w, new_h), Image.LANCZOS)
        
        key = (style_folder, sprite_file)
        sprite_cache[key] = (sprite_resized, new_w, new_h)
        
        cls = os.path.splitext(sprite_file)[0]
        save_dir = os.path.join(out_dir, cls)
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(N_VARIANTS):
            bg = random.choice(backgrounds)
            output_filename = f"{cls}_{style_folder}_{i}.jpg"
            all_tasks.append(
                ("sprite", output_filename, save_dir, bg, sprite_cache[key], common_config)
            )
    
    # Shuffle tasks for better load distribution
    random.shuffle(all_tasks)
    
    # Create batches
    batches = [all_tasks[i:i + CHUNK_SIZE] for i in range(0, len(all_tasks), CHUNK_SIZE)]
    
    print(f"\n🚀 Starting optimized patch generation...")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Total patches: {len(all_tasks):,}")
    print(f"   Batch size: {CHUNK_SIZE}")
    print(f"   Total batches: {len(batches)}")
    print("-" * 50)
    
    # Process with progress bar
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker, 
                           initargs=(cursors,)) as executor:
        futures = [executor.submit(generate_patch_worker_optimized, batch) for batch in batches]
        
        with tqdm(total=len(all_tasks), desc="🎨 Generating Optimized Patches") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(len(result))
    
    print("\n✅ All patches generated successfully with optimizations!")
    print(f"🚀 Performance improvements applied:")
    print(f"   - OpenCV for faster image operations")
    print(f"   - Batch processing to reduce overhead")
    print(f"   - Pre-cached sprite loading")
    print(f"   - Optimized memory usage")
    print(f"   - Direct JPEG writing with OpenCV")