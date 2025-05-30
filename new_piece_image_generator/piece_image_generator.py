#!/usr/bin/env python3
"""
Updated Chess Piece Image Generator - CPU-based, PyTorch Compatible
Fixed size mismatch and improved compatibility with PyTorch training

Key Changes from Original:
- Fixed PATCH_SIZE: 244 → 224 (standard for modern CNNs)
- Improved class naming consistency  
- Better augmentation parameters
- Optimized for PyTorch training pipeline
- Maintains CPU-based multiprocessing as requested
"""

from PIL import Image, ImageDraw, ImageEnhance
import os, random
import multiprocessing
from tqdm import tqdm
import json
from pathlib import Path

# ─── UPDATED CONFIG ──────────────────────────────────────────────────────────
boards_dir   = "boards"
boards_dot_dir = "boards_with_dot"
sprites_dir  = "pieces"
out_dir      = "data/train"

# Fixed: Changed from 244 to 224 to match modern CNN standards and PyTorch training
PATCH_SIZE   = 224      # Standard size for EfficientNet/ResNet
N_VARIANTS   = 1     # Keep your original setting
MAX_SHIFT    = 12       # Slightly increased for more variation
MAX_CROP     = 10       # Slightly increased

# Enhanced augmentation config for better model robustness
OCCLUSION_PROBABILITY = 0.5       # Increased for better occlusion handling
MAX_OCCLUSIONS_PER_IMAGE = 3      # Increased max occlusions
MIN_OCCLUSION_SIZE_FACTOR = 0.02  # Smaller min size
MAX_OCCLUSION_SIZE_FACTOR = 0.25  # Larger max size
OCCLUSION_COLOR_RANGES = ((0, 255), (0, 255), (0, 255))
OCCLUSION_ALPHA_RANGE = (40, 200) # Wider alpha range

# Enhanced color jitter for better generalization
COLOR_JITTER_PROBABILITY = 0.7    # Increased probability
BRIGHTNESS_FACTOR_RANGE = (0.5, 1.5)  # Wider range
CONTRAST_FACTOR_RANGE = (0.5, 1.5)    # Wider range  
SATURATION_FACTOR_RANGE = (0.4, 1.6)  # Wider range

# Additional augmentations
NOISE_PROBABILITY = 0.3           # Add subtle noise
SHADOW_PROBABILITY = 0.4          # Add shadow effects

NUM_WORKERS = os.cpu_count()      # Keep CPU-based as requested
# ─────────────────────────────────────────────────────────────────────────────

print(f"🎯 Chess Piece Generator - Updated for PyTorch")
print(f"   Patch Size: {PATCH_SIZE}px (fixed from 244px)")
print(f"   Variants per sprite: {N_VARIANTS}")
print(f"   CPU Workers: {NUM_WORKERS}")

# 1) Build & resize background patches
backgrounds = []
backgrounds_dot = []
orig_patch = None

# Load regular backgrounds
if os.path.exists(boards_dir):
    for bfile in os.listdir(boards_dir):
        if bfile.lower().endswith(('.png', '.jpg', '.jpeg')):
            board = Image.open(os.path.join(boards_dir, bfile)).convert("RGBA")
            sz = board.width // 8
            if orig_patch is None:
                orig_patch = sz
            for rank in range(8):
                for file in range(8):
                    p = board.crop((file*sz, rank*sz, (file+1)*sz, (rank+1)*sz))
                    backgrounds.append(p.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS))

# Load dot backgrounds
if os.path.exists(boards_dot_dir):
    for bfile in os.listdir(boards_dot_dir):
        if bfile.lower().endswith(('.png', '.jpg', '.jpeg')):
            board = Image.open(os.path.join(boards_dot_dir, bfile)).convert("RGBA")
            sz = board.width // 8
            if orig_patch is None:
                orig_patch = sz
            for rank in range(8):
                for file in range(8):
                    p = board.crop((file*sz, rank*sz, (file+1)*sz, (rank+1)*sz))
                    backgrounds_dot.append(p.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS))

scale = PATCH_SIZE / orig_patch if orig_patch else 1.0

print(f"📋 Loaded {len(backgrounds)} regular backgrounds")
print(f"📋 Loaded {len(backgrounds_dot)} dot backgrounds")

def add_noise(image, intensity=0.02):
    """Add subtle noise to make model more robust"""
    import numpy as np
    
    img_array = np.array(image)
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array, mode=image.mode)

def add_shadow_effect(image, strength=0.3):
    """Add subtle shadow/lighting gradient"""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Random shadow direction
    direction = random.choice(['top', 'bottom', 'left', 'right'])
    alpha_strength = int(255 * strength * random.uniform(0.5, 1.0))
    
    if direction == 'top':
        for y in range(image.height // 3):
            alpha = int(alpha_strength * (1 - y / (image.height // 3)))
            draw.rectangle([0, y, image.width, y + 1], fill=(0, 0, 0, alpha))
    elif direction == 'bottom':
        start_y = 2 * image.height // 3
        for y in range(start_y, image.height):
            alpha = int(alpha_strength * (y - start_y) / (image.height // 3))
            draw.rectangle([0, y, image.width, y + 1], fill=(0, 0, 0, alpha))
    
    return Image.alpha_composite(image, overlay)

def apply_augmentations(image_patch):
    """Enhanced augmentations for better model robustness"""
    output_patch = image_patch.copy()
    if output_patch.mode != 'RGBA':
        output_patch = output_patch.convert('RGBA')

    # Color Jitter (enhanced)
    if random.random() < COLOR_JITTER_PROBABILITY:
        enhancer = ImageEnhance.Brightness(output_patch)
        output_patch = enhancer.enhance(random.uniform(*BRIGHTNESS_FACTOR_RANGE))
        enhancer = ImageEnhance.Contrast(output_patch)
        output_patch = enhancer.enhance(random.uniform(*CONTRAST_FACTOR_RANGE))
        enhancer = ImageEnhance.Color(output_patch)
        output_patch = enhancer.enhance(random.uniform(*SATURATION_FACTOR_RANGE))

    # Enhanced Occlusions
    if random.random() < OCCLUSION_PROBABILITY:
        occlusion_overlay = Image.new("RGBA", output_patch.size, (0,0,0,0))
        draw = ImageDraw.Draw(occlusion_overlay)
        num_occlusions_to_add = random.randint(1, MAX_OCCLUSIONS_PER_IMAGE)

        for _ in range(num_occlusions_to_add):
            # More shape variety
            shape_type = random.choice(["rectangle", "ellipse", "triangle"])
            
            occ_w = random.randint(int(MIN_OCCLUSION_SIZE_FACTOR * output_patch.width), 
                                   int(MAX_OCCLUSION_SIZE_FACTOR * output_patch.width))
            occ_h = random.randint(int(MIN_OCCLUSION_SIZE_FACTOR * output_patch.height), 
                                   int(MAX_OCCLUSION_SIZE_FACTOR * output_patch.height))
            
            pos_x = random.randint(0, max(0, output_patch.width - occ_w))
            pos_y = random.randint(0, max(0, output_patch.height - occ_h))
            
            r = random.randint(*OCCLUSION_COLOR_RANGES[0])
            g = random.randint(*OCCLUSION_COLOR_RANGES[1])
            b = random.randint(*OCCLUSION_COLOR_RANGES[2])
            a = random.randint(*OCCLUSION_ALPHA_RANGE)
            fill_color = (r, g, b, a)
            
            if shape_type == "triangle":
                # Draw triangle
                points = [
                    (pos_x, pos_y + occ_h),
                    (pos_x + occ_w // 2, pos_y),
                    (pos_x + occ_w, pos_y + occ_h)
                ]
                draw.polygon(points, fill=fill_color)
            else:
                getattr(draw, shape_type)([pos_x, pos_y, pos_x + occ_w, pos_y + occ_h], fill=fill_color)
        
        output_patch = Image.alpha_composite(output_patch, occlusion_overlay)
    
    # Add noise occasionally
    if random.random() < NOISE_PROBABILITY:
        output_patch = add_noise(output_patch)
    
    # Add shadow effects occasionally  
    if random.random() < SHADOW_PROBABILITY:
        output_patch = add_shadow_effect(output_patch)
    
    return output_patch

def generate_patch_worker(task_args):
    """Worker function to generate and save a single image patch"""
    patch_type, output_filename, output_dir, base_bg_pil, sprite_info, common_config = task_args

    try:
        # Unpack common_config
        patch_size = common_config['PATCH_SIZE']
        max_crop = common_config['MAX_CROP']
        max_shift = common_config['MAX_SHIFT']

        # Start with a copy of the base background
        current_patch_state = base_bg_pil.copy()

        if patch_type == "sprite":
            scaled_sprite_pil, new_w, new_h = sprite_info
            # Ensure current_patch_state is RGBA if sprite has alpha
            if scaled_sprite_pil.mode == 'RGBA' and current_patch_state.mode != 'RGBA':
                current_patch_state = current_patch_state.convert('RGBA')
            
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            x_pos = (patch_size - new_w) // 2 + dx
            y_pos = (patch_size - new_h) // 2 + dy
            current_patch_state.paste(scaled_sprite_pil, (x_pos, y_pos), 
                                    scaled_sprite_pil if scaled_sprite_pil.mode == 'RGBA' else None)

        # Random edge-crop
        l = random.randint(0, max_crop)
        t = random.randint(0, max_crop)
        r = random.randint(0, max_crop)
        b = random.randint(0, max_crop)
        w_orig, h_orig = current_patch_state.size
        
        # Ensure crop box is valid
        crop_box = (l, t, max(l + 1, w_orig - r), max(t + 1, h_orig - b))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            cropped_intermediate = current_patch_state
        else:
            cropped_intermediate = current_patch_state.crop(crop_box)

        raw_final_patch = cropped_intermediate.resize((patch_size, patch_size), Image.LANCZOS)
        augmented_patch = apply_augmentations(raw_final_patch)

        # Save with high quality JPEG
        full_output_path = os.path.join(output_dir, output_filename)
        augmented_patch.convert("RGB").save(full_output_path, "JPEG", quality=95, optimize=True)
        return True
        
    except Exception as e:
        print(f"Error generating {output_filename}: {e}")
        return False

# 2) Gather all sprite entries with better error handling
sprite_entries = []
if os.path.exists(sprites_dir):
    for style_folder in os.listdir(sprites_dir):
        style_path = os.path.join(sprites_dir, style_folder)
        if not os.path.isdir(style_path):
            continue
        for sprite_file in os.listdir(style_path):
            if sprite_file.lower().endswith(('.png','.jpg','.jpeg')):
                sprite_entries.append((style_folder, sprite_file))

print(f"🎭 Found {len(sprite_entries)} sprite images")

# 3) Compute total tasks
total_empty = N_VARIANTS * 3      # Reduced multiplier for balance
total_empty_dot = N_VARIANTS * 3 if backgrounds_dot else 0
total_sprites = len(sprite_entries) * N_VARIANTS
total_tasks = total_empty + total_empty_dot + total_sprites

print(f"📊 Total images to generate: {total_tasks:,}")

if __name__ == "__main__":
    # Prepare common configuration to pass to workers
    common_config = {
        'PATCH_SIZE': PATCH_SIZE,
        'MAX_CROP': MAX_CROP,
        'MAX_SHIFT': MAX_SHIFT,
    }

    # 4) Prepare tasks for multiprocessing
    all_tasks_to_process = []

    # Create output directory structure
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 4a) Empty patches
    empty_dir = os.path.join(out_dir, "empty")
    os.makedirs(empty_dir, exist_ok=True)

    print(f"📝 Preparing {total_empty:,} empty patch tasks...")
    for i in range(total_empty):
        bg = random.choice(backgrounds) if backgrounds else None
        if bg:
            output_filename = f"empty_{i:06d}.jpg"  # Zero-padded for better sorting
            all_tasks_to_process.append(
                ("empty", output_filename, empty_dir, bg, None, common_config)
            )

    # 4b) Empty dot patches  
    if backgrounds_dot:
        empty_dot_dir = os.path.join(out_dir, "empty_dot")
        os.makedirs(empty_dot_dir, exist_ok=True)

        print(f"📝 Preparing {total_empty_dot:,} empty_dot patch tasks...")
        for i in range(total_empty_dot):
            bg = random.choice(backgrounds_dot)
            output_filename = f"empty_dot_{i:06d}.jpg"
            all_tasks_to_process.append(
                ("empty_dot", output_filename, empty_dot_dir, bg, None, common_config)
            )

    # 4c) Sprite patches
    print(f"🎭 Preparing {total_sprites:,} sprite patch tasks...")
    for style_folder, sprite_file in sprite_entries:
        try:
            style_path = os.path.join(sprites_dir, style_folder)
            sprite_original = Image.open(os.path.join(style_path, sprite_file)).convert("RGBA")
            
            new_w = int(sprite_original.width * scale)
            new_h = int(sprite_original.height * scale)
            sprite_resized_pil = sprite_original.resize((new_w, new_h), Image.LANCZOS)
            sprite_info_tuple = (sprite_resized_pil, new_w, new_h)

            # Use consistent class naming (matching class_mapping.txt format)
            cls = os.path.splitext(sprite_file)[0]
            save_dir_for_cls = os.path.join(out_dir, cls)
            os.makedirs(save_dir_for_cls, exist_ok=True)

            for i in range(N_VARIANTS):
                bg = random.choice(backgrounds) if backgrounds else None
                if bg:
                    output_filename = f"{cls}_{style_folder}_{i:06d}.jpg"
                    all_tasks_to_process.append(
                        ("sprite", output_filename, save_dir_for_cls, bg, sprite_info_tuple, common_config)
                    )
        except Exception as e:
            print(f"⚠️  Error loading sprite {sprite_file}: {e}")
            continue
    
    print(f"\n🚀 Starting generation with {NUM_WORKERS} CPU workers...")
    print(f"📊 Total tasks: {len(all_tasks_to_process):,}")
    
    # 5) Run generation with multiprocessing
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(generate_patch_worker, all_tasks_to_process), 
            total=len(all_tasks_to_process),
            desc="Generating patches"
        ))

    successful = sum(results)
    print(f"\n✅ Successfully generated {successful:,}/{len(all_tasks_to_process):,} images")
    
    # Save generation metadata for PyTorch training
    metadata = {
        'patch_size': PATCH_SIZE,
        'total_images': successful,
        'n_variants_per_sprite': N_VARIANTS,
        'sprite_count': len(sprite_entries),
        'background_count': len(backgrounds) + len(backgrounds_dot),
        'augmentation_config': {
            'occlusion_probability': OCCLUSION_PROBABILITY,
            'color_jitter_probability': COLOR_JITTER_PROBABILITY,
            'noise_probability': NOISE_PROBABILITY,
            'shadow_probability': SHADOW_PROBABILITY
        },
        'classes_found': list(set(os.path.splitext(sprite_file)[0] 
                                for _, sprite_file in sprite_entries)) + ['empty'] + (['empty_dot'] if backgrounds_dot else [])
    }
    
    with open(os.path.join(out_dir, 'generation_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Generation metadata saved to {os.path.join(out_dir, 'generation_metadata.json')}")
    print("🎯 Data generation complete! Ready for PyTorch training.")