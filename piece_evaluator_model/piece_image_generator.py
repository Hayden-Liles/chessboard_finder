from PIL import Image, ImageDraw, ImageEnhance
import os, random
import multiprocessing
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
boards_dir   = "boards"
boards_dot_dir = "boards_with_dot"
sprites_dir  = "pieces"
out_dir      = "data2/train"

PATCH_SIZE   = 244      # px
N_VARIANTS   = 5000     # per sprite
MAX_SHIFT    = 10       # max px to shift sprite
MAX_CROP     = 8        # max px to crop from edges
# --- Augmentation Config ---
OCCLUSION_PROBABILITY = 0.4       # Probability of adding occlusions to an image
MAX_OCCLUSIONS_PER_IMAGE = 2      # Max number of occluding shapes per image
MIN_OCCLUSION_SIZE_FACTOR = 0.05  # Min size of occlusion relative to patch dimension
MAX_OCCLUSION_SIZE_FACTOR = 0.20  # Max size of occlusion relative to patch dimension
OCCLUSION_COLOR_RANGES = ((0, 255), (0, 255), (0, 255)) # Min/Max for R, G, B of occlusion
OCCLUSION_ALPHA_RANGE = (70, 180) # Min/Max for alpha (0-255) of occlusion

COLOR_JITTER_PROBABILITY = 0.6    # Probability of applying color jitter
BRIGHTNESS_FACTOR_RANGE = (0.6, 1.4)
CONTRAST_FACTOR_RANGE = (0.6, 1.4)
SATURATION_FACTOR_RANGE = (0.6, 1.4)
NUM_WORKERS = os.cpu_count()  # Number of CPU cores to use, os.cpu_count() for all available
# ──────────────────────────────────────────────────────────────────────────────

# 1) Build & resize background patches
backgrounds = []
backgrounds_dot = []
orig_patch = None

for bfile in os.listdir(boards_dir):
    board = Image.open(os.path.join(boards_dir, bfile)).convert("RGBA")
    sz = board.width // 8
    if orig_patch is None:
        orig_patch = sz
    for rank in range(8):
        for file in range(8):
            p = board.crop((file*sz, rank*sz, (file+1)*sz, (rank+1)*sz))
            backgrounds.append(p.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS))

for bfile in os.listdir(boards_dot_dir):
    board = Image.open(os.path.join(boards_dot_dir, bfile)).convert("RGBA")
    sz = board.width // 8
    if orig_patch is None:
        orig_patch = sz
    for rank in range(8):
        for file in range(8):
            p = board.crop((file*sz, rank*sz, (file+1)*sz, (rank+1)*sz))
            backgrounds_dot.append(p.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS))

scale = PATCH_SIZE / orig_patch

def apply_augmentations(image_patch):
    """Applies configured augmentations to the image patch."""
    output_patch = image_patch.copy()
    if output_patch.mode != 'RGBA':
        output_patch = output_patch.convert('RGBA')

    # Color Jitter
    if random.random() < COLOR_JITTER_PROBABILITY:
        enhancer = ImageEnhance.Brightness(output_patch)
        output_patch = enhancer.enhance(random.uniform(BRIGHTNESS_FACTOR_RANGE[0], BRIGHTNESS_FACTOR_RANGE[1]))
        enhancer = ImageEnhance.Contrast(output_patch)
        output_patch = enhancer.enhance(random.uniform(CONTRAST_FACTOR_RANGE[0], CONTRAST_FACTOR_RANGE[1]))
        enhancer = ImageEnhance.Color(output_patch) # Adjusts saturation
        output_patch = enhancer.enhance(random.uniform(SATURATION_FACTOR_RANGE[0], SATURATION_FACTOR_RANGE[1]))

    # Occlusions
    if random.random() < OCCLUSION_PROBABILITY:
        occlusion_overlay = Image.new("RGBA", output_patch.size, (0,0,0,0)) # Transparent overlay
        draw = ImageDraw.Draw(occlusion_overlay)
        num_occlusions_to_add = random.randint(1, MAX_OCCLUSIONS_PER_IMAGE)

        for _ in range(num_occlusions_to_add):
            shape_type = random.choice(["rectangle", "ellipse"])
            
            occ_w = random.randint(int(MIN_OCCLUSION_SIZE_FACTOR * output_patch.width), 
                                   int(MAX_OCCLUSION_SIZE_FACTOR * output_patch.width))
            occ_h = random.randint(int(MIN_OCCLUSION_SIZE_FACTOR * output_patch.height), 
                                   int(MAX_OCCLUSION_SIZE_FACTOR * output_patch.height))
            
            pos_x = random.randint(0, max(0, output_patch.width - occ_w))
            pos_y = random.randint(0, max(0, output_patch.height - occ_h))
            
            r = random.randint(OCCLUSION_COLOR_RANGES[0][0], OCCLUSION_COLOR_RANGES[0][1])
            g = random.randint(OCCLUSION_COLOR_RANGES[1][0], OCCLUSION_COLOR_RANGES[1][1])
            b = random.randint(OCCLUSION_COLOR_RANGES[2][0], OCCLUSION_COLOR_RANGES[2][1])
            a = random.randint(OCCLUSION_ALPHA_RANGE[0], OCCLUSION_ALPHA_RANGE[1])
            fill_color = (r, g, b, a)
            
            getattr(draw, shape_type)([pos_x, pos_y, pos_x + occ_w, pos_y + occ_h], fill=fill_color)
        output_patch = Image.alpha_composite(output_patch, occlusion_overlay)
    return output_patch


def generate_patch_worker(task_args):
    """
    Worker function to generate and save a single image patch.
    """
    patch_type, output_filename, output_dir, base_bg_pil, sprite_info, common_config = task_args

    # Unpack common_config
    patch_size = common_config['PATCH_SIZE']
    max_crop = common_config['MAX_CROP']
    max_shift = common_config['MAX_SHIFT']

    # Start with a copy of the base background (which is already PATCH_SIZExPATCH_SIZE)
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
        current_patch_state.paste(scaled_sprite_pil, (x_pos, y_pos), scaled_sprite_pil if scaled_sprite_pil.mode == 'RGBA' else None)

    # Random edge-crop on the current_patch_state
    l = random.randint(0, max_crop)
    t = random.randint(0, max_crop)
    r = random.randint(0, max_crop)
    b = random.randint(0, max_crop)
    w_orig, h_orig = current_patch_state.size # Should be patch_size
    
    # Ensure crop box is valid
    crop_box = (l, t, max(l + 1, w_orig - r), max(t + 1, h_orig - b))
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]: # if crop is too aggressive
        cropped_intermediate = current_patch_state # skip crop
    else:
        cropped_intermediate = current_patch_state.crop(crop_box)

    raw_final_patch = cropped_intermediate.resize((patch_size, patch_size), Image.LANCZOS)
    augmented_patch = apply_augmentations(raw_final_patch)

    full_output_path = os.path.join(output_dir, output_filename)
    augmented_patch.convert("RGB").save(full_output_path)
    return True

# 2) Gather all sprite entries
sprite_entries = []
for style_folder in os.listdir(sprites_dir):
    style_path = os.path.join(sprites_dir, style_folder)
    if not os.path.isdir(style_path):
        continue
    for sprite_file in os.listdir(style_path):
        if sprite_file.lower().endswith(('.png','.jpg','.jpeg')):
            sprite_entries.append((style_folder, sprite_file))

# 3) Compute total tasks (empty + empty_dot + all sprite composites)
total_empty = N_VARIANTS * 5
total_empty_dot = N_VARIANTS * 5
total_sprites = len(sprite_entries) * N_VARIANTS
total_tasks = total_empty + total_empty_dot + total_sprites

if __name__ == "__main__":
    # Prepare common configuration to pass to workers
    common_config = {
        'PATCH_SIZE': PATCH_SIZE,
        'MAX_CROP': MAX_CROP,
        'MAX_SHIFT': MAX_SHIFT,
        # Augmentation params are global, apply_augmentations will access them
    }

    # 4) Prepare tasks for multiprocessing
    all_tasks_to_process = []

    # 4a) empty patches
    empty_dir = os.path.join(out_dir, "empty")
    empty_dot_dir = os.path.join(out_dir, "empty_dot")
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(empty_dot_dir, exist_ok=True)

    print(f"Preparing {total_empty} empty patch tasks...")
    for i in range(total_empty):
        bg = random.choice(backgrounds) # PIL Image object
        output_filename = f"empty_{i}.jpg"
        all_tasks_to_process.append(
            ("empty", output_filename, empty_dir, bg, None, common_config)
        )

    print(f"Preparing {total_empty_dot} empty_dot patch tasks...")
    for i in range(total_empty_dot):
        bg = random.choice(backgrounds_dot) # PIL Image object
        output_filename = f"empty_dot_{i}.jpg"
        all_tasks_to_process.append(
            ("empty_dot", output_filename, empty_dot_dir, bg, None, common_config)
        )

    print(f"Preparing {total_sprites} sprite patch tasks...")
    for style_folder, sprite_file in sprite_entries:
        style_path = os.path.join(sprites_dir, style_folder)
        sprite_original = Image.open(os.path.join(style_path, sprite_file)).convert("RGBA")
        
        new_w = int(sprite_original.width * scale)
        new_h = int(sprite_original.height * scale)
        sprite_resized_pil = sprite_original.resize((new_w, new_h), Image.LANCZOS)
        sprite_info_tuple = (sprite_resized_pil, new_w, new_h)

        cls = os.path.splitext(sprite_file)[0]
        save_dir_for_cls = os.path.join(out_dir, cls)
        os.makedirs(save_dir_for_cls, exist_ok=True) # Ensure dir exists

        for i in range(N_VARIANTS):
            bg = random.choice(backgrounds) # PIL Image object
            output_filename = f"{cls}_{style_folder}_{i}.jpg"
            all_tasks_to_process.append(
                ("sprite", output_filename, save_dir_for_cls, bg, sprite_info_tuple, common_config)
            )
    
    # 5) Run generation with multiprocessing and a single tqdm bar
    print(f"\nStarting patch generation with {NUM_WORKERS} workers for {len(all_tasks_to_process)} total patches...")
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Use imap_unordered for better tqdm updates if tasks are of varying length,
        # or if you want to see progress more smoothly.
        # list() is used to ensure all tasks are processed before tqdm closes.
        list(tqdm(pool.imap_unordered(generate_patch_worker, all_tasks_to_process), total=len(all_tasks_to_process), desc="Generating Patches"))

    print("✅ All patches generated.")
