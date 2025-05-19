from PIL import Image
import os, random
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
boards_dir   = "boards"
boards_dot_dir = "boards_with_dot"
sprites_dir  = "pieces"
out_dir      = "data2/train"

PATCH_SIZE   = 244      # px
N_VARIANTS   = 1     # per sprite
MAX_SHIFT    = 10       # max px to shift sprite
MAX_CROP     = 8        # max px to crop from edges
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
total_empty = N_VARIANTS
total_empty_dot = N_VARIANTS * 20000
total_sprites = len(sprite_entries) * N_VARIANTS
total_tasks = total_empty + total_empty_dot + total_sprites

# 4) Run generation with a single tqdm bar
pbar = tqdm(total=total_tasks, desc="Generating patches")

# 4a) empty patches
empty_dir = os.path.join(out_dir, "empty")
empty_dot_dir = os.path.join(out_dir, "empty_dot")
os.makedirs(empty_dir, exist_ok=True)
os.makedirs(empty_dot_dir, exist_ok=True)

# Regular empty patches
for i in range(total_empty):
    bg = random.choice(backgrounds).copy()
    # random edge-crop
    l = random.randint(0, MAX_CROP)
    t = random.randint(0, MAX_CROP)
    r = random.randint(0, MAX_CROP)
    b = random.randint(0, MAX_CROP)
    w, h = bg.size
    c = bg.crop((l, t, w-r, h-b))
    patch = c.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
    patch.convert("RGB").save(os.path.join(empty_dir, f"empty_{i}.jpg"))
    pbar.update(1)

# Dotted empty patches
for i in range(total_empty_dot):
    bg = random.choice(backgrounds_dot).copy()
    # random edge-crop
    l = random.randint(0, MAX_CROP)
    t = random.randint(0, MAX_CROP)
    r = random.randint(0, MAX_CROP)
    b = random.randint(0, MAX_CROP)
    w, h = bg.size
    c = bg.crop((l, t, w-r, h-b))
    patch = c.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
    patch.convert("RGB").save(os.path.join(empty_dot_dir, f"empty_dot_{i}.jpg"))
    pbar.update(1)

# 4b) composite sprites
for style_folder, sprite_file in sprite_entries:
    style_path = os.path.join(sprites_dir, style_folder)
    sprite = Image.open(os.path.join(style_path, sprite_file)).convert("RGBA")
    # scale sprite so it fills the new patch proportionally
    new_w = int(sprite.width * scale)
    new_h = int(sprite.height * scale)
    sprite = sprite.resize((new_w, new_h), Image.LANCZOS)

    cls = os.path.splitext(sprite_file)[0]
    save_dir = os.path.join(out_dir, cls)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(N_VARIANTS):
        bg = random.choice(backgrounds).copy()
        dx = random.randint(-MAX_SHIFT, MAX_SHIFT)
        dy = random.randint(-MAX_SHIFT, MAX_SHIFT)
        x = (PATCH_SIZE - new_w)//2 + dx
        y = (PATCH_SIZE - new_h)//2 + dy
        bg.paste(sprite, (x, y), sprite)

        # random edge-crop
        l = random.randint(0, MAX_CROP)
        t = random.randint(0, MAX_CROP)
        r = random.randint(0, MAX_CROP)
        b = random.randint(0, MAX_CROP)
        w, h = bg.size
        c = bg.crop((l, t, w-r, h-b))
        patch = c.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

        out_name = f"{cls}_{style_folder}_{i}.jpg"
        patch.convert("RGB").save(os.path.join(save_dir, out_name))
        pbar.update(1)

pbar.close()
print("✅ All patches generated.")
