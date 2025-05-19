from PIL import Image, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, required=True)
args = parser.parse_args()
num = args.num

# Load base board as RGBA to allow transparency
board = Image.open(f"./boards/{num}.png")
board = board.convert("RGBA")  # Ensures RGBA mode
width, height = board.size
square_size = width // 8

# Create transparent overlay (same size as board)
overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Draw transparent circles
circle_radius = square_size // 6
circle_color = (0, 0, 0, 30)  # Semi-transparent black

for row in range(8):
    for col in range(8):
        center_x = col * square_size + square_size // 2
        center_y = row * square_size + square_size // 2
        draw.ellipse(
            (center_x - circle_radius, center_y - circle_radius,
             center_x + circle_radius, center_y + circle_radius),
            fill=circle_color
        )

# Combine the overlay with the board
combined = Image.alpha_composite(board, overlay)

# Save with the original format and DPI (if desired)
combined.save(f"./boards_with_dot/{num}.png", format="PNG")
