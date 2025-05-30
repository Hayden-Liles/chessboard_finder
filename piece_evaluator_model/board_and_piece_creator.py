#!/usr/bin/env python3
import os
import argparse
from PIL import Image, ImageDraw


def parse_color(s):
    """
    Parse a color string 'R,G,B[,A]' into a tuple of ints.
    """
    parts = [int(x) for x in s.strip().split(',')]
    if len(parts) == 3:
        parts.append(255)
    return tuple(parts)


def add_dot_to_board(img, dot_color, dot_radius_factor):
    """
    Draw a filled dot on every square of an 8×8 board image.
    """
    img = img.convert('RGBA')
    w, h = img.size
    square = w // 8
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    dot_r = int(square * dot_radius_factor)
    for row in range(8):
        for col in range(8):
            cx = col * square + square // 2
            cy = row * square + square // 2
            draw.ellipse(
                (cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                fill=dot_color
            )
    return Image.alpha_composite(img, overlay)


def add_ring_to_board(img, ring_color, ring_radius_factor, ring_thickness_factor):
    """
    Draw an outline ring on every square of an 8×8 board image.
    """
    img = img.convert('RGBA')
    w, h = img.size
    square = w // 8
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    ring_r = int(square * ring_radius_factor)
    thickness = max(1, int(square * ring_thickness_factor))
    for row in range(8):
        for col in range(8):
            cx = col * square + square // 2
            cy = row * square + square // 2
            bbox = (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r)
            draw.ellipse(bbox, outline=ring_color, width=thickness)
    return Image.alpha_composite(img, overlay)


def add_ring_to_piece(img, ring_color, ring_radius_factor, ring_thickness_factor):
    """
    Draw a single outline ring centered on the piece image.
    """
    img = img.convert('RGBA')
    w, h = img.size
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    r = int(min(w, h) * ring_radius_factor)
    thickness = max(1, int(min(w, h) * ring_thickness_factor))
    cx, cy = w // 2, h // 2
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse(bbox, outline=ring_color, width=thickness)
    return Image.alpha_composite(img, overlay)


def process_boards(in_dir, out_dir,
                   dot_color, dot_radius_factor,
                   ring_color, ring_radius_factor, ring_thickness_factor):
    """
    Copy original boards and save dot-only and ring-only versions.
    """
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        src = os.path.join(in_dir, fn)
        name, ext = os.path.splitext(fn)
        img = Image.open(src)

        # original
        orig_dst = os.path.join(out_dir, fn)
        img.save(orig_dst)

        # dot version
        dot_img = add_dot_to_board(img, dot_color, dot_radius_factor)
        dot_dst = os.path.join(out_dir, f"{name}_dot{ext}")
        dot_img.save(dot_dst)

        # ring version
        ring_img = add_ring_to_board(img, ring_color, ring_radius_factor, ring_thickness_factor)
        ring_dst = os.path.join(out_dir, f"{name}_ring{ext}")
        ring_img.save(ring_dst)


def process_pieces(in_dir, out_dir, ring_color, ring_radius_factor, ring_thickness_factor):
    """
    Copy original piece images and save a ring-only version alongside.
    """
    for root, dirs, files in os.walk(in_dir):
        rel = os.path.relpath(root, in_dir)
        target_dir = os.path.join(out_dir, rel)
        os.makedirs(target_dir, exist_ok=True)

        for fn in files:
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            src = os.path.join(root, fn)
            name, ext = os.path.splitext(fn)
            img = Image.open(src).convert('RGBA')

            # original
            orig_dst = os.path.join(target_dir, fn)
            img.save(orig_dst)

            # ring version
            ring = add_ring_to_piece(img, ring_color, ring_radius_factor, ring_thickness_factor)
            ring_dst = os.path.join(target_dir, f"{name}_ring{ext}")
            ring.save(ring_dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate dart boards and piece capture rings')
    parser.add_argument('--boards-in', type=str, required=True,
                        help='Directory of input boards')
    parser.add_argument('--boards-out', type=str, default='boards_final',
                        help='Output directory for processed boards')
    parser.add_argument('--pieces-in', type=str, required=True,
                        help='Root directory of piece images')
    parser.add_argument('--pieces-out', type=str, default='pieces_with_ring',
                        help='Output root for piece images with ring')
    parser.add_argument('--color', type=str, default='0,0,0,50',
                        help="Dot & ring color as 'R,G,B,A'")
    parser.add_argument('--dot-radius-factor', type=float, default=1/6,
                        help='Dot radius = square_size * factor')
    parser.add_argument('--ring-radius-factor', type=float, default=0.35,
                        help='Ring radius = square_size * factor')
    parser.add_argument('--ring-thickness-factor', type=float, default=0.10,
                        help='Ring thickness = square_size * factor')
    args = parser.parse_args()

    dot_color = parse_color(args.color)
    ring_color = dot_color

    # Process boards: original, dot, ring versions
    process_boards(
        args.boards_in, args.boards_out,
        dot_color, args.dot_radius_factor,
        ring_color, args.ring_radius_factor, args.ring_thickness_factor
    )

    # Process pieces: original and ring variants
    process_pieces(
        args.pieces_in, args.pieces_out,
        ring_color, args.ring_radius_factor, args.ring_thickness_factor
    )
