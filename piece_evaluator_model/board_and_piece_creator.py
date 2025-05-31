#!/usr/bin/env python3
import os
import argparse
from PIL import Image, ImageDraw
import math


def parse_color(s):
    """
    Parse a color string 'R,G,B[,A]' into a tuple of ints.
    """
    parts = [int(x) for x in s.strip().split(',')]
    if len(parts) == 3:
        parts.append(255)
    return tuple(parts)


def add_dot_to_board(img, dot_color, dot_radius_factor, draw_scale_factor=1):
    """
    Draw a filled dot on every square of an 8×8 board image.
    """
    img = img.convert('RGBA')
    w, h = img.size

    # Create a higher-resolution overlay for drawing
    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    dot_r = int(square * dot_radius_factor)
    for row in range(8):
        for col in range(8):
            cx = col * square + square // 2
            cy = row * square + square // 2
            draw_hires.ellipse(
                (cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                fill=dot_color
            )

    # Scale down the high-resolution overlay and composite
    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_ring_to_board(img, ring_color, ring_radius_factor, ring_thickness_factor, draw_scale_factor=1):
    """
    Draw an outline ring on every square of an 8×8 board image.
    """
    img = img.convert('RGBA')
    w, h = img.size

    # Create a higher-resolution overlay for drawing
    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    ring_r = int(square * ring_radius_factor)
    thickness = max(1, int(square * ring_thickness_factor * draw_scale_factor)) # Scale thickness too

    for row in range(8):
        for col in range(8):
            cx = col * square + square // 2
            cy = row * square + square // 2
            bbox = (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r)
            draw_hires.ellipse(bbox, outline=ring_color, width=thickness)

    # Scale down the high-resolution overlay and composite
    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_ring_to_piece(img, ring_color, ring_radius_factor, ring_thickness_factor, draw_scale_factor=1):
    """
    Draw a single outline ring centered on the piece image.
    """
    img = img.convert('RGBA')
    w, h = img.size

    # Create a higher-resolution overlay for drawing
    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    r = int(min(scaled_w, scaled_h) * ring_radius_factor)
    thickness = max(1, int(min(scaled_w, scaled_h) * ring_thickness_factor * draw_scale_factor))
    cx, cy = scaled_w // 2, scaled_h // 2
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw_hires.ellipse(bbox, outline=ring_color, width=thickness)

    # Scale down the high-resolution overlay and composite
    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def draw_arrow_on_square(draw, square_coords, arrow_color, head_height, overall_width, direction):
    """
    Helper function to draw an arrow within a given square on the high-res overlay.
    'square_coords' is a tuple (square_x, square_y, square_size)
    """
    square_x, square_y, square_size = square_coords
    cx = square_x + square_size // 2
    cy = square_y + square_size // 2

    shaft_width = int(overall_width * 0.5)
    shaft_width = shaft_width if shaft_width % 2 == 0 else shaft_width + 1

    # Define points based on direction
    if direction == 'north':
        tri_tip_x, tri_tip_y = cx, cy
        tri_base_y = cy + head_height
        tri_left_x, tri_right_x = cx - overall_width // 2, cx + overall_width // 2
        shaft_start_y = tri_base_y
        shaft_end_y = square_y + square_size - 1 # Bottom edge of square
        draw.polygon([(tri_tip_x, tri_tip_y), (tri_left_x, tri_base_y), (tri_right_x, tri_base_y)], fill=arrow_color)
        draw.rectangle((cx - shaft_width // 2, shaft_start_y, cx + shaft_width // 2, shaft_end_y), fill=arrow_color)
    elif direction == 'south':
        tri_tip_x, tri_tip_y = cx, cy
        tri_base_y = cy - head_height
        tri_left_x, tri_right_x = cx - overall_width // 2, cx + overall_width // 2
        shaft_start_y = tri_base_y
        shaft_end_y = square_y + 1 # Top edge of square
        draw.polygon([(tri_tip_x, tri_tip_y), (tri_left_x, tri_base_y), (tri_right_x, tri_base_y)], fill=arrow_color)
        draw.rectangle((cx - shaft_width // 2, shaft_end_y, cx + shaft_width // 2, shaft_start_y), fill=arrow_color) # Draw top to bottom
    elif direction == 'west':
        tri_tip_x, tri_tip_y = cx, cy
        tri_base_x = cx + head_height
        tri_top_y, tri_bottom_y = cy - overall_width // 2, cy + overall_width // 2
        shaft_start_x = tri_base_x
        shaft_end_x = square_x + square_size - 1 # Right edge of square
        draw.polygon([(tri_tip_x, tri_tip_y), (tri_base_x, tri_top_y), (tri_base_x, tri_bottom_y)], fill=arrow_color)
        draw.rectangle((shaft_start_x, cy - shaft_width // 2, shaft_end_x, cy + shaft_width // 2), fill=arrow_color)
    elif direction == 'east':
        tri_tip_x, tri_tip_y = cx, cy
        tri_base_x = cx - head_height
        tri_top_y, tri_bottom_y = cy - overall_width // 2, cy + overall_width // 2
        shaft_start_x = tri_base_x
        shaft_end_x = square_x + 1 # Left edge of square
        draw.polygon([(tri_tip_x, tri_tip_y), (tri_base_x, tri_top_y), (tri_base_x, tri_bottom_y)], fill=arrow_color)
        draw.rectangle((shaft_end_x, cy - shaft_width // 2, shaft_start_x, cy + shaft_width // 2), fill=arrow_color) # Draw left to right

def rotate_point(p, angle, origin):
    """
    Rotate a point p around an origin.
    p: (x, y)
    angle: radians
    origin: (ox, oy)
    """
    ox, oy = origin
    px, py = p

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def draw_diagonal_arrow_on_square(draw, square_coords, arrow_color, head_height, overall_width, direction):
    """
    Helper function to draw a diagonal arrow within a given square on the high-res overlay.
    'square_coords' is a tuple (square_x, square_y, square_size)
    This function now draws the arrow on a temporary image that is the size of the square,
    and then pastes it onto the main overlay. This ensures the arrow is clipped to the square.
    """
    square_x, square_y, square_size = square_coords
    
    # Create a temporary square-sized image for drawing the arrow
    temp_square_img = Image.new('RGBA', (square_size, square_size), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_square_img)

    # The arrow will be centered within this temporary square image, so its "center"
    # is now relative to the temp_square_img's top-left (0,0)
    cx_temp = square_size // 2
    cy_temp = square_size // 2

    shaft_width = int(overall_width * 0.5)
    shaft_width = shaft_width if shaft_width % 2 == 0 else shaft_width + 1

    if direction == 'north_east':
        # Arrowhead tip is at (cx_temp, cy_temp)
        side_len = overall_width
        h_offset = head_height
        angle_arrowhead = -math.pi / 4 # North-East direction

        # Rotate reference points around (0,0) and translate to (cx_temp, cy_temp)
        ref_tip = (0, 0)
        ref_base_x = -h_offset
        ref_left = (ref_base_x, -side_len / 2)
        ref_right = (ref_base_x, side_len / 2)

        rot_ref_tip = rotate_point(ref_tip, angle_arrowhead, (0,0))
        rot_ref_left = rotate_point(ref_left, angle_arrowhead, (0,0))
        rot_ref_right = rotate_point(ref_right, angle_arrowhead, (0,0))

        tri_tip = (cx_temp + rot_ref_tip[0], cy_temp + rot_ref_tip[1])
        tri_left = (cx_temp + rot_ref_left[0], cy_temp + rot_ref_left[1])
        tri_right = (cx_temp + rot_ref_right[0], cy_temp + rot_ref_right[1])

        temp_draw.polygon([tri_tip, tri_left, tri_right], fill=arrow_color)

        # For the shaft: It extends from the arrowhead base towards the SW corner of the square.
        base_mid_x = (tri_left[0] + tri_right[0]) / 2
        base_mid_y = (tri_left[1] + tri_right[1]) / 2

        # Southwest corner of the temporary square (0,0) to (square_size, square_size)
        sw_corner_x = 0
        sw_corner_y = square_size

        # Calculate angle of the shaft
        shaft_angle = math.atan2(sw_corner_y - base_mid_y, sw_corner_x - base_mid_x)

        # Calculate shaft length (from base_mid to the square boundary in the direction of the corner)
        # This is tricky because we need to find the intersection point with the square's edge.
        # A simpler way, given we are drawing on a clipped canvas, is to make the shaft long enough
        # to guarantee it reaches the corner, and the clipping will handle the rest.
        # Let's make the shaft extend slightly beyond the square's diagonal if needed.
        # Max possible length for a diagonal in a square is sqrt(2) * side_size
        shaft_len = math.sqrt(2) * square_size 

        # Define points for a rectangle representing the shaft, originating from base_mid
        # and extending in the direction of shaft_angle
        # These points are relative to base_mid_x, base_mid_y
        rect_p1_rel = (0, -shaft_width / 2)
        rect_p2_rel = (shaft_len, -shaft_width / 2)
        rect_p3_rel = (shaft_len, shaft_width / 2)
        rect_p4_rel = (0, shaft_width / 2)

        rotated_rect_points = []
        for p_rel in [rect_p1_rel, rect_p2_rel, rect_p3_rel, rect_p4_rel]:
            # Rotate points around (0,0) and then translate by (base_mid_x, base_mid_y)
            rx = math.cos(shaft_angle) * p_rel[0] - math.sin(shaft_angle) * p_rel[1]
            ry = math.sin(shaft_angle) * p_rel[0] + math.cos(shaft_angle) * p_rel[1]
            rotated_rect_points.append((base_mid_x + rx, base_mid_y + ry))
        
        temp_draw.polygon(rotated_rect_points, fill=arrow_color)

    # Paste the temporary square image onto the main high-resolution overlay
    draw.paste(temp_square_img, (square_x, square_y), temp_square_img)


def add_north_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw an arrow pointing upwards on every square of an 8x8 board image.
    The arrowhead's tip is at the square's center, and the shaft extends to the bottom edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square_hires = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    for row in range(8):
        for col in range(8):
            square_x_hires = col * square_hires
            square_y_hires = row * square_hires

            head_height_hires = int(square_hires * arrow_head_height_factor)
            overall_width_hires = int(square_hires * arrow_width_factor)
            overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

            draw_arrow_on_square(
                draw_hires,
                (square_x_hires, square_y_hires, square_hires),
                arrow_color,
                head_height_hires,
                overall_width_hires,
                'north'
            )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_south_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw an arrow pointing downwards on every square of an 8x8 board image.
    The arrowhead's tip is at the square's center, and the shaft extends to the top edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square_hires = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    for row in range(8):
        for col in range(8):
            square_x_hires = col * square_hires
            square_y_hires = row * square_hires

            head_height_hires = int(square_hires * arrow_head_height_factor)
            overall_width_hires = int(square_hires * arrow_width_factor)
            overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

            draw_arrow_on_square(
                draw_hires,
                (square_x_hires, square_y_hires, square_hires),
                arrow_color,
                head_height_hires,
                overall_width_hires,
                'south'
            )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_west_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw an arrow pointing left on every square of an 8x8 board image.
    The arrowhead's tip is at the square's center, and the shaft extends to the right edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square_hires = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    for row in range(8):
        for col in range(8):
            square_x_hires = col * square_hires
            square_y_hires = row * square_hires

            head_height_hires = int(square_hires * arrow_head_height_factor)
            overall_width_hires = int(square_hires * arrow_width_factor)
            overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

            draw_arrow_on_square(
                draw_hires,
                (square_x_hires, square_y_hires, square_hires),
                arrow_color,
                head_height_hires,
                overall_width_hires,
                'west'
            )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_east_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw an arrow pointing right on every square of an 8x8 board image.
    The arrowhead's tip is at the square's center, and the shaft extends to the left edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square_hires = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    for row in range(8):
        for col in range(8):
            square_x_hires = col * square_hires
            square_y_hires = row * square_hires

            head_height_hires = int(square_hires * arrow_head_height_factor)
            overall_width_hires = int(square_hires * arrow_width_factor)
            overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

            draw_arrow_on_square(
                draw_hires,
                (square_x_hires, square_y_hires, square_hires),
                arrow_color,
                head_height_hires,
                overall_width_hires,
                'east'
            )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)

def add_north_east_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw an arrow pointing North-East on every square of an 8x8 board image.
    The arrowhead's tip is at the square's center, and the shaft extends from the southwest corner.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    square_hires = scaled_w // 8
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    # No direct draw object here, as draw_diagonal_arrow_on_square now handles its own temporary draw surface

    for row in range(8):
        for col in range(8):
            square_x_hires = col * square_hires
            square_y_hires = row * square_hires

            head_height_hires = int(square_hires * arrow_head_height_factor)
            overall_width_hires = int(square_hires * arrow_width_factor)
            overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

            # Call draw_diagonal_arrow_on_square which now draws and pastes
            # onto the main overlay_hires
            draw_diagonal_arrow_on_square(
                overlay_hires, # Pass the main overlay for pasting
                (square_x_hires, square_y_hires, square_hires),
                arrow_color,
                head_height_hires,
                overall_width_hires,
                'north_east'
            )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_north_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw a single arrow pointing upwards centered on the piece image.
    The arrowhead's tip is at the piece's center, and the shaft extends to the bottom edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    cx_hires, cy_hires = scaled_w // 2, scaled_h // 2
    base_dim_hires = min(scaled_w, scaled_h)

    head_height_hires = int(base_dim_hires * arrow_head_height_factor)
    overall_width_hires = int(base_dim_hires * arrow_width_factor)
    overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

    # Arrowhead tip is at (cx_hires, cy_hires)
    tri_tip_y_hires = cy_hires
    tri_base_y_hires = cy_hires + head_height_hires
    tri_left_x_hires = cx_hires - overall_width_hires // 2
    tri_right_x_hires = cx_hires + overall_width_hires // 2

    # Shaft dimensions (also scaled)
    shaft_width_hires = int(overall_width_hires * 0.5)
    shaft_width_hires = shaft_width_hires if shaft_width_hires % 2 == 0 else shaft_width_hires + 1
    shaft_start_y_hires = tri_base_y_hires
    shaft_end_y_hires = scaled_h - 1 # Go to the bottom edge of the high-res piece

    draw_hires.polygon(
        [
            (cx_hires, tri_tip_y_hires),
            (tri_left_x_hires, tri_base_y_hires),
            (tri_right_x_hires, tri_base_y_hires)
        ],
        fill=arrow_color
    )

    draw_hires.rectangle(
        (cx_hires - shaft_width_hires // 2, shaft_start_y_hires, cx_hires + shaft_width_hires // 2, shaft_end_y_hires),
        fill=arrow_color
    )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_south_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw a single arrow pointing downwards centered on the piece image.
    The arrowhead's tip is at the piece's center, and the shaft extends to the top edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    cx_hires, cy_hires = scaled_w // 2, scaled_h // 2
    base_dim_hires = min(scaled_w, scaled_h)

    head_height_hires = int(base_dim_hires * arrow_head_height_factor)
    overall_width_hires = int(base_dim_hires * arrow_width_factor)
    overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

    # Arrowhead tip is at (cx_hires, cy_hires)
    tri_tip_y_hires = cy_hires
    tri_base_y_hires = cy_hires - head_height_hires # Base is above tip
    tri_left_x_hires = cx_hires - overall_width_hires // 2
    tri_right_x_hires = cx_hires + overall_width_hires // 2

    # Shaft dimensions (also scaled)
    shaft_width_hires = int(overall_width_hires * 0.5)
    shaft_width_hires = shaft_width_hires if shaft_width_hires % 2 == 0 else shaft_width_hires + 1
    shaft_start_y_hires = tri_base_y_hires
    shaft_end_y_hires = 1 # Go to the top edge of the high-res piece (1 for safety)

    draw_hires.polygon(
        [
            (cx_hires, tri_tip_y_hires),
            (tri_left_x_hires, tri_base_y_hires),
            (tri_right_x_hires, tri_base_y_hires)
        ],
        fill=arrow_color
    )

    draw_hires.rectangle(
        (cx_hires - shaft_width_hires // 2, shaft_end_y_hires, cx_hires + shaft_width_hires // 2, shaft_start_y_hires),
        fill=arrow_color
    )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_west_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw a single arrow pointing left centered on the piece image.
    The arrowhead's tip is at the piece's center, and the shaft extends to the right edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    cx_hires, cy_hires = scaled_w // 2, scaled_h // 2
    base_dim_hires = min(scaled_w, scaled_h)

    head_height_hires = int(base_dim_hires * arrow_head_height_factor)
    overall_width_hires = int(base_dim_hires * arrow_width_factor)
    overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

    # Arrowhead tip is at (cx_hires, cy_hires)
    tri_tip_x_hires = cx_hires
    tri_base_x_hires = cx_hires + head_height_hires # Base is to the right of tip
    tri_top_y_hires = cy_hires - overall_width_hires // 2
    tri_bottom_y_hires = cy_hires + overall_width_hires // 2

    # Shaft dimensions (also scaled)
    shaft_width_hires = int(overall_width_hires * 0.5)
    shaft_width_hires = shaft_width_hires if shaft_width_hires % 2 == 0 else shaft_width_hires + 1
    shaft_start_x_hires = tri_base_x_hires
    shaft_end_x_hires = scaled_w - 1 # Go to the right edge of the high-res piece
    tri_tip_y_hires = cy_hires

    draw_hires.polygon(
        [
            (tri_tip_x_hires, tri_tip_y_hires),
            (tri_base_x_hires, tri_top_y_hires),
            (tri_base_x_hires, tri_bottom_y_hires)
        ],
        fill=arrow_color
    )

    draw_hires.rectangle(
        (shaft_start_x_hires, cy_hires - shaft_width_hires // 2, shaft_end_x_hires, cy_hires + shaft_width_hires // 2),
        fill=arrow_color
    )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def add_east_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw a single arrow pointing right centered on the piece image.
    The arrowhead's tip is at the piece's center, and the shaft extends to the left edge.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    cx_hires, cy_hires = scaled_w // 2, scaled_h // 2
    base_dim_hires = min(scaled_w, scaled_h)

    head_height_hires = int(base_dim_hires * arrow_head_height_factor)
    overall_width_hires = int(base_dim_hires * arrow_width_factor)
    overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

    # Arrowhead tip is at (cx_hires, cy_hires)
    tri_tip_x_hires = cx_hires
    tri_base_x_hires = cx_hires - head_height_hires # Base is to the left of tip
    tri_top_y_hires = cy_hires - overall_width_hires // 2
    tri_bottom_y_hires = cy_hires + overall_width_hires // 2

    # Shaft dimensions (also scaled)
    shaft_width_hires = int(overall_width_hires * 0.5)
    shaft_width_hires = shaft_width_hires if shaft_width_hires % 2 == 0 else shaft_width_hires + 1
    shaft_start_x_hires = tri_base_x_hires
    shaft_end_x_hires = 1 # Go to the left edge of the high-res piece (1 for safety)
    tri_tip_y_hires = cy_hires

    draw_hires.polygon(
        [
            (tri_tip_x_hires, tri_tip_y_hires),
            (tri_base_x_hires, tri_top_y_hires),
            (tri_base_x_hires, tri_bottom_y_hires)
        ],
        fill=arrow_color
    )

    draw_hires.rectangle(
        (shaft_end_x_hires, cy_hires - shaft_width_hires // 2, shaft_start_x_hires, cy_hires + shaft_width_hires // 2),
        fill=arrow_color
    )

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)

def add_north_east_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor=1):
    """
    Draw a single arrow pointing North-East centered on the piece image.
    The arrowhead's tip is at the piece's center, and the shaft extends from the southwest corner.
    """
    img = img.convert('RGBA')
    w, h = img.size

    scaled_w, scaled_h = w * draw_scale_factor, h * draw_scale_factor
    overlay_hires = Image.new('RGBA', (scaled_w, scaled_h), (0, 0, 0, 0))
    draw_hires = ImageDraw.Draw(overlay_hires)

    cx_hires, cy_hires = scaled_w // 2, scaled_h // 2
    base_dim_hires = min(scaled_w, scaled_h)

    head_height_hires = int(base_dim_hires * arrow_head_height_factor)
    overall_width_hires = int(base_dim_hires * arrow_width_factor)
    overall_width_hires = overall_width_hires if overall_width_hires % 2 == 0 else overall_width_hires + 1

    # Arrowhead tip is at (cx_hires, cy_hires)
    side_len = overall_width_hires
    h_offset = head_height_hires
    angle_arrowhead = -math.pi / 4 # North-East direction

    # Rotate reference points around (0,0) and translate to (cx_hires, cy_hires)
    ref_tip = (0, 0)
    ref_base_x = -h_offset
    ref_left = (ref_base_x, -side_len / 2)
    ref_right = (ref_base_x, side_len / 2)

    rot_ref_tip = rotate_point(ref_tip, angle_arrowhead, (0,0))
    rot_ref_left = rotate_point(ref_left, angle_arrowhead, (0,0))
    rot_ref_right = rotate_point(ref_right, angle_arrowhead, (0,0))

    tri_tip = (cx_hires + rot_ref_tip[0], cy_hires + rot_ref_tip[1])
    tri_left = (cx_hires + rot_ref_left[0], cy_hires + rot_ref_left[1])
    tri_right = (cx_hires + rot_ref_right[0], cy_hires + rot_ref_right[1])

    draw_hires.polygon([tri_tip, tri_left, tri_right], fill=arrow_color)

    # For the shaft: It extends from the arrowhead base towards the SW corner of the piece.
    base_mid_x = (tri_left[0] + tri_right[0]) / 2
    base_mid_y = (tri_left[1] + tri_right[1]) / 2

    # Southwest corner of the piece (bottom-left)
    sw_corner_x = 0 # Left edge of the high-res piece
    sw_corner_y = scaled_h - 1 # Bottom edge of the high-res piece

    shaft_len = math.sqrt((sw_corner_x - base_mid_x)**2 + (sw_corner_y - base_mid_y)**2)
    shaft_angle = math.atan2(sw_corner_y - base_mid_y, sw_corner_x - base_mid_x)

    shaft_width_hires = int(overall_width_hires * 0.5)
    shaft_width_hires = shaft_width_hires if shaft_width_hires % 2 == 0 else shaft_width_hires + 1

    rect_p1 = (0, -shaft_width_hires / 2)
    rect_p2 = (shaft_len, -shaft_width_hires / 2)
    rect_p3 = (shaft_len, shaft_width_hires / 2)
    rect_p4 = (0, shaft_width_hires / 2)

    rotated_rect_points = []
    for p in [rect_p1, rect_p2, rect_p3, rect_p4]:
        rx = math.cos(shaft_angle) * p[0] - math.sin(shaft_angle) * p[1]
        ry = math.sin(shaft_angle) * p[0] + math.cos(shaft_angle) * p[1]
        rotated_rect_points.append((base_mid_x + rx, base_mid_y + ry))

    draw_hires.polygon(rotated_rect_points, fill=arrow_color)

    overlay_lores = overlay_hires.resize((w, h), Image.LANCZOS)
    return Image.alpha_composite(img, overlay_lores)


def process_boards(in_dir, out_dir,
                   dot_color, dot_radius_factor,
                   ring_color, ring_radius_factor, ring_thickness_factor,
                   arrow_color, arrow_head_height_factor, arrow_width_factor,
                   draw_scale_factor):
    """
    Copy original boards and save dot-only, ring-only, and various arrow versions.
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
        dot_img = add_dot_to_board(img, dot_color, dot_radius_factor, draw_scale_factor)
        dot_dst = os.path.join(out_dir, f"{name}_dot{ext}")
        dot_img.save(dot_dst)

        # ring version
        ring_img = add_ring_to_board(img, ring_color, ring_radius_factor, ring_thickness_factor, draw_scale_factor)
        ring_dst = os.path.join(out_dir, f"{name}_ring{ext}")
        ring_img.save(ring_dst)

        # arrow versions
        north_arrow_img = add_north_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
        north_arrow_dst = os.path.join(out_dir, f"{name}_arrow_north{ext}")
        north_arrow_img.save(north_arrow_dst)

        south_arrow_img = add_south_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
        south_arrow_dst = os.path.join(out_dir, f"{name}_arrow_south{ext}")
        south_arrow_img.save(south_arrow_dst)

        west_arrow_img = add_west_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
        west_arrow_dst = os.path.join(out_dir, f"{name}_arrow_west{ext}")
        west_arrow_img.save(west_arrow_dst)

        east_arrow_img = add_east_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
        east_arrow_dst = os.path.join(out_dir, f"{name}_arrow_east{ext}")
        east_arrow_img.save(east_arrow_dst)

        north_east_arrow_img = add_north_east_arrow_to_board(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
        north_east_arrow_dst = os.path.join(out_dir, f"{name}_arrow_north_east{ext}")
        north_east_arrow_img.save(north_east_arrow_dst)


def process_pieces(in_dir, out_dir,
                   ring_color, ring_radius_factor, ring_thickness_factor,
                   arrow_color, arrow_head_height_factor, arrow_width_factor,
                   draw_scale_factor):
    """
    Copy original piece images and save ring-only and various arrow versions alongside.
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
            ring = add_ring_to_piece(img, ring_color, ring_radius_factor, ring_thickness_factor, draw_scale_factor)
            ring_dst = os.path.join(target_dir, f"{name}_ring{ext}")
            ring.save(ring_dst)

            # arrow versions
            north_arrow = add_north_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
            north_arrow_dst = os.path.join(target_dir, f"{name}_arrow_north{ext}")
            north_arrow.save(north_arrow_dst)

            south_arrow = add_south_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
            south_arrow_dst = os.path.join(target_dir, f"{name}_arrow_south{ext}")
            south_arrow.save(south_arrow_dst)

            west_arrow = add_west_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
            west_arrow_dst = os.path.join(target_dir, f"{name}_arrow_west{ext}")
            west_arrow.save(west_arrow_dst)

            east_arrow = add_east_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
            east_arrow_dst = os.path.join(target_dir, f"{name}_arrow_east{ext}")
            east_arrow.save(east_arrow_dst)

            north_east_arrow = add_north_east_arrow_to_piece(img, arrow_color, arrow_head_height_factor, arrow_width_factor, draw_scale_factor)
            north_east_arrow_dst = os.path.join(target_dir, f"{name}_arrow_north_east{ext}")
            north_east_arrow.save(north_east_arrow_dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate dart boards and piece capture rings and arrows')
    parser.add_argument('--boards-in', type=str, required=True,
                        help='Directory of input boards')
    parser.add_argument('--boards-out', type=str, default='boards_final',
                        help='Output directory for processed boards')
    parser.add_argument('--pieces-in', type=str, required=True,
                        help='Root directory of piece images')
    parser.add_argument('--pieces-out', type=str, default='pieces_with_overlays',
                        help='Output root for piece images with ring/arrow')
    parser.add_argument('--color', type=str, default='0,0,0,50',
                        help="Default overlay color as 'R,G,B,A'")
    parser.add_argument('--dot-radius-factor', type=float, default=1/6,
                        help='Dot radius = square_size * factor')
    parser.add_argument('--ring-radius-factor', type=float, default=0.35,
                        help='Ring radius = square_size * factor')
    parser.add_argument('--ring-thickness-factor', type=float, default=0.10,
                        help='Ring thickness = square_size * factor')
    parser.add_argument('--arrow-head-height-factor', type=float, default=0.3,
                        help='Arrowhead height = square_size * factor (for boards) or min_dim * factor (for pieces)')
    parser.add_argument('--arrow-width-factor', type=float, default=0.5,
                        help='Overall arrow width = square_size * factor (for boards) or min_dim * factor (for pieces)')
    parser.add_argument('--draw-scale-factor', type=int, default=4,
                        help='Scale factor for drawing overlays at higher resolution for smoother results (e.g., 2, 4, 8).')
    args = parser.parse_args()

    # Define a specific color for arrows, distinct from generic 'color' for other overlays
    ARROW_COLOR = "255,170,0" # A nice orange-ish color

    dot_color = parse_color(args.color)
    ring_color = parse_color(args.color)
    arrow_color = parse_color(ARROW_COLOR) # Use the specific arrow color

    # Process boards with all arrow directions
    process_boards(
        args.boards_in, args.boards_out,
        dot_color, args.dot_radius_factor,
        ring_color, args.ring_radius_factor, args.ring_thickness_factor,
        arrow_color, args.arrow_head_height_factor, args.arrow_width_factor,
        args.draw_scale_factor
    )

    # Process pieces with all arrow directions
    process_pieces(
        args.pieces_in, args.pieces_out,
        ring_color, args.ring_radius_factor, args.ring_thickness_factor,
        arrow_color, args.arrow_head_height_factor, args.arrow_width_factor,
        args.draw_scale_factor
    )