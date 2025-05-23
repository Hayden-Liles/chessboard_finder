"""
Digital chessboard detection with orientation recognition, square labeling, and piece identification.

This script detects a chess board in an image, determines its orientation,
labels all squares with proper chess notation (a1-h8), and identifies chess pieces
using a CNN model.

It can also process a directory of video frames, extract FEN positions from each frame,
and combine transcript segments with the same FEN position into a JSON file.

Steps:
1. Detect the board using color segmentation
2. Determine orientation by analyzing corner square colors
3. Label all 64 squares with proper chess notation
4. Identify the chess pieces on each square using a CNN model
5. Output annotated images showing the detected board and pieces
6. (Optional) Process a directory of video frames and extract FEN positions
7. (Optional) Combine transcript segments with the same FEN position
8. (Optional) Output the result in JSON format

Usage:
    # Process a single image
    python chess_board_detector.py path/to/image.png --model chess_piece_model.h5 --clusters 4 --downscale 800 --debug

    # Process a single video directory
    python chess_board_detector.py --process-video --video-dir path/to/video/imgs --srt-file path/to/transcript.srt --model chess_piece_model.h5 --output output.json

    # Process all videos in a parent directory (with parallel processing)
    python chess_board_detector.py --process-video --parent-dir path/to/videos --model chess_piece_model.h5 --max-parallel 4 --cpu-only

    # Process videos with GPU acceleration (if available)
    python chess_board_detector.py --process-video --parent-dir path/to/videos --model chess_piece_model.h5 --use-gpu
"""
import cv2
import numpy as np
import argparse
from itertools import combinations
import tensorflow as tf
import os
import json
import re
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Check if GPU is available
def is_gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    return len(gpus) > 0

# Set environment variable to control TensorFlow GPU memory allocation
# This helps prevent memory issues when using multiple processes
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Global flag to determine if we should use GPU
USE_GPU = is_gpu_available()

class Normalizer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / 255.0

    def get_config(self):
        return super().get_config()

# We'll refer to this function by name when loading:
_PREPROCESS_FN = tf.keras.applications.mobilenet_v2.preprocess_input

def order_points(pts):
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def extract_board_region(img, k=4, sample_size=10000, debug=False):
    """Extract chessboard corner points using color segmentation."""
    h, w = img.shape[:2]

    # 1) run k-means color clustering
    coords = np.random.choice(h*w, min(sample_size, h*w), replace=False)
    pixels = img.reshape(-1,3)[coords].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    all_pixels = img.reshape(-1,3).astype(np.float32)
    dists = np.linalg.norm(all_pixels[:,None] - centers[None,:], axis=2)
    full_labels = np.argmin(dists, axis=1).reshape(h, w)

    # 2) try every pair of clusters (i,j), build a mask and score by how square the largest contour is
    best_score = float('inf')
    best_box = None
    for i, j in combinations(range(k), 2):
        # build binary mask of these two clusters
        mask = np.zeros((h, w), np.uint8)
        mask[np.logical_or(full_labels==i, full_labels==j)] = 255

        # clean it up
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)

        # find the biggest blob
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # ignore tiny specks
        area = cv2.contourArea(cnt)
        if area < 0.01 * (h*w):
            continue

        # fit rotated rect and compute how far its aspect‐ratio deviates from 1.0
        rect = cv2.minAreaRect(cnt)
        _,(bw,bh),_ = rect  # Unpack but ignore center and angle
        aspect = max(bw,bh)/min(bw,bh)
        score = abs(1.0 - aspect)

        # heavily penalize masks that cover almost the whole image or almost nothing
        ar = area / float(h*w)
        if ar > 0.9 or ar < 0.02:
            score += 1.0

        if score < best_score:
            best_score = score
            best_box = cv2.boxPoints(rect).astype(np.float32)

            if debug:
                debug_mask = img.copy()
                cv2.drawContours(debug_mask, [cnt], -1, (0, 255, 0), 2)
                cv2.imwrite(f'debug_mask_{i}_{j}.png', debug_mask)

    if best_box is None:
        return None

    if debug:
        vis = img.copy()
        for i, (x, y) in enumerate(best_box):
            color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)][i]  # different color for each corner
            cv2.circle(vis, (int(x),int(y)), 10, color, -1)
            cv2.putText(vis, f"Corner {i}", (int(x)-10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imwrite('corners_color.png', vis)

    return best_box


def warp_and_draw(img, pts, size=800, thickness=2):
    """Warp the board to a square and overlay an 8x8 grid."""
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (size, size))

    overlay = np.zeros_like(warp)
    step = size // 8
    for i in range(9):
        cv2.line(overlay, (i*step,0), (i*step,size), (0,255,0), thickness)
        cv2.line(overlay, (0,i*step), (size,i*step), (0,255,0), thickness)

    back = cv2.warpPerspective(overlay, np.linalg.inv(M),
                              (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    result = img.copy()
    mask = back.sum(axis=2) > 0
    result[mask] = back[mask]
    return result, warp, M


def determine_orientation(warped_img, debug=False):
    """
    Determine board orientation by analyzing corner square colors.
    In standard chess orientation, the "white square in the right" rule applies:
    - Bottom-right (H1) is white
    - Top-left (A8) is white
    - Bottom-left (A1) is black
    - Top-right (H8) is black

    Returns an orientation code:
    0 - Standard orientation (A1 is bottom-left, black)
    1 - Rotated 90° clockwise (A1 is top-left, black)
    2 - Rotated 180° (A1 is top-right, black)
    3 - Rotated 270° clockwise (A1 is bottom-right, black)

    Note: This only determines the board orientation for chess notation.
    It cannot determine which player (white/black) is playing from which side
    without analyzing the pieces.
    """
    h, w = warped_img.shape[:2]
    square_size = h // 8

    # Sample colors from the 4 corner squares (offset from absolute corners to avoid grid lines)
    offset = square_size // 4

    # Corner indices in order: Top-left, Top-right, Bottom-right, Bottom-left
    corners = [
        (offset, offset),                   # Top-left (a8 in standard orientation)
        (w - offset, offset),               # Top-right (h8 in standard orientation)
        (w - offset, h - offset),           # Bottom-right (h1 in standard orientation)
        (offset, h - offset)                # Bottom-left (a1 in standard orientation)
    ]

    # Sample inner points of each corner square to get more reliable color
    corner_squares = []
    for x, y in corners:
        # Take 5x5 grid of samples in each corner square
        samples = []
        for dx in range(-offset//2, offset//2+1, offset//4):
            for dy in range(-offset//2, offset//2+1, offset//4):
                samples.append(warped_img[int(y+dy), int(x+dx)])

        # Average color of corner square
        corner_squares.append(np.mean(samples, axis=0))

    # Convert to brightness and determine if each corner is white
    brightness = [np.mean(color) for color in corner_squares]

    # Sort brightness to determine threshold (midpoint between light and dark)
    sorted_bright = sorted(brightness)
    threshold = (sorted_bright[1] + sorted_bright[2]) / 2

    # Determine if each corner is white
    is_white = [b > threshold for b in brightness]

    if debug:
        debug_img = warped_img.copy()
        for i, (x, y) in enumerate(corners):
            color = (0, 255, 0) if is_white[i] else (0, 0, 255)  # Green for white, Red for black
            cv2.circle(debug_img, (int(x), int(y)), 10, color, -1)
            cv2.putText(debug_img, f"{'W' if is_white[i] else 'B'}", (int(x)-5, int(y)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('corner_colors.png', debug_img)

    # Check orientation based on the pattern of white and black squares
    # In a standard chess board:
    # - TL: white (A8)
    # - TR: black (H8)
    # - BR: white (H1)
    # - BL: black (A1)

    # Standard orientation (A1 at bottom-left)
    if is_white[0] and not is_white[1] and is_white[2] and not is_white[3]:
        return 0

    # Rotated 90° clockwise (A1 at top-left)
    elif not is_white[0] and is_white[1] and not is_white[2] and is_white[3]:
        return 1

    # Rotated 180° (A1 at top-right)
    elif not is_white[0] and is_white[1] and not is_white[2] and is_white[3]:
        return 2

    # Rotated 270° clockwise (A1 at bottom-right)
    elif is_white[0] and not is_white[1] and is_white[2] and not is_white[3]:
        return 3

    # If the pattern doesn't match exactly, use a more reliable approach with diagonal colors
    # In a chessboard, squares on the same diagonal have the same color
    # Check if diagonal corners match
    diagonals_match = (is_white[0] == is_white[2]) and (is_white[1] == is_white[3])

    if diagonals_match:
        # If bottom-right (physical H1 in standard orientation) is white
        if is_white[2]:
            # Check bottom-left to determine if standard or rotated 270°
            if not is_white[3]:  # Bottom-left is black
                return 0  # Standard orientation
            else:  # Bottom-left is white
                return 3  # Rotated 270°
        else:  # Bottom-right is black
            # Check bottom-left to determine if rotated 90° or 180°
            if is_white[3]:  # Bottom-left is white
                return 1  # Rotated 90°
            else:  # Bottom-left is black
                return 2  # Rotated 180°

    # Last resort fallback
    print("Warning: Chess board color pattern unclear. Defaulting to standard orientation.")
    return 0


def label_squares(warped_img, orientation=0, font_scale=0.5, thickness=1):
    """
    Label all 64 squares with chess notation.

    Args:
        warped_img: Warped chessboard image
        orientation: Board orientation (0-3)

    Returns:
        labeled_img: Image with chess notation labels
    """
    h, _ = warped_img.shape[:2]  # Only need height
    square_size = h // 8

    labeled_img = warped_img.copy()
    files = "abcdefgh"
    ranks = "12345678"

    for row in range(8):
        for col in range(8):
            # Determine notation based on orientation
            if orientation == 0:  # Standard (A1 at bottom-left)
                notation = files[col] + ranks[7-row]
            elif orientation == 1:  # Rotated 90° clockwise (A1 at top-left)
                notation = files[7-row] + ranks[7-col]
            elif orientation == 2:  # Rotated 180° (A1 at top-right)
                notation = files[7-col] + ranks[row]
            elif orientation == 3:  # Rotated 270° clockwise (A1 at bottom-right)
                notation = files[row] + ranks[col]

            # Calculate center of square for label placement
            center_x = int((col + 0.5) * square_size)
            center_y = int((row + 0.5) * square_size)

            # Determine text color (for visibility)
            roi = labeled_img[row*square_size:(row+1)*square_size,
                              col*square_size:(col+1)*square_size]
            avg_brightness = np.mean(roi)
            text_color = (0, 0, 0) if avg_brightness > 128 else (255, 255, 255)

            # Add notation text
            cv2.putText(labeled_img, notation, (center_x-10, center_y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return labeled_img


def get_square_mapping(orientation=0):
    """
    Create a mapping from chessboard coordinates (0-7, 0-7) to chess notation.

    Args:
        orientation: Board orientation (0-3)

    Returns:
        mapping: Dictionary mapping (row, col) to chess notation
    """
    mapping = {}
    files = "abcdefgh"
    ranks = "12345678"

    for row in range(8):
        for col in range(8):
            # Determine notation based on orientation
            if orientation == 0:  # Standard (A1 at bottom-left)
                notation = files[col] + ranks[7-row]
            elif orientation == 1:  # Rotated 90° clockwise (A1 at top-left)
                notation = files[7-row] + ranks[7-col]
            elif orientation == 2:  # Rotated 180° (A1 at top-right)
                notation = files[7-col] + ranks[row]
            elif orientation == 3:  # Rotated 270° clockwise (A1 at bottom-right)
                notation = files[row] + ranks[col]

            mapping[(row, col)] = notation

    return mapping


def detect_piece_position(warped_img, x, y):
    """
    Determine chess notation for a given pixel position.

    Args:
        warped_img: Warped chessboard image
        x, y: Pixel coordinates in warped image

    Returns:
        notation: Chess notation (e.g., 'e4') for the square at (x,y)
        coords: Row, col coordinates (0-7, 0-7) of the square
    """
    h, _ = warped_img.shape[:2]  # Only need height
    square_size = h // 8

    # Convert pixels to square coordinates
    col = min(int(x // square_size), 7)
    row = min(int(y // square_size), 7)

    # Get orientation from a pre-detected orientation or detect it
    orientation = determine_orientation(warped_img)

    # Get mapping and look up notation
    mapping = get_square_mapping(orientation)
    notation = mapping[(row, col)]

    return notation, (row, col)


def analyze_chess_position(warped_img, model_path, orientation=0, debug=False):
    """
    Identify all 64 squares, classify each via the CNN, and return:
      - position dict (e.g. {'a1': 'k_w', …})
      - a visual overlay image
      - a FEN string
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1) load the model, registering both the old Normalizer and preprocess_input
    custom_objects = {
        'Normalizer': Normalizer,
        'preprocess_input': _PREPROCESS_FN
    }
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
    # ──────────────────────────────────────────────────────────────────────────

    # 2) load metadata for class->name mapping
    meta_path = os.path.splitext(model_path)[0] + ".keras.metadata"
    print(f"meta_path", meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    class_names = [meta["class_mapping"][str(i)]
                   for i in range(len(meta["class_mapping"]))]

    # 3) prep sizes & containers
    h, _ = warped_img.shape[:2]  # Only need height
    sz = h // 8
    visual = warped_img.copy()
    square_map = get_square_mapping(orientation)
    position = {}
    if debug:
        os.makedirs("squares", exist_ok=True)

    # 4) loop through each square, preprocess & predict
    for row in range(8):
        for col in range(8):
            x1, y1 = col * sz, row * sz
            sq = warped_img[y1:y1+sz, x1:x1+sz]

            # → BGR→RGB, resize, float, preprocess, batch
            proc = cv2.resize(sq, (100, 100))
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32)
            # proc = _PREPROCESS_FN(proc)
            proc = np.expand_dims(proc, 0)

            # predict
            pred = model.predict(proc, verbose=0)[0]
            idx = int(np.argmax(pred))
            piece = class_names[idx]
            position[square_map[(row, col)]] = piece

            # annotate non‐empty
            if piece not in ("unknown", "error"):
                cx, cy = x1 + sz//2, y1 + sz//2
                bri = np.mean(sq)
                colr = (0,0,0) if bri > 128 else (255,255,255)
                text = get_short_piece_notation(piece)
                cv2.putText(visual, text, (cx-10, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colr, 2)

            if debug:
                cv2.imwrite(f"squares/{row}_{col}.png", sq)

    # 5) build FEN and return
    fen = position_to_fen(position, orientation)
    return position, visual, fen



def decode_prediction(prediction):
    """
    Convert model prediction to a piece type string.
    Adapted for the model's output format which uses p_w (pawn white), p_b (pawn black), etc.
    """
    # If model outputs probabilities for each class
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 2:  # Batch of predictions (most common case)
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
        else:  # Single array of probabilities
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

        print(f"Predicted class index: {class_idx}, confidence: {confidence:.2f}")

        # Define class names for the 14-class model
        class_names = [
            'b_b', 'b_w', 'empty', 'empty_dot',
            'k_b', 'k_w', 'n_b', 'n_w',
            'p_b', 'p_w', 'q_b', 'q_w',
            'r_b', 'r_w', 'unknown'  # The 14th class
        ]

        # Make sure class_idx is within range of class_names
        if class_idx < len(class_names):
            return class_names[class_idx]
        else:
            print(f"Warning: Class index {class_idx} is out of range for class_names")
            return "unknown"

    # For other output formats, add appropriate handling
    return "unknown"


def get_short_piece_notation(piece_type):
    """Convert p_w/p_b style piece type to short notation for visualization."""
    # Map piece types to short notations for display
    piece_map = {
        'p_w': 'wP', 'n_w': 'wN', 'b_w': 'wB', 'r_w': 'wR', 'q_w': 'wQ', 'k_w': 'wK',
        'p_b': 'bP', 'n_b': 'bN', 'b_b': 'bB', 'r_b': 'bR', 'q_b': 'bQ', 'k_b': 'bK',
        'empty': '', 'empty_dot': '',
    }

    return piece_map.get(piece_type, piece_type)


def position_to_fen(position, orientation=0):  # orientation parameter kept for API compatibility
    """
    Convert position dictionary to FEN (Forsyth-Edwards Notation).
    Adapted for p_w/p_b style piece naming.
    """
    # Map piece types to FEN characters
    fen_map = {
        'p_w': 'P', 'n_w': 'N', 'b_w': 'B', 'r_w': 'R', 'q_w': 'Q', 'k_w': 'K',
        'p_b': 'p', 'n_b': 'n', 'b_b': 'b', 'r_b': 'r', 'q_b': 'q', 'k_b': 'k',
        'empty': '1', 'error': '1', 'unknown': '1', 'empty_dot': '1'
    }

    # Generate a board representation
    board = []
    for rank in range(8, 0, -1):  # FEN starts at rank 8
        rank_str = ''
        empty_count = 0

        for file in 'abcdefgh':
            square = file + str(rank)
            piece = position.get(square, 'empty')

            if piece in ['empty', 'error', 'unknown']:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += fen_map.get(piece, '1')

        if empty_count > 0:
            rank_str += str(empty_count)

        board.append(rank_str)

    # Join ranks with slashes
    fen = '/'.join(board)

    # Add other FEN components (to make it a valid FEN)
    # For now, just assume it's white to move, both sides can castle, no en passant, etc.
    fen += " w KQkq - 0 1"

    return fen


def parse_srt_file(srt_path):
    """
    Parse an SRT file and extract timestamps and text.

    Args:
        srt_path: Path to the SRT file

    Returns:
        List of dictionaries with 'start_time', 'end_time', and 'text' keys
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split the content into blocks
    blocks = re.split(r"\n\s*\n", content)
    transcripts = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Extract timestamp line
        timestamp_line = lines[1]
        timestamps = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
        if not timestamps:
            continue

        start_time = timestamps.group(1)
        end_time = timestamps.group(2)

        # Extract text (could be multiple lines)
        text = ' '.join(lines[2:])

        # Convert timestamp to a format matching the image filenames
        start_time_for_img = start_time.replace(':', '-').replace(',', '.')

        transcripts.append({
            'start_time': start_time,
            'end_time': end_time,
            'start_time_for_img': start_time_for_img,
            'text': text
        })

    return transcripts


def process_image(img_path, model_path, debug=False):
    """
    Process a single image and extract FEN position.

    Args:
        img_path: Path to the image
        model_path: Path to the chess piece CNN model
        debug: Whether to output debug information

    Returns:
        FEN position or None if processing fails
    """
    # Load and process the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}")
        return None

    # Downscale for faster processing
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Detect the board corners
    pts = extract_board_region(img, k=4, debug=debug)
    if pts is None:
        print(f"Failed to find board region in {img_path}")
        return None

    # Warp the board to a square
    _, warped, _ = warp_and_draw(img, pts)  # Ignore overlay and transform_matrix

    # Determine board orientation
    orientation = determine_orientation(warped, debug=debug)

    # Identify pieces and get FEN
    try:
        _, _, fen = analyze_chess_position(warped, model_path, orientation, debug)  # Ignore position and visual
        return fen
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def process_frames_directory(frames_dir, model_path, debug=False, use_gpu=None, cpu_only=False):
    """
    Process all frames in a directory and extract FEN positions.

    Args:
        frames_dir: Directory containing the frames
        model_path: Path to the chess piece CNN model
        debug: Whether to output debug information
        use_gpu: Override to force GPU usage (True) or CPU usage (False)
        cpu_only: Force CPU-only mode for all processing

    Returns:
        Dictionary mapping frame filenames to FEN positions
    """
    # Determine whether to use GPU or CPU
    should_use_gpu = USE_GPU if use_gpu is None else use_gpu
    if cpu_only:
        should_use_gpu = False

    # Load the model (we don't need to store it as we pass model_path directly to analyze_chess_position)
    custom_objects = {
        'Normalizer': Normalizer,
        'preprocess_input': _PREPROCESS_FN
    }
    tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )

    # Get all image files in the directory
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in {frames_dir}")
        return {}

    # Sort image files to ensure consistent processing order
    image_files = sorted(image_files)

    # Process images in parallel using multiple CPU cores
    fen_positions = {}

    # If using GPU, process sequentially to avoid CUDA context issues
    # If using CPU, use parallel processing
    if should_use_gpu:
        print(f"Processing {len(image_files)} images sequentially using GPU...")

        # Process images in batches to show progress
        batch_size = 10
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}...")

            for img_file in batch:
                img_path = os.path.join(frames_dir, img_file)
                try:
                    fen = process_image(img_path, model_path, debug)
                    if fen:
                        fen_positions[img_file] = fen
                        print(f"FEN for {img_file}: {fen}")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    else:
        # CPU mode - use parallel processing
        num_workers = min(multiprocessing.cpu_count(), 12)  # Use up to 4 CPU cores
        print(f"Processing {len(image_files)} images in parallel using {num_workers} CPU workers...")

        # Process images in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}...")

            # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA context issues
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks for each image in the batch
                futures = {}
                for img_file in batch:
                    img_path = os.path.join(frames_dir, img_file)
                    futures[executor.submit(process_image, img_path, model_path, debug)] = img_file

                # Process results as they complete
                for future in as_completed(futures):
                    img_file = futures[future]
                    try:
                        fen = future.result()
                        if fen:
                            fen_positions[img_file] = fen
                            print(f"FEN for {img_file}: {fen}")
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")

    print(f"Processed {len(fen_positions)}/{len(image_files)} images successfully")
    return fen_positions


def ensure_proper_spacing(text1, text2):
    """
    Ensure proper spacing when combining two text segments.

    Args:
        text1: First text segment
        text2: Second text segment

    Returns:
        Combined text with proper spacing
    """
    # Check if text1 ends with a space
    if not text1.endswith(' '):
        text1 += ' '

    # Check if text2 starts with a space
    if text2.startswith(' '):
        text2 = text2[1:]

    return text1 + text2


def combine_transcripts_by_fen(transcripts, fen_positions):
    """
    Combine transcript segments with the same FEN position.

    Args:
        transcripts: List of transcript dictionaries
        fen_positions: Dictionary mapping frame filenames to FEN positions

    Returns:
        List of dictionaries with 'fen' and 'description' keys
    """
    # Group transcripts by FEN position
    fen_to_transcripts = {}

    for transcript in transcripts:
        img_file = transcript['start_time_for_img'] + '.png'
        if img_file not in fen_positions:
            print(f"Warning: No FEN position found for {img_file}")
            continue

        fen = fen_positions[img_file]
        if fen not in fen_to_transcripts:
            fen_to_transcripts[fen] = []

        fen_to_transcripts[fen].append(transcript)

    # Combine transcripts with the same FEN
    combined_data = []
    for fen, fen_transcripts in fen_to_transcripts.items():
        # Sort transcripts by start time
        fen_transcripts.sort(key=lambda x: x['start_time'])

        # Combine text
        combined_text = fen_transcripts[0]['text']
        for i in range(1, len(fen_transcripts)):
            combined_text = ensure_proper_spacing(combined_text, fen_transcripts[i]['text'])

        combined_data.append({
            'fen': fen,
            'description': combined_text
        })

    return combined_data


def save_to_json(combined_data, output_path):
    """
    Save the combined data to a JSON file.

    Args:
        combined_data: List of dictionaries with 'fen' and 'description' keys
        output_path: Path to save the JSON file

    Returns:
        Path to the saved JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Saved combined data to {output_path}")
    return output_path


def find_video_directories(parent_dir):
    """
    Find all video directories with their corresponding SRT files and image directories.

    Args:
        parent_dir: Parent directory containing video folders

    Returns:
        List of dictionaries with 'video_dir', 'srt_file', and 'imgs_dir' keys
    """
    video_dirs = []

    # Walk through the parent directory
    for root, _, files in os.walk(parent_dir):  # Ignore dirs
        # Look for SRT files
        srt_files = [f for f in files if f.lower().endswith('.srt')]

        # If there are SRT files in this directory
        for srt_file in srt_files:
            srt_path = os.path.join(root, srt_file)

            # Check if there's an 'imgs' directory
            imgs_dir = os.path.join(root, 'imgs')
            if os.path.isdir(imgs_dir):
                # Check if the imgs directory has image files
                img_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    video_dirs.append({
                        'video_dir': root,
                        'srt_file': srt_path,
                        'imgs_dir': imgs_dir
                    })

    return video_dirs


def process_single_video(video_info, model_path, debug=False, cpu_only=True):
    """
    Process a single video directory in parallel.

    Args:
        video_info: Dictionary with 'video_dir', 'srt_file', and 'imgs_dir' keys
        model_path: Path to the chess piece CNN model
        debug: Whether to output debug information
        cpu_only: Force CPU-only mode for processing frames

    Returns:
        Dictionary with 'video_dir', 'output_path', and 'success' keys
    """
    start_time = time.time()
    video_dir = video_info['video_dir']
    srt_file = video_info['srt_file']
    imgs_dir = video_info['imgs_dir']

    # Set default output path
    video_dir_name = os.path.basename(os.path.normpath(video_dir))
    output_path = os.path.join(video_dir, f"{video_dir_name}_output.json")
    cache_path = os.path.join(video_dir, f"{video_dir_name}_fen_cache.pkl")

    print(f"Processing video: {video_dir_name}")
    print(f"  Images directory: {imgs_dir}")
    print(f"  SRT file: {srt_file}")
    print(f"  CPU-only mode: {cpu_only}")

    # Check if cache exists
    fen_positions = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                fen_positions = pickle.load(f)
            print(f"  Loaded {len(fen_positions)} cached FEN positions")
        except Exception as e:
            print(f"  Error loading cache: {e}")
            fen_positions = {}

    # If cache is empty or doesn't exist, process the frames
    if not fen_positions:
        print(f"  Processing video frames...")
        fen_positions = process_frames_directory(imgs_dir, model_path, debug, cpu_only=cpu_only)

        # Save cache for future use
        if fen_positions:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(fen_positions, f)
                print(f"  Cached {len(fen_positions)} FEN positions")
            except Exception as e:
                print(f"  Error saving cache: {e}")

    if not fen_positions:
        print(f"  No FEN positions extracted for {video_dir_name}. Skipping.")
        return {'video_dir': video_dir, 'output_path': output_path, 'success': False}

    # Parse the SRT file
    print(f"  Parsing SRT file...")
    transcripts = parse_srt_file(srt_file)

    if not transcripts:
        print(f"  No transcript segments extracted for {video_dir_name}. Skipping.")
        return {'video_dir': video_dir, 'output_path': output_path, 'success': False}

    # Combine transcripts by FEN position
    print(f"  Combining transcript segments by FEN position...")
    combined_data = combine_transcripts_by_fen(transcripts, fen_positions)

    # Save to JSON
    save_to_json(combined_data, output_path)

    elapsed_time = time.time() - start_time
    print(f"  Processing complete for {video_dir_name}. Output saved to {output_path}")
    print(f"  Time taken: {elapsed_time:.2f} seconds")

    return {'video_dir': video_dir, 'output_path': output_path, 'success': True}


def main():
    parser = argparse.ArgumentParser(
        description="Detect a chessboard, determine orientation, label squares, and identify pieces."
    )
    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--model', default='chess_piece_model.h5',
                        help='Path to chess piece CNN model')
    parser.add_argument('--clusters', type=int, default=4,
                        help='Number of color clusters for k-means')
    parser.add_argument('--downscale', type=int, default=800,
                        help='Max dimension for faster processing')
    parser.add_argument('--debug', action='store_true',
                        help='Show intermediate masks and figures')

    # Add new arguments for video processing
    parser.add_argument('--process-video', action='store_true',
                        help='Process a directory of video frames and combine transcript segments')
    parser.add_argument('--video-dir', help='Directory containing video frames')
    parser.add_argument('--srt-file', help='Path to the SRT transcript file')
    parser.add_argument('--output', help='Path to save the output JSON file')
    parser.add_argument('--parent-dir', help='Parent directory containing multiple video folders to process')
    parser.add_argument('--max-parallel', type=int,
                        help='Maximum number of videos to process in parallel (default: number of CPU cores)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only mode for processing (avoids CUDA/GPU errors)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Force GPU usage if available (default: auto-detect)')

    args = parser.parse_args()

    # Check if we're processing videos
    if args.process_video:
        # Check if model exists
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} does not exist")
            return

        # Check if we're processing a parent directory or a single video
        if args.parent_dir:
            if not os.path.exists(args.parent_dir):
                print(f"Error: Parent directory {args.parent_dir} does not exist")
                return

            # Find all video directories with SRT files and image directories
            print(f"Searching for videos in {args.parent_dir}...")
            video_dirs = find_video_directories(args.parent_dir)

            if not video_dirs:
                print(f"No video directories with SRT files and image directories found in {args.parent_dir}")
                return

            print(f"Found {len(video_dirs)} video directories to process")

            # Add option to limit the number of parallel videos
            max_parallel_videos = min(len(video_dirs), args.max_parallel or multiprocessing.cpu_count())
            print(f"Processing up to {max_parallel_videos} videos in parallel")

            # Print GPU status
            if USE_GPU:
                print(f"GPU detected and available")
            else:
                print(f"No GPU detected, using CPU only")

            # Check if user forced CPU-only mode
            if args.cpu_only:
                print(f"Forcing CPU-only mode as requested")
                use_gpu = False
            elif args.use_gpu and USE_GPU:
                print(f"Forcing GPU usage as requested")
                use_gpu = True
            else:
                use_gpu = USE_GPU

            # If using GPU, process videos sequentially to avoid CUDA context issues
            # If using CPU, process videos in parallel
            start_time = time.time()
            results = []

            if use_gpu and not args.cpu_only:
                print(f"Processing videos sequentially using GPU...")
                for video_info in video_dirs:
                    video_dir = video_info['video_dir']
                    video_dir_name = os.path.basename(os.path.normpath(video_dir))

                    try:
                        result = process_single_video(video_info, args.model, args.debug, cpu_only=False)
                        results.append(result)
                        if result['success']:
                            print(f"Successfully processed: {video_dir_name}")
                        else:
                            print(f"Failed to process: {video_dir_name}")
                    except Exception as e:
                        print(f"Error processing {video_dir_name}: {e}")
            else:
                # CPU mode - use parallel processing
                print(f"Processing videos in parallel using CPU only...")
                # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA context issues
                with ThreadPoolExecutor(max_workers=max_parallel_videos) as executor:
                    # Submit tasks for each video
                    futures = {
                        executor.submit(process_single_video, video_info, args.model, args.debug, cpu_only=True): video_info
                        for video_info in video_dirs
                    }

                    # Process results as they complete
                    for future in as_completed(futures):
                        video_info = futures[future]
                        video_dir = video_info['video_dir']
                        video_dir_name = os.path.basename(os.path.normpath(video_dir))

                        try:
                            result = future.result()
                            results.append(result)
                            if result['success']:
                                print(f"Successfully processed: {video_dir_name}")
                            else:
                                print(f"Failed to process: {video_dir_name}")
                        except Exception as e:
                            print(f"Error processing {video_dir_name}: {e}")

            # Print summary
            total_time = time.time() - start_time
            successful = sum(1 for r in results if r['success'])
            print(f"\nProcessing complete: {successful}/{len(video_dirs)} videos processed successfully")
            print(f"Total time: {total_time:.2f} seconds")
            return

        # Process a single video directory
        elif args.video_dir and args.srt_file:
            if not os.path.exists(args.video_dir):
                print(f"Error: Video directory {args.video_dir} does not exist")
                return

            if not os.path.exists(args.srt_file):
                print(f"Error: SRT file {args.srt_file} does not exist")
                return

            # Set default output path if not provided
            output_path = args.output
            if not output_path:
                video_dir_name = os.path.basename(os.path.normpath(args.video_dir))
                output_path = os.path.join(os.path.dirname(args.video_dir), f"{video_dir_name}_output.json")

            # Print GPU status
            if USE_GPU:
                print(f"GPU detected and available")
            else:
                print(f"No GPU detected, using CPU only")

            # Check if user forced CPU-only mode
            if args.cpu_only:
                print(f"Forcing CPU-only mode as requested")
                use_cpu_only = True
            elif args.use_gpu and USE_GPU:
                print(f"Forcing GPU usage as requested")
                use_cpu_only = False
            else:
                use_cpu_only = not USE_GPU

            # Process the video frames
            print(f"Processing video frames in {args.video_dir}...")
            fen_positions = process_frames_directory(args.video_dir, args.model, args.debug, cpu_only=use_cpu_only)

            if not fen_positions:
                print("No FEN positions extracted. Exiting.")
                return

            # Parse the SRT file
            print(f"Parsing SRT file {args.srt_file}...")
            transcripts = parse_srt_file(args.srt_file)

            if not transcripts:
                print("No transcript segments extracted. Exiting.")
                return

            # Combine transcripts by FEN position
            print("Combining transcript segments by FEN position...")
            combined_data = combine_transcripts_by_fen(transcripts, fen_positions)

            # Save to JSON
            save_to_json(combined_data, output_path)

            print(f"Processing complete. Output saved to {output_path}")
            return

        else:
            print("Error: Either --parent-dir or both --video-dir and --srt-file are required when using --process-video")
            return

    # Original functionality for processing a single image
    if not args.image:
        print("Error: image path is required when not using --process-video")
        return

    # Load and optionally downscale
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image {args.image}")
        return
    h, w = img.shape[:2]
    scale = args.downscale / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    # 1) detect the board corners
    pts = extract_board_region(img, k=args.clusters, debug=args.debug)
    if pts is None:
        print("Failed to find board region via color clustering.")
        return

    # 2) warp the board to a square and draw grid
    overlay, warped, _ = warp_and_draw(img, pts)  # Ignore transform_matrix

    # 3) determine board orientation
    orientation = determine_orientation(warped, debug=args.debug)
    orientation_names = ["Standard", "Rotated 90°", "Rotated 180°", "Rotated 270°"]
    print(f"Detected orientation: {orientation_names[orientation]}")

    # 4) label squares with chess notation
    labeled_warped = label_squares(warped, orientation)

    # 5) identify pieces on each square
    if os.path.exists(args.model):
        position, pieces_visual, fen = analyze_chess_position(warped, args.model, orientation, args.debug)

        # Save results with pieces
        cv2.imwrite('board_with_pieces.png', pieces_visual)

        if position is not None:
            # Print the detected pieces
            print("\nDetected pieces:")
            empty_count = 0
            for notation, piece in sorted(position.items()):
                if piece not in ['empty', 'error', 'unknown', 'empty_dot']:
                    print(f"{notation}: {piece}")
                else:
                    empty_count += 1
            print(f"Empty squares: {empty_count}")

            # Print FEN
            print(f"\nFEN notation: {fen}")
    else:
        print(f"Model file {args.model} not found. Skipping piece detection.")

    # Save results
    cv2.imwrite('board_grid.png', overlay)
    cv2.imwrite('board_warped.png', warped)
    cv2.imwrite('board_labeled.png', labeled_warped)

    # Create mapping
    square_map = get_square_mapping(orientation)
    print("\nSquare mapping (row, col) -> notation:")
    for row in range(8):
        for col in range(8):
            print(f"({row}, {col}) -> {square_map[(row, col)]}", end="\t")
        print()

    print('\nSaved board_grid.png, board_warped.png, board_labeled.png')
    if os.path.exists(args.model):
        print('and board_with_pieces.png')

    # Simple test - detect a position from pixel
    test_x, test_y = warped.shape[1] // 2, warped.shape[0] // 2  # center of image
    notation, coords = detect_piece_position(warped, test_x, test_y)
    print(f"\nCenter square is at {test_x}, {test_y} -> {notation} (coordinates: {coords})")


if __name__ == '__main__':
    main()