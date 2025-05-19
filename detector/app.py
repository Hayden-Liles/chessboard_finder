"""
Digital chessboard detection with orientation recognition, square labeling, and piece identification.

This script detects a chess board in an image, determines its orientation,
labels all squares with proper chess notation (a1-h8), and identifies chess pieces
using a CNN model.

Steps:
1. Detect the board using color segmentation
2. Determine orientation by analyzing corner square colors
3. Label all 64 squares with proper chess notation
4. Identify the chess pieces on each square using a CNN model
5. Output annotated images showing the detected board and pieces

Usage:
    python chess_board_detector.py path/to/image.png --model chess_piece_model.h5 --clusters 4 --downscale 800 --debug
"""
import cv2
import numpy as np
import argparse
from itertools import combinations
import tensorflow as tf
import os


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
        (cx,cy),(bw,bh),angle = rect
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
    h, w = warped_img.shape[:2]
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
    h, w = warped_img.shape[:2]
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
    Analyze the chess position by identifying pieces on each square.
    
    Args:
        warped_img: Warped chessboard image
        model_path: Path to the chess piece CNN model
        orientation: Board orientation (0-3)
        debug: Whether to save debug images
        
    Returns:
        position: Dictionary mapping square notation to piece type
        visual: Annotated image showing pieces detected
        fen: Forsyth-Edwards Notation of the position
    """
    # Define the custom Normalizer layer for loading the model
    class Normalizer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Normalizer, self).__init__(**kwargs)
        
        def call(self, inputs):
            return inputs / 255.0
        
        def get_config(self):
            config = super(Normalizer, self).get_config()
            return config
    
    # Try to load the model with custom layer
    try:
        with tf.keras.utils.custom_object_scope({'Normalizer': Normalizer}):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Try with Lambda layer as fallback
        try:
            custom_objects = {
                'Normalizer': tf.keras.layers.Lambda(lambda x: x / 255.0)
            }
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Successfully loaded model using Lambda layer fallback")
        except Exception as e2:
            print(f"Both loading methods failed: {e2}")
            return None, warped_img, None
    
    # Process the warped board image
    h, w = warped_img.shape[:2]
    square_size = h // 8
    
    # Create a copy of the image for visualization
    visual = warped_img.copy()
    
    # Get the square mapping based on orientation
    square_map = get_square_mapping(orientation)
    
    # Create a dictionary to store the position
    position = {}
    
    # Create a directory to save individual square images if debugging
    if debug:
        os.makedirs('squares', exist_ok=True)
    
    # Process each square
    for row in range(8):
        for col in range(8):
            # Extract the square
            y1, y2 = row * square_size, (row + 1) * square_size
            x1, x2 = col * square_size, (col + 1) * square_size
            square_img = warped_img[y1:y2, x1:x2]
            
            # Save the square image for debugging
            if debug:
                notation = square_map[(row, col)]
                cv2.imwrite(f'squares/square_{notation}.png', square_img)
            
            # Preprocess for the model
            # Resize to expected input size (100x100)
            processed_img = cv2.resize(square_img, (100, 100))
            
            # Handle grayscale images (convert to RGB)
            if len(processed_img.shape) == 2:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            elif processed_img.shape[2] == 1:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            
            # Add batch dimension (no need to normalize as it's handled by model)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Predict the piece
            try:
                prediction = model.predict(processed_img, verbose=0)
                piece_type = decode_prediction(prediction)
            except Exception as e:
                print(f"Error predicting square at {row}, {col}: {e}")
                piece_type = 'error'
            
            # Get square notation
            notation = square_map[(row, col)]
            
            # Store the result
            position[notation] = piece_type
            
            # Annotate the visual with piece type
            if piece_type not in ['empty', 'error', 'unknown']:
                center_x = int((col + 0.5) * square_size)
                center_y = int((row + 0.5) * square_size)
                
                # Choose text color for visibility
                avg_brightness = np.mean(square_img)
                text_color = (0, 0, 0) if avg_brightness > 128 else (255, 255, 255)
                
                # Use a shorter piece notation for visual clarity
                short_notation = get_short_piece_notation(piece_type)
                cv2.putText(visual, short_notation, (center_x-10, center_y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Generate FEN representation
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

        # Define class names based on typical alphabetical sorting of folders
        # You may need to adjust this based on your actual class indices
        class_names = [
            'b_b', 'b_w', 'empty', 
            'k_b', 'k_w', 'n_b', 'n_w',
            'p_b', 'p_w', 'q_b', 'q_w',
            'r_b', 'r_w'
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
        'empty': '',
    }
    
    return piece_map.get(piece_type, piece_type)


def position_to_fen(position, orientation=0):
    """
    Convert position dictionary to FEN (Forsyth-Edwards Notation).
    Adapted for p_w/p_b style piece naming.
    """
    # Map piece types to FEN characters
    fen_map = {
        'p_w': 'P', 'n_w': 'N', 'b_w': 'B', 'r_w': 'R', 'q_w': 'Q', 'k_w': 'K',
        'p_b': 'p', 'n_b': 'n', 'b_b': 'b', 'r_b': 'r', 'q_b': 'q', 'k_b': 'k',
        'empty': '1', 'error': '1', 'unknown': '1'
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


def main():
    parser = argparse.ArgumentParser(
        description="Detect a chessboard, determine orientation, label squares, and identify pieces."
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', default='chess_piece_model.h5',
                        help='Path to chess piece CNN model')
    parser.add_argument('--clusters', type=int, default=4,
                        help='Number of color clusters for k-means')
    parser.add_argument('--downscale', type=int, default=800,
                        help='Max dimension for faster processing')
    parser.add_argument('--debug', action='store_true',
                        help='Show intermediate masks and figures')
    args = parser.parse_args()

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
    overlay, warped, transform_matrix = warp_and_draw(img, pts)

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
                if piece not in ['empty', 'error', 'unknown']:
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