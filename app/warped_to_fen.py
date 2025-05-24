import os
# Set CUDA_VISIBLE_DEVICES to an empty string or "-1" to force CPU usage
# This must be done BEFORE importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
import json
from itertools import combinations # For extract_board_region_optimized
import multiprocessing

# Global model cache to avoid reloading models
MODEL_CACHE = {}

class Normalizer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / 255.0

    def get_config(self):
        return super().get_config()

# We'll refer to this function by name when loading:
_PREPROCESS_FN = tf.keras.applications.mobilenet_v2.preprocess_input

def load_model_once(model_path):
    """Load the model once and cache it for future use"""
    global MODEL_CACHE
    
    if model_path not in MODEL_CACHE:
        # Define custom objects needed for model loading
        custom_objects = {
            'Normalizer': Normalizer,
            'preprocess_input': _PREPROCESS_FN
        }
        
        # Load the model
        try:
            MODEL_CACHE[model_path] = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print(f"Model loaded successfully from warper_to_fen: {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path} in warper_to_fen: {e}")
            raise
    
    return MODEL_CACHE[model_path]

# --- Board Detection and Warping Functions (adapted from main.py / app2.py) ---
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

def extract_board_region_optimized(img,
                                 k=4,
                                 kmeans_sample_size=5000,
                                 kmeans_iterations=30,
                                 morph_kernel_size=15,
                                 debug=False,
                                 debug_path_prefix=None):
    """
    Optimized version of extract_board_region that preserves original image quality.
    Focuses on optimizing the algorithm, not reducing image quality.
    """
    h, w = img.shape[:2]
    
    img_small = img
    h_small, w_small = h, w
    
    current_kmeans_sample_size = min(kmeans_sample_size, h_small*w_small)
    if h_small * w_small == 0:
        print("Warning (warper_to_fen): Image dimensions are zero in extract_board_region_optimized.")
        return None
    if current_kmeans_sample_size == 0 and h_small * w_small > 0:
        current_kmeans_sample_size = 1

    if current_kmeans_sample_size > 0:
        coords = np.random.choice(h_small*w_small, current_kmeans_sample_size, replace=False)
        pixels = img_small.reshape(-1,3)[coords].astype(np.float32)
    else:
        pixels = img_small.reshape(-1,3).astype(np.float32)
        if not pixels.size: # Check if pixels array is empty
            return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    if len(pixels) < k:
        print(f"Warning (warper_to_fen): Not enough pixel samples ({len(pixels)}) for k-means with k={k}.")
        return None

    _, _, centers = cv2.kmeans(pixels, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_iterations, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    
    stride = 2
    y_indices, x_indices = np.mgrid[0:h_small:stride, 0:w_small:stride]
    pixel_indices = y_indices.flatten() * w_small + x_indices.flatten()
    
    sampled_pixels = img_small.reshape(-1, 3)[pixel_indices].astype(np.float32)
    
    dists = np.linalg.norm(sampled_pixels[:,None] - centers[None,:], axis=2)
    sampled_labels = np.argmin(dists, axis=1)
    
    grid_labels = sampled_labels.reshape(y_indices.shape)
    
    full_labels = np.zeros((h_small, w_small), dtype=np.int32)
    for y_idx_grid, row_val in enumerate(y_indices[:,0]):
        for x_idx_grid, col_val in enumerate(x_indices[0,:]):
            y_start, y_end = row_val, min(row_val + stride, h_small)
            x_start, x_end = col_val, min(col_val + stride, w_small)
            full_labels[y_start:y_end, x_start:x_end] = grid_labels[y_idx_grid, x_idx_grid]
    
    best_score = float('inf')
    best_box = None

    for i_cluster, j_cluster in combinations(range(k), 2):
        mask = np.zeros((h_small, w_small), np.uint8)
        mask[np.logical_or(full_labels==i_cluster, full_labels==j_cluster)] = 255

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        if area < 0.01 * (h_small*w_small): continue

        rect = cv2.minAreaRect(cnt)
        _,(bw,bh),_ = rect
        if min(bw,bh) == 0: continue
        aspect = max(bw,bh)/min(bw,bh)
        score = abs(1.0 - aspect)

        ar = area / float(h_small*w_small)
        if ar > 0.9 or ar < 0.02: score += 1.0

        if score < best_score:
            best_score = score
            best_box = cv2.boxPoints(rect).astype(np.float32)
            if debug and debug_path_prefix:
                debug_mask_img = img.copy()
                cv2.drawContours(debug_mask_img, [cnt.astype(int)], -1, (0, 255, 0), 2)
                cv2.imwrite(f'{debug_path_prefix}mask_clusters_{i_cluster}_{j_cluster}.png', debug_mask_img)
    return best_box


def warp_and_draw(img, pts, size=800, thickness=2):
    """Warp the board to a square and optionally overlay an 8x8 grid."""
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (size, size))

    # Create overlay for the grid on the original image
    # overlay_on_original = np.zeros_like(img) # Grid lines will be drawn on this
    # grid_color = (0,255,0) # Green grid lines
    # step = size // 8
    # for i in range(9): # Draw 9 lines for 8 squares
    #     # Create points for grid lines on the warped image
    #     pt1_h_warped = (i*step,0)
    #     pt2_h_warped = (i*step,size-1)
    #     pt1_v_warped = (0,i*step)
    #     pt2_v_warped = (size-1,i*step)
        
    #     # Draw on warped image (for visualization if needed)
    #     # cv2.line(warp, pt1_h_warped, pt2_h_warped, grid_color, thickness)
    #     # cv2.line(warp, pt1_v_warped, pt2_v_warped, grid_color, thickness)

    # We will return the warped image and the transform matrix.
    return warp, M

def find_and_warp_chessboard(image_path: str,
                             output_dir: str = None,
                             debug: bool = False,
                             kmeans_sample_size: int = 5000,
                             kmeans_iterations: int = 30,
                             morph_kernel_size: int = 15):
    """
    Loads an image, finds the chessboard, and warps it.
    Returns the warped board image (NumPy array) or None if not found.
    Saves other debug images (like corners, k-means masks) to output_dir 
    if debug is True and output_dir is provided.
    """
    if debug and output_dir: # Ensure output_dir is created only if debug is True and output_dir is provided
        os.makedirs(output_dir, exist_ok=True)
    elif debug and not output_dir:
        print("Warning (detector): Debug is True, but no output_dir provided. Detector debug images will not be saved.")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error (detector): Could not load image from {image_path}")
        return None
    
    # print(f"Processing {image_path}...") # Already printed by _process_single_image_to_fen or get_fen_from_image_or_dir
    
    h_orig, w_orig = img.shape[:2]
    scale_cap = 800 
    scale = 1.0
    if max(h_orig, w_orig) > scale_cap:
        scale = scale_cap / max(h_orig, w_orig)
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # print(f"Image scaled down by factor of {scale:.2f}") # Can be verbose
    else:
        img_scaled = img.copy()

    debug_prefix = None
    if debug and output_dir:
        debug_prefix = os.path.join(output_dir, os.path.basename(image_path) + "_debug_")
        
    pts = extract_board_region_optimized(img_scaled, k=4,
                                         kmeans_sample_size=kmeans_sample_size,
                                         kmeans_iterations=kmeans_iterations,
                                         morph_kernel_size=morph_kernel_size,
                                         debug=debug,
                                         debug_path_prefix=debug_prefix)

    if pts is not None:
        pts_original_scale = pts / scale
        warped_board, _ = warp_and_draw(img, pts_original_scale, size=800)
        return warped_board
    else:
        # print(f"Could not find chessboard in the image: {image_path}") # find_and_warp_chessboard in _process_single_image_to_fen handles this
        return None

def get_square_color(rank_idx: int, file_idx: int) -> str:
    """
    Determines if a square (rank_idx, file_idx) is light or dark.
    Assumes standard chessboard pattern where a1 (0,0 from white's POV) is dark.
    rank_idx, file_idx are 0-7 (image coordinates, not chess rank numbers).
    """
    # (0,0) is a1 (dark), (0,1) is b1 (light), (1,0) is a2 (light)
    # If (rank_idx + file_idx) is even, it's a dark square. If odd, it's light.
    return "light" if (rank_idx + file_idx) % 2 != 0 else "dark"

def determine_orientation_from_corners(warped_img, debug=False, debug_image_path=None):
    """
    Determine board orientation SOLELY by analyzing corner square colors.
    This is a fallback or initial guess. Returns an orientation code: 0-3.
    """
    h, w = warped_img.shape[:2]
    square_size = h // 8
    offset = square_size // 4

    # Corner indices in order: Top-left, Top-right, Bottom-right, Bottom-left
    corners = [
        (offset, offset),                   # Top-left (a8 in standard orientation)
        (w - offset, offset),               # Top-right (h8 in standard orientation)
        (w - offset, h - offset),           # Bottom-right (h1 in standard orientation)
        (offset, h - offset)                # Bottom-left (a1 in standard orientation)
    ]


    corner_squares = []
    for x, y in corners:
        samples = []
        for dx in range(-offset//2, offset//2+1, offset//4):
            for dy in range(-offset//2, offset//2+1, offset//4):
                samples.append(warped_img[int(y+dy), int(x+dx)])
        corner_squares.append(np.mean(samples, axis=0))

    brightness = [np.mean(color) for color in corner_squares]
    sorted_bright = sorted(brightness)
    if len(sorted_bright) < 4: # Should not happen with 4 corners
        print("Warning (determine_orientation_from_corners): Not enough brightness values. Defaulting to 0.")
        return 0
    threshold = (sorted_bright[1] + sorted_bright[2]) / 2
    is_white = [b > threshold for b in brightness]

    diagonals_match = (is_white[0] == is_white[2]) and (is_white[1] == is_white[3])
    if diagonals_match:
        # If BR (H1 in std) is white
        if is_white[2]: 
            # if BL (A1 in std) is black -> standard
            if not is_white[3]: return 0 
            # else (BL is white) -> 270 deg (A1 is BR (black), H1 is BL (white))
            else: return 3 
        # Else BR is black
        else: 
            # if BL is white -> 90 deg (A1 is TL (black), H1 is TR (white))
            if is_white[3]: return 1 
            # else (BL is black) -> 180 deg (A1 is TR (black), H1 is BL (white))
            else: return 2 

    print("Warning (determine_orientation_from_corners): Corner color pattern unclear. Defaulting to 0.")
    return 0

def get_rank_file_from_square_notation(square_notation):
    """Converts 'a1' to (0,0), 'h8' to (7,7) etc. (rank_idx, file_idx)"""
    if not isinstance(square_notation, str) or len(square_notation) != 2:
        return None
    file_char = square_notation[0]
    rank_char = square_notation[1]
    if file_char not in 'abcdefgh' or rank_char not in '12345678':
        return None
    file_idx = 'abcdefgh'.index(file_char)
    rank_idx = '12345678'.index(rank_char) # This gives rank 0 for '1', 7 for '8'
    # We want rank 0 for white's 1st rank (bottom), rank 7 for white's 8th rank (top)
    # So, if perspective is White's POV, rank_idx from '1' should be 0.
    # The '12345678'.index(rank_char) already does this if we consider rank 0 as the '1' rank.
    # However, visually, row 0 is the 8th rank, row 7 is the 1st rank.
    # Let's return (visual_row_idx_for_white_pov, visual_col_idx_for_white_pov)
    # a1 -> (7,0), h8 -> (0,7)
    return (7 - rank_idx, file_idx)


def determine_refined_orientation(warped_img, position_dict_from_initial_orientation, initial_orientation_code=0, debug=False, debug_image_path=None):
    """
    Refines board orientation using piece positions.
    `position_dict_from_initial_orientation` is the piece dictionary assuming `initial_orientation_code`.
    Returns a refined orientation code (0-3), where 0 is White's standard POV.
    """
    
    # 1. Fallback to corner-based detection if no pieces
    if not any(p not in ['empty', 'empty_dot', 'unknown', 'error'] for p in position_dict_from_initial_orientation.values()):
        print("Info (determine_refined_orientation): No pieces found. Using corner color based orientation.")
        return determine_orientation_from_corners(warped_img, debug, debug_image_path)

    # Helper to transform square notation if initial_orientation_code was not 0
    # This function will map a square from the perspective of initial_orientation_code
    # back to what its notation would be if the board was truly at 0-degree (White's POV).
    # This is complex because get_square_mapping already applies the initial_orientation_code.
    # Instead, we'll analyze ranks based on the *given* position_dict and its inherent orientation.

    # --- Pawn Analysis ---
    white_pawn_ranks = [] # Ranks from White's perspective (0-7, where 0 is rank '1', 1 is rank '2')
    black_pawn_ranks = []

    for square, piece in position_dict_from_initial_orientation.items():
        # `square` is already in algebraic notation based on `initial_orientation_code`
        # We need to convert this algebraic notation to a consistent rank index (0-7 for White's 1st to 8th rank)
        # Example: if initial_orientation_code = 0, 'a2' -> rank 1. if initial_orientation_code = 2 (180deg), 'a7' (which is white's a2) -> rank 1.
        
        # Let's get the (row, col) assuming the board IS at initial_orientation_code
        # And then determine the "true white rank" based on that.
        # This is tricky. Simpler: analyze based on the given notation directly.
        # If initial_orientation_code = 0, 'e2' is white pawn on rank 1 (0-indexed).
        # If initial_orientation_code = 2 (180 deg), white pawns are on 'e7' (rank 6).
        
        rank_char = square[1]
        if rank_char not in '12345678': continue
        
        # Rank index from White's actual 1st rank (rank '1' = 0, rank '8' = 7)
        # This interpretation depends on the true orientation.
        # Let's use the numeric part of the square notation directly for now.
        # If board is White's POV (true_orientation=0), white pawns are on ranks '2'-'4', black on '5'-'7'.
        # If board is Black's POV (true_orientation=2), white pawns on '7'-'5', black on '4'-'2'.

        numeric_rank = int(rank_char) # 1 to 8

        if piece == 'p_w':
            white_pawn_ranks.append(numeric_rank)
        elif piece == 'p_b':
            black_pawn_ranks.append(numeric_rank)

    if white_pawn_ranks or black_pawn_ranks:
        avg_white_pawn_rank = np.mean(white_pawn_ranks) if white_pawn_ranks else 7 # Default far if no white pawns
        avg_black_pawn_rank = np.mean(black_pawn_ranks) if black_pawn_ranks else 2 # Default far if no black pawns

        # If current view (initial_orientation_code) has white pawns on low ranks and black on high
        if avg_white_pawn_rank < 4.5 and avg_black_pawn_rank > 4.5:
            # This suggests the initial_orientation_code correctly represents White's POV
            return initial_orientation_code # Which should be 0 if we started with default
        # If current view has white pawns on high ranks and black on low
        elif avg_white_pawn_rank > 4.5 and avg_black_pawn_rank < 4.5:
            # This suggests the initial_orientation_code represents Black's POV
            # If initial was 0, new is 2 (180). If initial was 2, new is 0.
            # This logic needs to be robust if initial_orientation_code wasn't 0.
            # For now, assuming we call this after an analysis with initial_orientation_code=0:
            if initial_orientation_code == 0: return 2 # Rotate 180
            if initial_orientation_code == 2: return 0 # Already rotated, so it's correct
            # Other initial orientations are more complex to map directly here without full transform.
            # Let's simplify: if this function is called, initial_orientation_code is the current "best guess".
            # If pawns say it's black's POV, we need to flip it by 180 deg from initial_orientation_code.
            return (initial_orientation_code + 2) % 4

    # --- Queen Color Analysis (if pawn analysis inconclusive or no pawns) ---
    # This is more complex as it requires knowing the "true" square color if White is at bottom.
    # For now, we'll rely on the pawn analysis or the initial corner guess.
    # A full implementation would:
    # 1. Find wQ. Get its square (e.g., 'd1' from current position_dict).
    # 2. Convert 'd1' to (rank_idx, file_idx) assuming current orientation.
    # 3. Determine if that (rank_idx, file_idx) *should* be light if it were White's POV.
    # 4. Compare with actual color. If mismatch, adjust orientation.

    print("Info (determine_refined_orientation): Pawn analysis inconclusive or no pawns. Falling back to corner-based orientation.")
    # Fallback to corner-based if other methods are not definitive
    # Or, if initial_orientation_code was already non-zero and pawn analysis confirmed it, stick with it.
    # If pawn analysis was the decider, it would have returned already.
    # So if we reach here, pawn analysis was not definitive.
    corner_based_orientation = determine_orientation_from_corners(warped_img, debug, debug_image_path)
    return corner_based_orientation


def get_square_mapping(orientation: int = 0) -> dict:
    """
    Create a mapping from chessboard coordinates (0-7, 0-7) to chess notation.
    """
    mapping = {}
    files = "abcdefgh"
    ranks = "12345678"
    for row in range(8):
        for col in range(8):
            if orientation == 0: notation = files[col] + ranks[7-row]
            elif orientation == 1: notation = files[7-row] + ranks[7-col]
            elif orientation == 2: notation = files[7-col] + ranks[row]
            elif orientation == 3: notation = files[row] + ranks[col]
            else: notation = files[col] + ranks[7-row] # Default
            mapping[(row, col)] = notation
    return mapping

def get_short_piece_notation(piece_type: str) -> str:
    """Convert p_w/p_b style piece type to short notation for visualization."""
    piece_map = {
        'p_w': 'wP', 'n_w': 'wN', 'b_w': 'wB', 'r_w': 'wR', 'q_w': 'wQ', 'k_w': 'wK',
        'p_b': 'bP', 'n_b': 'bN', 'b_b': 'bB', 'r_b': 'bR', 'q_b': 'bQ', 'k_b': 'bK',
        'empty': '', 'empty_dot': '',
    }
    return piece_map.get(piece_type, piece_type)

def position_to_fen_improved(position: dict, previous_position: dict = None, move_context: dict = None) -> str:
    """
    Convert position dictionary to FEN with improved game state inference.
    
    Args:
        position: Current position dictionary
        previous_position: Previous position for turn inference (optional)
        move_context: Additional context like move number (optional)
    """
    fen_map = {
        'p_w': 'P', 'n_w': 'N', 'b_w': 'B', 'r_w': 'R', 'q_w': 'Q', 'k_w': 'K',
        'p_b': 'p', 'n_b': 'n', 'b_b': 'b', 'r_b': 'r', 'q_b': 'q', 'k_b': 'k',
        'empty': '1', 'error': '1', 'unknown': '1', 'empty_dot': '1'
    }
    
    # Build board FEN
    board_fen = []
    for rank_num in range(8, 0, -1):
        rank_str = ''
        empty_count = 0
        for file_char in 'abcdefgh':
            square = file_char + str(rank_num)
            piece = position.get(square, 'empty')
            if piece in ['empty', 'error', 'unknown', 'empty_dot']:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += fen_map.get(piece, '1')
        if empty_count > 0:
            rank_str += str(empty_count)
        board_fen.append(rank_str)
    
    board_fen_str = '/'.join(board_fen)
    
    # Analyze position context
    if move_context:
        context = move_context
    else:
        context = analyze_position_context(position)
    
    # Infer turn if we have previous position
    if previous_position is not None:
        context['turn'] = infer_turn_from_position_sequence(position, previous_position)
    
    # Build complete FEN
    fen = f"{board_fen_str} {context['turn']} {context['castling']} {context['en_passant']} {context['halfmove']} {context['fullmove']}"
    
    return fen

def _load_model_and_metadata(model_path: str) -> tuple[tf.keras.Model, list[str]]:
    """Loads the Keras model and its class name metadata."""
    model = load_model_once(model_path)
    meta_path = os.path.splitext(model_path)[0] + ".keras.metadata"
    class_names = []

    if not os.path.exists(meta_path):
        alt_meta_path = model_path.replace(".h5", ".keras.metadata") # Common alternative
        if os.path.exists(alt_meta_path):
            meta_path = alt_meta_path
        else:
            # Try one level up if model is in a subdirectory like 'models/'
            parent_dir_meta_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), os.path.basename(meta_path))
            if os.path.exists(parent_dir_meta_path):
                 meta_path = parent_dir_meta_path
            else:
                print(f"Warning (_load_model_and_metadata): Metadata file not found. Using default class names.")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            class_names = [meta["class_mapping"][str(i)] for i in range(len(meta["class_mapping"]))]
        except Exception as e:
            print(f"Warning (_load_model_and_metadata): Error loading metadata from {meta_path}: {e}. Using default class names.")
    
    if not class_names and model: # If metadata loading failed or file not found, use default
        class_names = [f'class_{i}' for i in range(model.output_shape[-1])]
    return model, class_names

def _extract_preprocess_and_predict_squares(warped_img: np.ndarray, model: tf.keras.Model, debug: bool = False, debug_squares_output_dir: str = None) -> tuple[np.ndarray, list, int]:
    """Extracts 64 squares, preprocesses them, and returns model predictions."""
    h, _ = warped_img.shape[:2]
    sz = h // 8
    if debug and debug_squares_output_dir:
        os.makedirs(debug_squares_output_dir, exist_ok=True)


    squares_batch_for_model = []
    square_coords_for_mapping = []

    for r_idx in range(8):
        for c_idx in range(8):
            x1, y1 = c_idx * sz, r_idx * sz
            square_img_roi = warped_img[y1:y1+sz, x1:x1+sz]
            
            processed_square = cv2.resize(square_img_roi, (100, 100))
            processed_square = cv2.cvtColor(processed_square, cv2.COLOR_BGR2RGB).astype(np.float32)
            squares_batch_for_model.append(processed_square)
            square_coords_for_mapping.append((r_idx, c_idx))
            
            if debug and debug_squares_output_dir:
                cv2.imwrite(os.path.join(debug_squares_output_dir, f"{r_idx}_{c_idx}.png"), square_img_roi)

    if not squares_batch_for_model:
        print("Warning (_extract_preprocess_and_predict_squares): No squares extracted for prediction.")
        return np.array([]), [], sz # Return empty predictions and coords

    predictions_from_model = model.predict(np.array(squares_batch_for_model), verbose=0, batch_size=64)
    return predictions_from_model, square_coords_for_mapping, sz

# Add these improved functions to warped_to_fen.py to replace the existing position_to_fen function

def analyze_position_context(position: dict) -> dict:
    """
    Analyze the chess position to infer game state information.
    Returns a dictionary with inferred turn, castling rights, etc.
    """
    context = {
        'turn': 'w',  # Default to white
        'castling': 'KQkq',  # Default to all castling available
        'en_passant': '-',
        'halfmove': 0,
        'fullmove': 1
    }
    
    # Count pieces to get a rough estimate of game progress
    white_pieces = sum(1 for piece in position.values() if piece.endswith('_w'))
    black_pieces = sum(1 for piece in position.values() if piece.endswith('_b'))
    total_pieces = white_pieces + black_pieces
    
    # Simple heuristic: if we have fewer pieces, we're later in the game
    if total_pieces < 20:
        context['fullmove'] = max(10, (32 - total_pieces) // 2)
    elif total_pieces < 28:
        context['fullmove'] = max(5, (32 - total_pieces) // 3)
    
    # Check castling rights based on piece positions
    castling_rights = []
    
    # White castling
    if position.get('e1') == 'k_w':  # White king on starting square
        if position.get('h1') == 'r_w':  # Kingside rook
            castling_rights.append('K')
        if position.get('a1') == 'r_w':  # Queenside rook
            castling_rights.append('Q')
    
    # Black castling
    if position.get('e8') == 'k_b':  # Black king on starting square
        if position.get('h8') == 'r_b':  # Kingside rook
            castling_rights.append('k')
        if position.get('a8') == 'r_b':  # Queenside rook
            castling_rights.append('q')
    
    context['castling'] = ''.join(castling_rights) if castling_rights else '-'
    
    return context

def infer_turn_from_position_sequence(current_position: dict, previous_position: dict = None) -> str:
    """
    Try to infer whose turn it is based on piece movement between positions.
    This is a complex heuristic and may not always be accurate.
    """
    if previous_position is None:
        return 'w'  # Default to white for starting position
    
    # Count pieces moved
    white_changes = 0
    black_changes = 0
    
    for square in set(list(current_position.keys()) + list(previous_position.keys())):
        current_piece = current_position.get(square, 'empty')
        previous_piece = previous_position.get(square, 'empty')
        
        if current_piece != previous_piece:
            if current_piece.endswith('_w') or previous_piece.endswith('_w'):
                white_changes += 1
            if current_piece.endswith('_b') or previous_piece.endswith('_b'):
                black_changes += 1
    
    # Simple heuristic: if white pieces moved more recently, it's black's turn
    if white_changes > black_changes:
        return 'b'
    elif black_changes > white_changes:
        return 'w'
    else:
        return 'w'  # Default to white if unclear


def _build_results_from_predictions_improved(
        predictions_matrix: np.ndarray, 
        square_coords_for_mapping: list, 
        class_names: list, 
        orientation: int, 
        warped_img: np.ndarray, 
        square_size: int, 
        previous_position: dict = None,
        move_number: int = 1,
        debug: bool = False) -> tuple[dict, np.ndarray, str]:
    """
    Improved version that tracks game state for better FEN generation.
    """
    position = {}
    visual = warped_img.copy() if debug else None
    square_map = get_square_mapping(orientation)

    for i, (r_idx, c_idx) in enumerate(square_coords_for_mapping):
        predicted_class_index = int(np.argmax(predictions_matrix[i]))
        
        if predicted_class_index < len(class_names):
            piece_name = class_names[predicted_class_index]
        else:
            piece_name = "unknown"
            print(f"Warning (analyze_chess_position): Predicted class index {predicted_class_index} out of bounds for class_names (len: {len(class_names)}).")

        position[square_map[(r_idx, c_idx)]] = piece_name
        
        if debug and visual is not None and piece_name not in ("unknown", "error", "empty", "empty_dot"):
            x1, y1 = c_idx * square_size, r_idx * square_size
            cx, cy = x1 + square_size//2, y1 + square_size//2
            square_img_roi = warped_img[y1:y1+square_size, x1:x1+square_size]
            avg_brightness = np.mean(square_img_roi)
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            short_notation = get_short_piece_notation(piece_name)
            cv2.putText(visual, short_notation, (cx-10, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Use improved FEN generation
    move_context = {
        'turn': 'w' if move_number % 2 == 1 else 'b',  # Alternate turns
        'castling': 'KQkq',  # Will be analyzed by the function
        'en_passant': '-',
        'halfmove': 0,
        'fullmove': (move_number + 1) // 2
    }
    
    fen_string = position_to_fen_improved(position, previous_position, move_context)
    return position, visual, fen_string

def _process_single_image_to_fen(image_path: str,
                                 model_path: str,
                                 debug: bool = False,
                                 debug_output_dir: str = None,
                                 image_filename_for_debug: str = "",
                                 kmeans_sample_size: int = 5000,
                                 kmeans_iterations: int = 30,
                                 morph_kernel_size: int = 15) -> str | None:
    """Helper function to process one image and return its FEN string."""
    # Define the output directory for find_and_warp_chessboard's own debug files (e.g., corners image).
    # This is only used if 'debug' is True.
    fawc_debug_output_dir = None # fawc = find_and_warp_chessboard
    if debug: # If overall debugging is enabled for the FEN process
        if debug_output_dir: # If a main debug directory is provided for warped_to_fen
            fawc_debug_output_dir = os.path.join(debug_output_dir, "detector_outputs")
            # find_and_warp_chessboard will create this subdir if it needs to save debug files.
        else:
            # If no main debug_output_dir is provided, fawc_debug_output_dir remains None.
            # find_and_warp_chessboard will print a warning if its debug is True but output_dir is None.
            # warped_to_fen's own debug images also won't be saved if debug_output_dir is None.
            pass

    # Call find_and_warp_chessboard. It returns the warped image (NumPy array) or None.
    # Its 'debug' flag controls its *internal* debug image generation (e.g., corners).
    # Its 'output_dir' (fawc_debug_output_dir) is where its debug images are saved.
    warped_board = find_and_warp_chessboard(
        image_path=image_path,
        output_dir=fawc_debug_output_dir, 
        debug=debug,
        kmeans_sample_size=kmeans_sample_size,
        kmeans_iterations=kmeans_iterations,
        morph_kernel_size=morph_kernel_size
    )

    if warped_board is None:
        return None
    
    # Load model and metadata once
    model, class_names = _load_model_and_metadata(model_path)
    if model is None: # Model loading failed
        return None

    # Extract squares and get predictions once
    # For the initial pass of saving debug squares, we can create a specific path
    initial_debug_squares_dir = None
    if debug and debug_output_dir and image_filename_for_debug:
        initial_debug_squares_dir = os.path.join(debug_output_dir, image_filename_for_debug + "_predicted_squares_raw")

    predictions_matrix, square_coords, square_size = _extract_preprocess_and_predict_squares(
        warped_board, model, debug=(debug and bool(initial_debug_squares_dir)), debug_squares_output_dir=initial_debug_squares_dir
    )

    if not predictions_matrix.size: # No predictions could be made
        return "8/8/8/8/8/8/8/8 w KQkq - 0 1" # Return empty FEN

    # Step 1: Build initial position_dict with default orientation (0) for refined orientation check
    initial_position_dict, _, _ = _build_results_from_predictions_improved(
        predictions_matrix, square_coords, class_names, orientation=0,
        warped_img=warped_board, square_size=square_size, debug=False # No visual overlay needed for this step
    )

    # Step 2: Refine orientation
    orientation_debug_image_save_path = None
    if debug and debug_output_dir and image_filename_for_debug:
        orientation_debug_image_save_path = os.path.join(debug_output_dir, image_filename_for_debug + "_orientation_debug.png")
    final_orientation = determine_refined_orientation(
        warped_board, initial_position_dict, initial_orientation_code=0, # We used 0 for the initial analysis
        debug=debug, debug_image_path=orientation_debug_image_save_path
    )

    # Step 3: Build final results (position_dict, visual, FEN) using the refined orientation
    # and the predictions we already have.
    analysis_debug_squares_save_dir = None
    if debug and debug_output_dir and image_filename_for_debug:
        analysis_debug_squares_save_dir = os.path.join(debug_output_dir, image_filename_for_debug + "_predicted_squares")
        # Note: _extract_preprocess_and_predict_squares already saved raw squares if initial_debug_squares_dir was set.
        # This path is for potentially different/final debug outputs if needed, or just for consistency.

    try:
        final_position_dict, visual_overlay, fen_string = _build_results_from_predictions_improved(
            predictions_matrix, square_coords, class_names, final_orientation,
            warped_img=warped_board, square_size=square_size, debug=debug
        )
        # If debug and visual_overlay is created, you might want to save it:
        # if debug and visual_overlay is not None and debug_output_dir and image_filename_for_debug:
        #     cv2.imwrite(os.path.join(debug_output_dir, image_filename_for_debug + "_final_overlay.png"), visual_overlay)
        return fen_string
    except Exception as e:
        print(f"Error (_process_single_image_to_fen): Exception during FEN analysis for {image_path}: {e}")
        return None


def _process_single_image_to_fen_wrapper(args):
    """Helper function to unpack arguments for multiprocessing.Pool.map"""
    return _process_single_image_to_fen(*args)

def get_fen_from_image_or_dir(path_to_process: str,
                              model_path: str,
                              debug: bool = False,
                              debug_output_dir: str = "w2f_debug_output",
                              kmeans_sample_size: int = 5000,
                              kmeans_iterations: int = 30,
                              morph_kernel_size: int = 15,
                              num_workers: int = None) -> str | dict | None:
    """
    Processes a single image or all images in a directory to extract FEN positions.

    Args:
        path_to_process (str): Path to a single image file or a directory of images.
        model_path (str): Path to the CNN model for piece recognition.
        debug (bool): If True, saves intermediate debug images.
        debug_output_dir (str): Directory to save debug images.
        kmeans_sample_size (int): Sample size for k-means in board detection.
        kmeans_iterations (int): Max iterations for k-means.
        morph_kernel_size (int): Kernel size for morphological operations.
        num_workers (int, optional): Number of worker processes for directory processing. Defaults to CPU count.

    Returns:
        str: FEN string if a single image is processed successfully.
        dict: Dictionary mapping image filenames to FEN strings if a directory is processed.
              Value is None if FEN extraction failed for a specific image.
        None: If the input path is invalid or other critical errors occur.
    """
    if not os.path.exists(path_to_process):
        print(f"Error (get_fen_from_image_or_dir): Path does not exist: {path_to_process}")
        return None

    if debug and debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)

    if os.path.isfile(path_to_process):
        base_name = os.path.splitext(os.path.basename(path_to_process))[0]
        return _process_single_image_to_fen(path_to_process, model_path,
                                            debug, debug_output_dir, base_name,
                                            kmeans_sample_size, kmeans_iterations, morph_kernel_size)
    
    elif os.path.isdir(path_to_process):
        fen_results = {}
        image_files = [f for f in os.listdir(path_to_process) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Prepare arguments for each image processing task
        tasks = []
        for img_file in sorted(image_files):
            full_path = os.path.join(path_to_process, img_file)
            base_name = os.path.splitext(img_file)[0]
            tasks.append((full_path, model_path, debug, debug_output_dir, base_name,
                          kmeans_sample_size, kmeans_iterations, morph_kernel_size))

        if num_workers == 1: # Sequential processing if num_workers is 1
            print("  Processing images sequentially...")
            for i, task_args in enumerate(tasks):
                img_file = image_files[i]
                print(f"    Processing: {img_file}")
                fen_results[img_file] = _process_single_image_to_fen(*task_args)
        else:
            # Use multiprocessing Pool
            # If num_workers is None, Pool defaults to os.cpu_count()
            actual_num_workers = num_workers if num_workers is not None else os.cpu_count()
            print(f"Processing images with {actual_num_workers} workers...")
            with multiprocessing.Pool(processes=actual_num_workers) as pool:
                # map will block until all results are processed
                # it also preserves the order of results corresponding to tasks
                results_list = pool.map(_process_single_image_to_fen_wrapper, tasks)
            
            for i, img_file in enumerate(sorted(image_files)):
                fen_results[img_file] = results_list[i]
                print(f"    Finished: {img_file} -> {results_list[i]}")

        return fen_results
    else:
        print(f"Error (get_fen_from_image_or_dir): Path is not a valid file or directory: {path_to_process}")
        return None

if __name__ == "__main__":
    # Default usage:
    results = get_fen_from_image_or_dir(
         '../detector/images', # Replace with your image path or directory
         '../detector/chess_piece_model.keras', # Replace with your model path
         debug=False, # Set to True to get debug images
         num_workers=None # Use None for os.cpu_count() or 1 for sequential
    )
    
    if isinstance(results, dict):
        for img_name, fen in results.items():
            print(f"{img_name}: {fen}")
    elif results is not None:
        print(f"FEN: {results}")