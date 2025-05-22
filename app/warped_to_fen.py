import cv2
import numpy as np
import tensorflow as tf
import os
import json
from itertools import combinations # For extract_board_region_optimized

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

def extract_board_region_optimized(img, k=4, sample_size=5000, debug=False, debug_path_prefix=None):
    """
    Optimized version of extract_board_region that preserves original image quality.
    Focuses on optimizing the algorithm, not reducing image quality.
    """
    h, w = img.shape[:2]
    
    img_small = img
    h_small, w_small = h, w
    
    current_sample_size = min(sample_size, h_small*w_small)
    if h_small * w_small == 0:
        print("Warning (warper_to_fen): Image dimensions are zero in extract_board_region_optimized.")
        return None
    if current_sample_size == 0 and h_small * w_small > 0:
        current_sample_size = 1

    if current_sample_size > 0:
        coords = np.random.choice(h_small*w_small, current_sample_size, replace=False)
        pixels = img_small.reshape(-1,3)[coords].astype(np.float32)
    else:
        pixels = img_small.reshape(-1,3).astype(np.float32)
        if pixels.shape[0] == 0:
            return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    if len(pixels) < k:
        print(f"Warning (warper_to_fen): Not enough pixel samples ({len(pixels)}) for k-means with k={k}.")
        return None

    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
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

        kern_size = 15
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_size, kern_size))
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

def get_warped_image(img, pts, size=800):
    """Warps the detected board region to a square image."""
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (size, size))
    return warp

# --- End of Board Detection and Warping Functions ---

def determine_orientation(warped_img, debug=False, debug_image_path=None):
    """
    Determine board orientation by analyzing corner square colors.
    Returns an orientation code: 0-3.
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
        print("Warning (warper_to_fen): Not enough brightness values to determine threshold. Defaulting orientation.")
        return 0
    threshold = (sorted_bright[1] + sorted_bright[2]) / 2
    is_white = [b > threshold for b in brightness]

    if debug and debug_image_path:get_short_piece_notation

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

    print("Warning (warper_to_fen): Chess board color pattern unclear. Defaulting to standard orientation.")
    return 0

def get_square_mapping(orientation=0):
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

def get_short_piece_notation(piece_type):
    """Convert p_w/p_b style piece type to short notation for visualization."""
    piece_map = {
        'p_w': 'wP', 'n_w': 'wN', 'b_w': 'wB', 'r_w': 'wR', 'q_w': 'wQ', 'k_w': 'wK',
        'p_b': 'bP', 'n_b': 'bN', 'b_b': 'bB', 'r_b': 'bR', 'q_b': 'bQ', 'k_b': 'bK',
        'empty': '', 'empty_dot': '',
    }
    return piece_map.get(piece_type, piece_type)

def position_to_fen(position, orientation=0): # orientation parameter is not strictly used here
    """
    Convert position dictionary to FEN (Forsyth-Edwards Notation).
    """
    fen_map = {
        'p_w': 'P', 'n_w': 'N', 'b_w': 'B', 'r_w': 'R', 'q_w': 'Q', 'k_w': 'K',
        'p_b': 'p', 'n_b': 'n', 'b_b': 'b', 'r_b': 'r', 'q_b': 'q', 'k_b': 'k',
        'empty': '1', 'error': '1', 'unknown': '1', 'empty_dot': '1'
    }
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
    fen = '/'.join(board_fen)
    fen += " w KQkq - 0 1" # Default to white to move, standard castling, no en passant, etc.
    return fen

def analyze_chess_position(warped_img, model_path, orientation=0, debug=False, debug_squares_output_dir=None):
    """
    Identify all 64 squares, classify each via the CNN, and return:
      - position dict (e.g. {'a1': 'k_w', …})
      - a visual overlay image
      - a FEN string
    """
    model = load_model_once(model_path)
    
    meta_path = os.path.splitext(model_path)[0] + ".keras.metadata"
    if not os.path.exists(meta_path):
        # Fallback for older model naming or if metadata is in a different location
        alt_meta_path = model_path.replace(".h5", ".keras.metadata") # Common alternative
        if os.path.exists(alt_meta_path):
            meta_path = alt_meta_path
        else:
            # Try one level up if model is in a subdirectory like 'models/'
            parent_dir_meta_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), os.path.basename(meta_path))
            if os.path.exists(parent_dir_meta_path):
                 meta_path = parent_dir_meta_path
            else:
                print(f"Warning (warper_to_fen): Metadata file not found at {meta_path} or {alt_meta_path} or {parent_dir_meta_path}. Piece names might be incorrect.")
                # Default class names if metadata is missing
                class_names = [f'class_{i}' for i in range(model.output_shape[-1])] # Assuming model output gives number of classes

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            class_names = [meta["class_mapping"][str(i)] for i in range(len(meta["class_mapping"]))]
        except Exception as e:
            print(f"Warning (warper_to_fen): Error loading metadata from {meta_path}: {e}. Using default class names.")
            class_names = [f'class_{i}' for i in range(model.output_shape[-1])]

    h, _ = warped_img.shape[:2]
    sz = h // 8
    visual = warped_img.copy()
    square_map = get_square_mapping(orientation)
    position = {}
    
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
        print("Warning (warper_to_fen): No squares extracted for prediction.")
        return {}, visual, "8/8/8/8/8/8/8/8 w KQkq - 0 1" # Empty FEN

    predictions_from_model = model.predict(np.array(squares_batch_for_model), verbose=0, batch_size=64)
    
    for i, (r_idx, c_idx) in enumerate(square_coords_for_mapping):
        predicted_class_index = int(np.argmax(predictions_from_model[i]))
        
        if predicted_class_index < len(class_names):
            piece_name = class_names[predicted_class_index]
        else:
            piece_name = "unknown" # Fallback if index is out of bounds
            print(f"Warning (warper_to_fen): Predicted class index {predicted_class_index} out of bounds for class_names (len: {len(class_names)}).")

        position[square_map[(r_idx, c_idx)]] = piece_name
        
        if piece_name not in ("unknown", "error", "empty", "empty_dot"):
            x1, y1 = c_idx * sz, r_idx * sz
            cx, cy = x1 + sz//2, y1 + sz//2
            square_img_roi = warped_img[y1:y1+sz, x1:x1+sz]
            avg_brightness = np.mean(square_img_roi)
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            short_notation = get_short_piece_notation(piece_name)
            cv2.putText(visual, short_notation, (cx-10, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    fen_string = position_to_fen(position, orientation) # orientation is not used by current position_to_fen
    return position, visual, fen_string


def _process_single_image_to_fen(image_path, model_path, debug=False, debug_output_dir=None, image_filename_for_debug=""):
    """Helper function to process one image and return its FEN string."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error (warper_to_fen): Could not load image from {image_path}")
        return None

    h_orig, w_orig = img.shape[:2]
    scale_cap = 800 
    scale = 1.0
    if max(h_orig, w_orig) > scale_cap:
        scale = scale_cap / max(h_orig, w_orig)
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_scaled = img.copy()

    current_debug_prefix_for_extract = None
    if debug and debug_output_dir and image_filename_for_debug:
        current_debug_prefix_for_extract = os.path.join(debug_output_dir, image_filename_for_debug + "_extract_debug_")
    
    pts_scaled = extract_board_region_optimized(img_scaled, k=4, sample_size=5000, 
                                                debug=debug, debug_path_prefix=current_debug_prefix_for_extract)

    if pts_scaled is None:
        print(f"Info (warper_to_fen): Could not find chessboard in {image_path}")
        return None

    pts_original_scale = pts_scaled / scale if scale < 1.0 else pts_scaled

    warped_board = get_warped_image(img, pts_original_scale, size=800)

    if debug and debug_output_dir and image_filename_for_debug:
        warped_debug_path = os.path.join(debug_output_dir, image_filename_for_debug + "_warped.png")
        cv2.imwrite(warped_debug_path, warped_board)
        # print(f"Debug: Saved warped board to {warped_debug_path}")

    orientation_debug_image_save_path = None
    if debug and debug_output_dir and image_filename_for_debug:
        orientation_debug_image_save_path = os.path.join(debug_output_dir, image_filename_for_debug + "_orientation_colors.png")
    orientation = determine_orientation(warped_board, debug=debug, debug_image_path=orientation_debug_image_save_path)

    analysis_debug_squares_save_dir = None
    if debug and debug_output_dir and image_filename_for_debug:
        analysis_debug_squares_save_dir = os.path.join(debug_output_dir, image_filename_for_debug + "_predicted_squares")
    
    try:
        _, _, fen_string = analyze_chess_position(warped_board, model_path, orientation, 
                                                  debug=debug, debug_squares_output_dir=analysis_debug_squares_save_dir)
        return fen_string
    except Exception as e:
        print(f"Error (warper_to_fen): Exception during FEN analysis for {image_path}: {e}")
        return None

def get_fen_from_image_or_dir(path_to_process, model_path, debug=False, debug_output_dir="w2f_debug_output"):
    """
    Processes a single image or all images in a directory to extract FEN positions.

    Args:
        path_to_process (str): Path to a single image file or a directory of images.
        model_path (str): Path to the CNN model for piece recognition.
        debug (bool): If True, saves intermediate debug images.
        debug_output_dir (str): Directory to save debug images.

    Returns:
        str: FEN string if a single image is processed successfully.
        dict: Dictionary mapping image filenames to FEN strings if a directory is processed.
              Value is None if FEN extraction failed for a specific image.
        None: If the input path is invalid or other critical errors occur.
    """
    if not os.path.exists(path_to_process):
        print(f"Error (warper_to_fen): Path does not exist: {path_to_process}")
        return None

    if debug and debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"Info (warper_to_fen): Debug output will be saved to: {debug_output_dir}")

    if os.path.isfile(path_to_process):
        print(f"Info (warper_to_fen): Processing single image: {path_to_process}")
        base_name = os.path.splitext(os.path.basename(path_to_process))[0]
        return _process_single_image_to_fen(path_to_process, model_path, debug, debug_output_dir, base_name)
    
    elif os.path.isdir(path_to_process):
        print(f"Info (warper_to_fen): Processing directory: {path_to_process}")
        fen_results = {}
        image_files = [f for f in os.listdir(path_to_process) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        for img_file in sorted(image_files): # Sort for consistent processing order
            full_path = os.path.join(path_to_process, img_file)
            base_name = os.path.splitext(img_file)[0]
            print(f"  Processing: {img_file}")
            fen_results[img_file] = _process_single_image_to_fen(full_path, model_path, debug, debug_output_dir, base_name)
        return fen_results
    else:
        print(f"Error (warper_to_fen): Path is not a valid file or directory: {path_to_process}")
        return None

print(get_fen_from_image_or_dir('../detector/images', '../detector/chess_piece_model.keras'))