import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import json
from itertools import combinations
import multiprocessing
from PIL import Image
from tqdm import tqdm
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import psutil

# Global model cache to avoid reloading models
MODEL_CACHE = {}

# Set multiprocessing start method to avoid CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

class ChessPieceModel(nn.Module):
    """MobileNetV2-based model for chess piece recognition - must match train.py"""
    
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained MobileNetV2 (fix deprecation warning)
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.frozen = freeze_backbone
    
    def forward(self, x):
        return self.backbone(x)

def load_model_once(model_path, force_cpu_in_multiprocessing=True):
    """Load the PyTorch model once and cache it for future use"""
    global MODEL_CACHE
    
    # Create cache key that includes CPU/GPU preference
    cache_key = f"{model_path}_cpu" if force_cpu_in_multiprocessing else model_path
    
    if cache_key not in MODEL_CACHE:
        try:
            print(f"Loading PyTorch model from: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get class mapping - handle different possible formats
            class_mapping = checkpoint.get('class_mapping', {})
            
            # If class_mapping is empty, try other possible keys
            if not class_mapping:
                # Try alternative keys that might contain the mapping
                for key in ['class_to_idx', 'idx_to_class', 'classes', 'class_names']:
                    if key in checkpoint:
                        class_mapping = checkpoint[key]
                        print(f"Found class mapping under key: {key}")
                        break
            
            num_classes = len(class_mapping)
            
            if num_classes == 0:
                raise ValueError("No class mapping found in checkpoint")
            
            print(f"Number of classes: {num_classes}")
            
            # Create model instance
            model = ChessPieceModel(num_classes, pretrained=False, freeze_backbone=False)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading directly (in case it's just the state dict)
                model.load_state_dict(checkpoint)
            
            # Set to evaluation mode
            model.eval()
            
            # Device selection: force CPU in multiprocessing to avoid CUDA issues
            if force_cpu_in_multiprocessing and multiprocessing.current_process().name != 'MainProcess':
                device = torch.device('cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu_in_multiprocessing else 'cpu')
            
            model = model.to(device)
            
            MODEL_CACHE[cache_key] = {
                'model': model,
                'class_mapping': class_mapping,
                'device': device
            }
            
            print(f"Model loaded successfully: {num_classes} classes on {device}")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
    
    return MODEL_CACHE[cache_key]

class OptimizedChessProcessor:
    """Optimized chess board processor with batch processing and caching"""
    
    def __init__(self, model_path, batch_size=32, cache_size=100):
        self.model_path = model_path
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.board_cache = {}  # Cache for detected board regions
        self.model = None
        self.class_names = None
        self.device = None
        self.transform = None
        self._init_model()
        
    def _init_model(self):
        """Initialize model and preprocessing once"""
        model_info = load_model_once(self.model_path, force_cpu_in_multiprocessing=False)
        self.model = model_info['model']
        class_mapping = model_info['class_mapping']
        self.device = model_info['device']
        
        # Convert class mapping to ordered list
        num_classes = len(class_mapping)
        class_names = [None] * num_classes
        
        first_key = next(iter(class_mapping))
        first_value = class_mapping[first_key]
        
        if isinstance(first_value, int):
            for class_name, idx in class_mapping.items():
                if 0 <= idx < num_classes:
                    class_names[idx] = class_name
        elif isinstance(first_key, int):
            for idx, class_name in class_mapping.items():
                if 0 <= idx < num_classes:
                    class_names[idx] = class_name
        elif isinstance(first_key, str) and first_key.isdigit():
            for idx_str, class_name in class_mapping.items():
                idx = int(idx_str)
                if 0 <= idx < num_classes:
                    class_names[idx] = class_name
        
        self.class_names = class_names
        
        # Initialize preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def process_images_batch(self, image_paths, progress_callback=None):
        """Process multiple images in batches for maximum efficiency"""
        results = {}
        
        # Step 1: Detect and warp boards in parallel
        if progress_callback:
            progress_callback("Detecting chess boards...")
            
        board_data = []
        with ThreadPoolExecutor(max_workers=min(8, len(image_paths))) as executor:
            futures = []
            for img_path in image_paths:
                future = executor.submit(self._detect_and_warp_board, img_path)
                futures.append((img_path, future))
            
            for img_path, future in tqdm(futures, desc="Board detection", disable=not progress_callback):
                try:
                    warped_board = future.result(timeout=30)  # 30 second timeout per image
                    if warped_board is not None:
                        board_data.append((img_path, warped_board))
                    else:
                        results[os.path.basename(img_path)] = None
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    results[os.path.basename(img_path)] = None
        
        if not board_data:
            return results
            
        if progress_callback:
            progress_callback("Extracting chess squares...")
            
        # Step 2: Extract all squares from all boards
        all_squares = []
        square_metadata = []  # (image_path, row, col)
        
        for img_path, warped_board in tqdm(board_data, desc="Square extraction", disable=not progress_callback):
            squares = self._extract_squares_from_board(warped_board)
            for (r, c), square_img in squares:
                all_squares.append(square_img)
                square_metadata.append((img_path, r, c))
        
        if not all_squares:
            return results
            
        if progress_callback:
            progress_callback("Running neural network inference...")
            
        # Step 3: Batch process all squares through neural network
        all_predictions = self._predict_squares_batch(all_squares, progress_callback)
        
        if progress_callback:
            progress_callback("Building FEN positions...")
            
        # Step 4: Group predictions back by image and build FEN
        current_img_path = None
        current_predictions = {}
        
        for i, (img_path, r, c) in enumerate(square_metadata):
            if current_img_path != img_path:
                # Process previous image if exists
                if current_img_path is not None:
                    fen = self._build_fen_from_predictions(current_predictions)
                    results[os.path.basename(current_img_path)] = fen
                
                # Start new image
                current_img_path = img_path
                current_predictions = {}
            
            current_predictions[(r, c)] = all_predictions[i]
        
        # Process last image
        if current_img_path is not None:
            fen = self._build_fen_from_predictions(current_predictions)
            results[os.path.basename(current_img_path)] = fen
            
        return results
    
    def _detect_and_warp_board(self, image_path):
        """Detect and warp chess board with caching"""
        # Check cache first
        cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
        if cache_key in self.board_cache:
            return self.board_cache[cache_key]
            
        try:
            warped_board = find_and_warp_chessboard(
                image_path=image_path,
                debug=False,
                kmeans_sample_size=3000,  # Reduced for speed
                kmeans_iterations=20,     # Reduced for speed
                morph_kernel_size=12      # Reduced for speed
            )
            
            # Cache result (limit cache size)
            if len(self.board_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.board_cache))
                del self.board_cache[oldest_key]
                
            self.board_cache[cache_key] = warped_board
            return warped_board
            
        except Exception as e:
            print(f"Error detecting board in {image_path}: {e}")
            return None
    
    def _extract_squares_from_board(self, warped_board):
        """Extract all 64 squares from warped board"""
        h, w = warped_board.shape[:2]
        square_size = h // 8
        squares = []
        
        for r in range(8):
            for c in range(8):
                y1, y2 = r * square_size, (r + 1) * square_size
                x1, x2 = c * square_size, (c + 1) * square_size
                square_img = warped_board[y1:y2, x1:x2]
                
                # Convert to RGB PIL Image for transforms
                square_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(square_rgb)
                transformed = self.transform(pil_image)
                
                squares.append(((r, c), transformed))
                
        return squares
    
    def _predict_squares_batch(self, square_tensors, progress_callback=None):
        """Predict all squares in batches"""
        all_predictions = []
        num_batches = (len(square_tensors) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(square_tensors), self.batch_size), 
                         desc="Neural network", total=num_batches, 
                         disable=not progress_callback):
                batch_tensors = square_tensors[i:i + self.batch_size]
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                outputs = self.model(batch_tensor)
                predictions = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(predictions, dim=1)
                
                for pred_idx in predicted_classes:
                    class_name = self.class_names[pred_idx.item()]
                    all_predictions.append(class_name)
        
        return all_predictions
    
    def _build_fen_from_predictions(self, predictions_dict):
        """Build FEN string from square predictions"""
        # Create position dictionary
        position = {}
        square_map = get_square_mapping(0)  # Assume standard orientation for now
        
        for (r, c), piece_name in predictions_dict.items():
            square_notation = square_map[(r, c)]
            position[square_notation] = piece_name
            
        return position_to_fen_improved(position)

# Keep original functions for compatibility
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
                                 kmeans_sample_size=3000,  # Reduced default
                                 kmeans_iterations=20,     # Reduced default
                                 morph_kernel_size=12,     # Reduced default
                                 debug=False,
                                 debug_path_prefix=None):
    """Optimized board detection with reduced parameters for speed"""
    h, w = img.shape[:2]
    
    # Use smaller image for detection to speed up
    max_size = 600
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h_small, w_small = img_small.shape[:2]
    else:
        img_small = img
        h_small, w_small = h, w
        scale = 1.0
    
    current_kmeans_sample_size = min(kmeans_sample_size, h_small*w_small)
    if h_small * w_small == 0:
        return None
    if current_kmeans_sample_size == 0 and h_small * w_small > 0:
        current_kmeans_sample_size = 1

    if current_kmeans_sample_size > 0:
        coords = np.random.choice(h_small*w_small, current_kmeans_sample_size, replace=False)
        pixels = img_small.reshape(-1,3)[coords].astype(np.float32)
    else:
        pixels = img_small.reshape(-1,3).astype(np.float32)
        if not pixels.size:
            return None

    if len(pixels) < k:
        return None

    _, _, centers = cv2.kmeans(pixels, k, None, 
                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_iterations, 1.0), 
                              3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    
    # Use larger stride for speed
    stride = 3
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
        if not cnts: 
            continue
        cnt = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        if area < 0.01 * (h_small*w_small): 
            continue

        rect = cv2.minAreaRect(cnt)
        _,(bw,bh),_ = rect
        if min(bw,bh) == 0: 
            continue
        aspect = max(bw,bh)/min(bw,bh)
        score = abs(1.0 - aspect)

        ar = area / float(h_small*w_small)
        if ar > 0.9 or ar < 0.02: 
            score += 1.0

        if score < best_score:
            best_score = score
            box_points = cv2.boxPoints(rect).astype(np.float32)
            # Scale back to original image coordinates
            best_box = box_points / scale

    return best_box

def warp_and_draw(img, pts, size=800, thickness=2):
    """Warp the board to a square and optionally overlay an 8x8 grid."""
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (size, size))
    return warp, M

def find_and_warp_chessboard(image_path: str,
                             output_dir: str = None,
                             debug: bool = False,
                             kmeans_sample_size: int = 3000,
                             kmeans_iterations: int = 20,
                             morph_kernel_size: int = 12):
    """Optimized chessboard detection and warping"""
    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h_orig, w_orig = img.shape[:2]
    scale_cap = 800 
    scale = 1.0
    if max(h_orig, w_orig) > scale_cap:
        scale = scale_cap / max(h_orig, w_orig)
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
        pts_original_scale = pts / scale if scale != 1.0 else pts
        warped_board, _ = warp_and_draw(img, pts_original_scale, size=800)
        return warped_board
    else:
        return None

def get_square_mapping(orientation: int = 0) -> dict:
    """Create a mapping from chessboard coordinates (0-7, 0-7) to chess notation."""
    mapping = {}
    files = "abcdefgh"
    ranks = "12345678"
    for row in range(8):
        for col in range(8):
            if orientation == 0: 
                notation = files[col] + ranks[7-row]
            elif orientation == 1: 
                notation = files[7-row] + ranks[7-col]
            elif orientation == 2: 
                notation = files[7-col] + ranks[row]
            elif orientation == 3: 
                notation = files[row] + ranks[col]
            else: 
                notation = files[col] + ranks[7-row]
            mapping[(row, col)] = notation
    return mapping

def position_to_fen_improved(position: dict, previous_position: dict = None, move_context: dict = None) -> str:
    """Convert position dictionary to FEN with improved game state inference."""
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
    
    # Default game state
    context = {
        'turn': 'w',
        'castling': 'KQkq',
        'en_passant': '-',
        'halfmove': 0,
        'fullmove': 1
    }
    
    if move_context:
        context.update(move_context)
    
    fen = f"{board_fen_str} {context['turn']} {context['castling']} {context['en_passant']} {context['halfmove']} {context['fullmove']}"
    return fen

def get_fen_from_image_or_dir_optimized(path_to_process: str,
                                        model_path: str,
                                        debug: bool = False,
                                        debug_output_dir: str = "w2f_debug_output",
                                        batch_size: int = 32,
                                        progress_callback=None) -> str | dict | None:
    """
    Optimized version with batch processing and progress tracking.
    
    Args:
        path_to_process: Path to image file or directory
        model_path: Path to PyTorch model
        debug: Enable debug mode
        debug_output_dir: Debug output directory
        batch_size: Batch size for neural network inference
        progress_callback: Callback function for progress updates
    """
    if not os.path.exists(path_to_process):
        print(f"Error: Path does not exist: {path_to_process}")
        return None

    if debug and debug_output_dir:
        os.makedirs(debug_output_dir, exist_ok=True)

    # Initialize optimized processor
    try:
        processor = OptimizedChessProcessor(model_path, batch_size=batch_size)
    except Exception as e:
        print(f"Error initializing chess processor: {e}")
        return None

    if os.path.isfile(path_to_process):
        # Single image processing
        if progress_callback:
            progress_callback("Processing single image...")
        results = processor.process_images_batch([path_to_process], progress_callback)
        return results.get(os.path.basename(path_to_process))
    
    elif os.path.isdir(path_to_process):
        # Directory processing
        image_files = [f for f in os.listdir(path_to_process) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No images found in {path_to_process}")
            return {}
            
        image_paths = [os.path.join(path_to_process, f) for f in sorted(image_files)]
        
        if progress_callback:
            progress_callback(f"Processing {len(image_paths)} images...")
            
        return processor.process_images_batch(image_paths, progress_callback)
    
    else:
        print(f"Error: Path is not a valid file or directory: {path_to_process}")
        return None

# Backward compatibility function
def get_fen_from_image_or_dir(path_to_process: str,
                              model_path: str,
                              debug: bool = False,
                              debug_output_dir: str = "w2f_debug_output",
                              kmeans_sample_size: int = 3000,
                              kmeans_iterations: int = 20,
                              morph_kernel_size: int = 12,
                              num_workers: int = None) -> str | dict | None:
    """Original function maintained for backward compatibility"""
    return get_fen_from_image_or_dir_optimized(
        path_to_process=path_to_process,
        model_path=model_path,
        debug=debug,
        debug_output_dir=debug_output_dir,
        batch_size=32,
        progress_callback=lambda x: print(f"Progress: {x}")
    )

if __name__ == "__main__":
    # Example usage with progress tracking
    def progress_callback(message):
        print(f"🔄 {message}")
    
    results = get_fen_from_image_or_dir_optimized(
        './test_videos/2/imgs',
        './chess_model.pth',
        debug=False,
        batch_size=64,  # Larger batch size for better performance
        progress_callback=progress_callback
    )
    
    if isinstance(results, dict):
        successful = sum(1 for fen in results.values() if fen is not None)
        print(f"✅ Successfully processed {successful}/{len(results)} images")
        for img_name, fen in list(results.items())[:5]:  # Show first 5
            print(f"{img_name}: {fen}")
    elif results is not None:
        print(f"FEN: {results}")