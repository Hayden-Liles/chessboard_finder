import cv2
import numpy as np
import argparse
from itertools import combinations
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

def extract_board_region_optimized(img, k=4, sample_size=5000, debug=False, debug_path_prefix="debug_"):
    """
    Optimized version of extract_board_region that preserves original image quality.
    Focuses on optimizing the algorithm, not reducing image quality.
    """
    h, w = img.shape[:2]
    
    # Process at original resolution - preserve image quality
    img_small = img
    h_small, w_small = h, w
    
    # Optimize: Use a smaller sample size for faster clustering
    current_sample_size = min(sample_size, h_small*w_small)
    if h_small * w_small == 0: # Handle empty image case
        print("Warning: Image dimensions are zero.")
        return None
    if current_sample_size == 0 and h_small * w_small > 0: # Ensure sample size is at least 1 if image is not empty
        current_sample_size = 1

    if current_sample_size > 0:
        coords = np.random.choice(h_small*w_small, current_sample_size, replace=False)
        pixels = img_small.reshape(-1,3)[coords].astype(np.float32)
    else: # If image is too small for even one sample point (e.g. 1x1 and sample_size=0)
        pixels = img_small.reshape(-1,3).astype(np.float32)
        if pixels.shape[0] == 0: # if image is 0x0
            return None

    # Run k-means with fewer iterations (30 instead of 50)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    if len(pixels) < k: # K-means requires num_samples >= num_clusters
        print(f"Warning: Not enough pixel samples ({len(pixels)}) for k-means clustering with k={k}. Skipping board detection.")
        return None

    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    
    # Optimize: Process fewer pixels for full image segmentation
    # Instead of processing all pixels, sample a grid
    stride = 2  # Process every other pixel
    y_indices, x_indices = np.mgrid[0:h_small:stride, 0:w_small:stride]
    pixel_indices = y_indices.flatten() * w_small + x_indices.flatten()
    
    # Get sampled pixels
    sampled_pixels = img_small.reshape(-1, 3)[pixel_indices].astype(np.float32)
    
    # Calculate distances and labels for sampled pixels
    dists = np.linalg.norm(sampled_pixels[:,None] - centers[None,:], axis=2)
    sampled_labels = np.argmin(dists, axis=1)
    
    # Reshape labels to match sampled grid
    grid_labels = sampled_labels.reshape(y_indices.shape)
    
    # Resize back to original size
    full_labels = np.zeros((h_small, w_small), dtype=np.int32)
    for y_idx_grid, row_val in enumerate(y_indices[:,0]):
        for x_idx_grid, col_val in enumerate(x_indices[0,:]):
            y_start, y_end = row_val, min(row_val + stride, h_small)
            x_start, x_end = col_val, min(col_val + stride, w_small)
            full_labels[y_start:y_end, x_start:x_end] = grid_labels[y_idx_grid, x_idx_grid]
    
    best_score = float('inf')
    best_box = None

    for i, j in combinations(range(k), 2):
        mask = np.zeros((h_small, w_small), np.uint8)
        mask[np.logical_or(full_labels==i, full_labels==j)] = 255

        kern_size = 15
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_size, kern_size))
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
        if min(bw,bh) == 0: continue # Avoid division by zero
        aspect = max(bw,bh)/min(bw,bh)
        score = abs(1.0 - aspect)

        ar = area / float(h_small*w_small)
        if ar > 0.9 or ar < 0.02:
            score += 1.0

        if score < best_score:
            best_score = score
            best_box = cv2.boxPoints(rect).astype(np.float32)

            if debug:
                debug_mask_img = img.copy()
                cv2.drawContours(debug_mask_img, [cnt.astype(int)], -1, (0, 255, 0), 2)
                cv2.imwrite(f'{debug_path_prefix}mask_clusters_{i}_{j}.png', debug_mask_img)

    if best_box is not None and debug:
        vis = img.copy()
        for idx, (x, y) in enumerate(best_box):
            color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)][idx % 4]
            cv2.circle(vis, (int(x),int(y)), 10, color, -1)
            cv2.putText(vis, f"C{idx}", (int(x)-10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imwrite(f'{debug_path_prefix}corners_color.png', vis)

    return best_box

def warp_and_draw(img, pts, size=800, thickness=2):
    """Warp the board to a square and optionally overlay an 8x8 grid."""
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (size, size))

    # Create overlay for the grid on the original image
    overlay_on_original = np.zeros_like(img) # Grid lines will be drawn on this
    grid_color = (0,255,0) # Green grid lines
    step = size // 8
    for i in range(9): # Draw 9 lines for 8 squares
        # Create points for grid lines on the warped image
        pt1_h_warped = (i*step,0)
        pt2_h_warped = (i*step,size-1)
        pt1_v_warped = (0,i*step)
        pt2_v_warped = (size-1,i*step)
        
        # Draw on warped image (for visualization if needed)
        # cv2.line(warp, pt1_h_warped, pt2_h_warped, grid_color, thickness)
        # cv2.line(warp, pt1_v_warped, pt2_v_warped, grid_color, thickness)

    # To draw grid on original image, we need to inverse warp the grid lines
    # This part is more complex if you want to draw the grid back onto the original image
    # For simplicity, this example returns the warped board and the original with corners.
    # The original `warp_and_draw` drew the grid on a separate overlay then warped it back.
    # We will return the warped image and the transform matrix.
    return warp, M

def find_and_warp_chessboard(image_path, output_dir, debug=False):
    """Loads an image, finds the chessboard, warps it, and saves the results."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Processing {image_path}...")
    
    # Downscale image for faster processing if it's very large, similar to app2.py
    h_orig, w_orig = img.shape[:2]
    scale_cap = 800 
    if max(h_orig, w_orig) > scale_cap:
        scale = scale_cap / max(h_orig, w_orig)
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Image scaled down by factor of {scale:.2f}")
    else:
        img_scaled = img.copy()

    
    debug_prefix = os.path.join(output_dir, os.path.basename(image_path) + "_debug_")
    pts = extract_board_region_optimized(img_scaled, k=4, sample_size=5000, debug=debug, debug_path_prefix=debug_prefix)

    if pts is not None:
        print("Chessboard corners found.")
        # Scale points back to original image coordinates if scaled
        if img_scaled.shape != img.shape:
            pts_original_scale = pts / scale
        else:
            pts_original_scale = pts

        # Draw corners on the original full-size image
        img_with_corners = img.copy()
        for i, p in enumerate(pts_original_scale):
            cv2.circle(img_with_corners, tuple(p.astype(int)), 10, (0,0,255) if i==0 else (0,255,0), -1) # TL red, others green
        
        corner_path = os.path.join(output_dir, os.path.basename(image_path) + "_corners.png")
        cv2.imwrite(corner_path, img_with_corners)
        print(f"Saved image with detected corners to {corner_path}")

        warped_board, _ = warp_and_draw(img, pts_original_scale, size=800) # Use original image for warping for best quality
        warped_path = os.path.join(output_dir, os.path.basename(image_path) + "_warped.png")
        cv2.imwrite(warped_path, warped_board)
        print(f"Saved warped chessboard to {warped_path}")
    else:
        print("Could not find chessboard in the image.")

find_and_warp_chessboard('./images/1.jpeg', './output2')