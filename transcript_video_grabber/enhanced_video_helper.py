#!/usr/bin/env python3
import os
import re
import sys
import argparse
import subprocess
import shutil
import cv2
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import board detection functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from warped_to_fen import find_and_warp_chessboard, extract_board_region_optimized, order_points
except ImportError:
    print("Error: Could not import warped_to_fen. Make sure it's in the same directory.")
    sys.exit(1)

# Cache for board regions to avoid re-detecting for every frame
BOARD_REGION_CACHE = {}

def detect_board_region_advanced(video_path, sample_times=[10.0, 30.0, 60.0, 90.0]):
    """
    Advanced board detection that tries multiple timestamps and filters results.
    Returns the best board corner points or None if not found.
    """
    if video_path in BOARD_REGION_CACHE:
        return BOARD_REGION_CACHE[video_path]
    
    print(f"Detecting chess board region in video: {os.path.basename(video_path)}")
    
    candidates = []
    
    for sample_time in sample_times:
        print(f"  Trying detection at {sample_time}s...")
        
        # Extract a sample frame
        temp_frame = f"/tmp/board_detect_frame_{sample_time}.png"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(sample_time), '-i', video_path,
            '-frames:v', '1', temp_frame
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Load the frame
            img = cv2.imread(temp_frame)
            if img is None:
                continue
            
            print(f"    Frame size: {img.shape}")
            
            # Save the detection frame for debugging
            debug_frame_path = f"/tmp/detection_debug_{sample_time}.png"
            cv2.imwrite(debug_frame_path, img)
            print(f"    Debug frame saved: {debug_frame_path}")
            
            # Try multiple detection approaches
            detection_results = []
            
            # Approach 1: Original scaling method
            h_orig, w_orig = img.shape[:2]
            scale_cap = 800
            scale = 1.0
            if max(h_orig, w_orig) > scale_cap:
                scale = scale_cap / max(h_orig, w_orig)
                img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                img_scaled = img.copy()
            
            # Try different parameters for board detection
            param_sets = [
                {'k': 4, 'kmeans_sample_size': 8000, 'kmeans_iterations': 50, 'morph_kernel_size': 20},
                {'k': 3, 'kmeans_sample_size': 5000, 'kmeans_iterations': 30, 'morph_kernel_size': 15},
                {'k': 5, 'kmeans_sample_size': 10000, 'kmeans_iterations': 40, 'morph_kernel_size': 25},
            ]
            
            for i, params in enumerate(param_sets):
                print(f"    Trying parameter set {i+1}: k={params['k']}, samples={params['kmeans_sample_size']}")
                
                pts = extract_board_region_optimized(
                    img_scaled, 
                    k=params['k'],
                    kmeans_sample_size=params['kmeans_sample_size'],
                    kmeans_iterations=params['kmeans_iterations'],
                    morph_kernel_size=params['morph_kernel_size'],
                    debug=False,
                    debug_path_prefix=f"/tmp/debug_{sample_time}_param{i}_"
                )
                
                if pts is not None:
                    # Scale points back to original size
                    pts_original = pts / scale
                    
                    # Validate the detection
                    score = validate_board_detection(img, pts_original)
                    print(f"      Detection score: {score:.3f}")
                    
                    if score > 0.3:  # Minimum quality threshold
                        detection_results.append({
                            'points': pts_original,
                            'score': score,
                            'timestamp': sample_time,
                            'params': params
                        })
                        
                        # Save debug image with detected region
                        debug_img = img.copy()
                        cv2.polylines(debug_img, [pts_original.astype(int)], True, (0, 255, 0), 3)
                        debug_path = f"/tmp/detection_result_{sample_time}_param{i}_score{score:.2f}.png"
                        cv2.imwrite(debug_path, debug_img)
                        print(f"      Detection result saved: {debug_path}")
            
            candidates.extend(detection_results)
            
        except Exception as e:
            print(f"    Error at {sample_time}s: {e}")
        finally:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
    
    # Select the best candidate
    if candidates:
        best_candidate = max(candidates, key=lambda x: x['score'])
        print(f"Best detection: timestamp={best_candidate['timestamp']}s, score={best_candidate['score']:.3f}")
        
        BOARD_REGION_CACHE[video_path] = best_candidate['points']
        return best_candidate['points']
    else:
        print("No valid board detection found!")
        return None

def validate_board_detection(img, pts):
    """
    Validate if the detected region is likely a chess board.
    Returns a score from 0-1 where higher is better.
    """
    if pts is None:
        return 0.0
    
    try:
        # Check if points form a reasonable quadrilateral
        rect = order_points(pts)
        
        # Calculate area
        area = cv2.contourArea(pts)
        img_area = img.shape[0] * img.shape[1]
        area_ratio = area / img_area
        
        # Board should be a significant portion but not the entire image
        if area_ratio < 0.05 or area_ratio > 0.8:
            return 0.0
        
        # Check aspect ratio (should be close to square)
        width = np.linalg.norm(rect[1] - rect[0])
        height = np.linalg.norm(rect[3] - rect[0])
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        if aspect_ratio > 2.0:  # Too rectangular
            return 0.0
        
        # Check if region is roughly centered (chess boards are usually prominent)
        center_x, center_y = np.mean(pts, axis=0)
        img_center_x, img_center_y = img.shape[1] / 2, img.shape[0] / 2
        center_dist = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
        center_score = 1.0 - (center_dist / max_dist)
        
        # Extract the region and check for board-like patterns
        warped = extract_board_region_from_frame_direct(img, pts)
        if warped is not None:
            pattern_score = analyze_board_pattern(warped)
        else:
            pattern_score = 0.0
        
        # Combine scores
        area_score = min(area_ratio * 5, 1.0)  # Favor larger boards
        aspect_score = max(0, 1.0 - (aspect_ratio - 1.0))  # Favor square shapes
        
        total_score = (area_score * 0.3 + aspect_score * 0.3 + center_score * 0.2 + pattern_score * 0.2)
        
        return min(total_score, 1.0)
        
    except Exception as e:
        print(f"Error in validation: {e}")
        return 0.0

def analyze_board_pattern(warped_img):
    """
    Analyze if the warped region has chess board-like patterns.
    Returns a score from 0-1.
    """
    try:
        if warped_img is None:
            return 0.0
        
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        # Check for alternating light/dark pattern
        h, w = gray.shape
        square_size = h // 8
        
        if square_size < 10:  # Too small to analyze
            return 0.0
        
        # Sample squares and check for alternating pattern
        pattern_score = 0.0
        sample_count = 0
        
        for row in range(0, 8, 2):  # Sample every other row
            for col in range(0, 8, 2):  # Sample every other column
                y1, y2 = row * square_size, (row + 1) * square_size
                x1, x2 = col * square_size, (col + 1) * square_size
                
                if y2 <= h and x2 <= w:
                    square = gray[y1:y2, x1:x2]
                    brightness = np.mean(square)
                    
                    # Check adjacent squares for contrast
                    adjacent_brightness = []
                    if col + 1 < 8:
                        adj_x1, adj_x2 = (col + 1) * square_size, (col + 2) * square_size
                        if adj_x2 <= w:
                            adj_square = gray[y1:y2, adj_x1:adj_x2]
                            adjacent_brightness.append(np.mean(adj_square))
                    
                    if row + 1 < 8:
                        adj_y1, adj_y2 = (row + 1) * square_size, (row + 2) * square_size
                        if adj_y2 <= h:
                            adj_square = gray[adj_y1:adj_y2, x1:x2]
                            adjacent_brightness.append(np.mean(adj_square))
                    
                    # Check for reasonable contrast
                    for adj_bright in adjacent_brightness:
                        contrast = abs(brightness - adj_bright)
                        if contrast > 20:  # Some contrast expected
                            pattern_score += 1.0
                        sample_count += 1
        
        if sample_count > 0:
            return min(pattern_score / sample_count, 1.0)
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error in pattern analysis: {e}")
        return 0.0

def extract_board_region_from_frame_direct(img, board_pts):
    """
    Extract board region directly from image and points.
    """
    if img is None or board_pts is None:
        return None
    
    try:
        # Order points and create perspective transform
        rect = order_points(board_pts)
        size = 800  # Standard board size
        dst = np.array([[0,0], [size-1,0], [size-1,size-1], [0,size-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Warp the board region
        warped = cv2.warpPerspective(img, M, (size, size))
        return warped
    except Exception as e:
        print(f"Error in direct board extraction: {e}")
        return None

def detect_board_region_fallback(video_path, sample_time=10.0):
    """
    Fallback board detection method (original approach).
    """
    print("Using fallback board detection method...")
    
    # Extract a sample frame
    temp_frame = "/tmp/board_detect_frame_fallback.png"
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(sample_time), '-i', video_path,
        '-frames:v', '1', temp_frame
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Load the frame
        img = cv2.imread(temp_frame)
        if img is None:
            return None
        
        # Save debug frame
        debug_fallback_path = "/tmp/fallback_debug_frame.png"
        cv2.imwrite(debug_fallback_path, img)
        print(f"Fallback debug frame saved: {debug_fallback_path}")
        
        # Detect board region
        h_orig, w_orig = img.shape[:2]
        scale_cap = 800
        scale = 1.0
        if max(h_orig, w_orig) > scale_cap:
            scale = scale_cap / max(h_orig, w_orig)
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            img_scaled = img.copy()
        
        # Use the board detection function
        pts = extract_board_region_optimized(
            img_scaled, 
            k=4,
            kmeans_sample_size=5000,
            kmeans_iterations=30,
            morph_kernel_size=15,
            debug=False,
            debug_path_prefix="/tmp/fallback_debug_"
        )
        
        if pts is not None:
            # Scale points back to original size
            pts_original = pts / scale
            
            # Save debug image with detected region
            debug_img = img.copy()
            cv2.polylines(debug_img, [pts_original.astype(int)], True, (0, 0, 255), 3)
            debug_path = "/tmp/fallback_detection_result.png"
            cv2.imwrite(debug_path, debug_img)
            print(f"Fallback detection result saved: {debug_path}")
            
            return pts_original
            
    except Exception as e:
        print(f"Error in fallback board detection: {e}")
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)
    
    return None

def extract_board_region_from_frame(frame_path, board_pts):
    """
    Extract just the board region from a frame using the detected corner points.
    Returns the warped board image.
    """
    img = cv2.imread(frame_path)
    if img is None or board_pts is None:
        return None
    
    return extract_board_region_from_frame_direct(img, board_pts)

def detect_motion_in_board_region(frame_path, prev_frame_path, board_pts, 
                                 current_timestamp, prev_timestamp,
                                 threshold=0.4,  # This is now mainly for very high motion
                                 pixel_diff_threshold=80,  # Lower for more sensitivity
                                 motion_squares_threshold=2,  # Lower default for chess
                                 max_time_gap=.5,  # Maximum seconds between frames for comparison
                                 debug=False):  # Enable debug by default
    """
    Detect motion only within the board region.
    Returns True if significant motion detected (frame should be skipped).
    """
    print(f"\n=== MOTION DETECTION DEBUG ===")
    print(f"Current frame: {os.path.basename(frame_path)} at {current_timestamp}")
    print(f"Previous frame: {os.path.basename(prev_frame_path) if prev_frame_path else 'None'} at {prev_timestamp if prev_timestamp else 'None'}")
    
    if not prev_frame_path or not os.path.exists(prev_frame_path) or board_pts is None:
        print("DECISION: No motion detection (missing prev frame or board points)")
        return False
    
    # Check time gap between frames
    if prev_timestamp:
        current_sec = timestamp_to_seconds(current_timestamp)
        prev_sec = timestamp_to_seconds(prev_timestamp)
        time_gap = current_sec - prev_sec
        print(f"Time gap: {time_gap:.1f} seconds")
        
        if time_gap > max_time_gap:
            print(f"DECISION: No motion detection (time gap {time_gap:.1f}s > {max_time_gap}s - too far apart)")
            return False
    
    # Extract board regions from both frames
    curr_board = extract_board_region_from_frame(frame_path, board_pts)
    prev_board = extract_board_region_from_frame(prev_frame_path, board_pts)
    
    if curr_board is None or prev_board is None:
        print("DECISION: No motion detection (failed to extract board regions)")
        return False
    
    print(f"Board region size: {curr_board.shape}")
    
    # Save board regions for inspection
    debug_dir = os.path.join(os.path.dirname(frame_path), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    frame_basename = os.path.basename(frame_path).replace('.png', '')
    curr_board_path = os.path.join(debug_dir, f"{frame_basename}_curr_board.png")
    prev_board_path = os.path.join(debug_dir, f"{frame_basename}_prev_board.png")
    cv2.imwrite(curr_board_path, curr_board)
    cv2.imwrite(prev_board_path, prev_board)
    
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr_board, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_board, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    # Calculate difference
    diff = cv2.absdiff(curr_gray, prev_gray)
    
    # Basic stats on the difference
    diff_mean = np.mean(diff)
    diff_max = np.max(diff)
    diff_std = np.std(diff)
    print(f"Difference stats: mean={diff_mean:.2f}, max={diff_max}, std={diff_std:.2f}")
    
    # Use higher threshold to ignore compression artifacts
    _, thresh = cv2.threshold(diff, pixel_diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate motion ratio
    motion_ratio = np.sum(thresh > 0) / thresh.size
    print(f"Motion ratio: {motion_ratio:.4f} (threshold: {threshold})")
    
    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Calculate motion ratio after morphological operations
    motion_ratio_cleaned = np.sum(thresh > 0) / thresh.size
    print(f"Motion ratio (after morphology): {motion_ratio_cleaned:.4f}")
    
    # Check for localized motion (piece movement)
    square_size = curr_board.shape[0] // 8
    motion_squares = 0
    significant_motion_squares = 0
    
    print(f"Analyzing 8x8 squares (size={square_size}x{square_size}):")
    
    for row in range(8):
        for col in range(8):
            y1, y2 = row * square_size, (row + 1) * square_size
            x1, x2 = col * square_size, (col + 1) * square_size
            square_motion = np.sum(thresh[y1:y2, x1:x2] > 0) / (square_size * square_size)
            
            if square_motion > 0.1:  # More than 10% of square changed
                motion_squares += 1
                print(f"  Square {chr(97+col)}{8-row}: {square_motion:.3f} motion")
            if square_motion > 0.3:  # More than 30% of square changed - significant
                significant_motion_squares += 1
                print(f"  Square {chr(97+col)}{8-row}: SIGNIFICANT motion ({square_motion:.3f})")
    
    print(f"Motion squares: {motion_squares} (threshold: {motion_squares_threshold})")
    print(f"Significant motion squares: {significant_motion_squares}")
    
    # Save additional debug images
    diff_path = os.path.join(debug_dir, f"{frame_basename}_diff.png")
    thresh_path = os.path.join(debug_dir, f"{frame_basename}_thresh.png")
    cv2.imwrite(diff_path, diff)
    cv2.imwrite(thresh_path, thresh)
    
    # More sophisticated decision logic - optimized for chess piece movements
    skip_reasons = []
    
    # Chess-specific motion detection rules
    if motion_ratio_cleaned > 0.7:  # Very high overall change (likely scene change)
        skip_reasons.append(f"very high motion ratio ({motion_ratio_cleaned:.3f} > 0.7)")
    
    if significant_motion_squares >= 3:  # Multiple pieces moving simultaneously
        skip_reasons.append(f"many significant changes in {significant_motion_squares} squares (>= 3)")
    
    # Key change: Detect localized chess piece movements
    if motion_squares >= motion_squares_threshold:
        # For chess, even small localized movements should be detected
        # Lower the motion ratio requirement significantly
        if motion_ratio_cleaned > 0.01:  # Much lower threshold for localized movements
            skip_reasons.append(f"localized chess movement: {motion_squares} squares with motion (>= {motion_squares_threshold}) and ratio {motion_ratio_cleaned:.4f} > 0.01")
        else:
            # If motion ratio is extremely low, it might be noise - require more squares
            if motion_squares >= motion_squares_threshold * 2:
                skip_reasons.append(f"multiple motion squares: {motion_squares} squares (>= {motion_squares_threshold * 2}) suggests piece movement")
    
    should_skip = len(skip_reasons) > 0
    
    if should_skip:
        print(f"DECISION: SKIP frame - Reasons: {'; '.join(skip_reasons)}")
    else:
        print(f"DECISION: KEEP frame - No significant motion detected")
    
    print("=" * 40)
    
    return should_skip

def calculate_board_hash(board_img, hash_size=32):
    """
    Calculate perceptual hash of the board position.
    Uses higher resolution hash for better chess position sensitivity.
    """
    if board_img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur to reduce noise but preserve piece shapes
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Resize to hash size (using larger hash for better sensitivity)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Calculate horizontal gradient
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert to hash
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes."""
    return bin(hash1 ^ hash2).count('1')

def extract_frame_with_metadata(video_path, ts, imgs_dir, board_pts, use_gpu=False):
    """
    Extract a frame and compute its board hash for deduplication.
    """
    frame_path = extract_frame(video_path, ts, imgs_dir, use_gpu)
    
    # Extract board region and calculate hash
    board_img = extract_board_region_from_frame(frame_path, board_pts)
    board_hash = calculate_board_hash(board_img) if board_img is not None else None
    
    return {
        'timestamp': ts,
        'path': frame_path,
        'board_hash': board_hash,
        'board_img': board_img
    }

def process_file_with_board_detection(srt_path, use_gpu=False, 
                                    extract_interval=1.0,
                                    motion_threshold=0.4,  # Higher default
                                    motion_squares_threshold=2,  # Lower for chess
                                    pixel_diff_threshold=40,  # Lower for sensitivity
                                    hash_distance_threshold=15,  # Much lower for chess positions
                                    max_duration_minutes=180,
                                    enable_motion_detection=True,  # Option to disable
                                    enable_deduplication=True,  # Option to disable
                                    force_fallback_detection=False,
                                    detection_times=[10.0, 30.0, 60.0, 90.0]):
    """
    Enhanced processing that uses board detection for motion and deduplication.
    Limited to first max_duration_minutes of video.
    """
    # 1) Clean the .srt
    clean_srt(srt_path)
    
    # 2) Find video
    video = find_video_for_srt(srt_path)
    if not video:
        return
    
    # 3) Detect board region with advanced method
    print("Detecting chess board region...")
    if force_fallback_detection:
        print("Using fallback detection method (forced)")
        board_pts = detect_board_region_fallback(video, detection_times[0])
    else:
        print("Using advanced detection method")
        board_pts = detect_board_region_advanced(video, detection_times)
        if board_pts is None:
            print("Warning: Could not detect board region with advanced method. Trying fallback...")
            board_pts = detect_board_region_fallback(video, detection_times[0])
    
    if board_pts is None:
        print("Error: Could not detect board region with any method.")
        print("Debug files should be available in /tmp/ for inspection")
        print("Continuing with processing but motion detection will be disabled...")
    else:
        print("Board region detected successfully.")
    
    # 4) Create output directory
    imgs_dir = os.path.join(os.path.dirname(srt_path), 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)
    
    # 5) Extract timestamps at intervals (LIMITED TO FIRST N MINUTES)
    max_seconds = max_duration_minutes * 60
    print(f"Processing only first {max_duration_minutes} minutes ({max_seconds} seconds) of video")
    timestamps = extract_timestamps_at_intervals(srt_path, extract_interval, max_seconds)
    
    # 6) Extract frames with board region data
    print(f"Extracting {len(timestamps)} frames...")
    frame_data = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futures = {
            pool.submit(extract_frame_with_metadata, video, ts, imgs_dir, board_pts, use_gpu): ts
            for ts in timestamps
        }
        
        for fut in as_completed(futures):
            try:
                data = fut.result()
                frame_data.append(data)
            except Exception as e:
                print(f"Failed to extract frame: {e}")
    
    # Sort by timestamp
    frame_data.sort(key=lambda x: timestamp_to_seconds(x['timestamp']))
    
    # 7) Filter out motion frames (only in board region)
    if enable_motion_detection and board_pts is not None:
        print("Filtering frames with motion in board region...")
        stable_frames = []
        prev_frame = None
        motion_skipped = 0
        
        for i, frame in enumerate(frame_data):
            print(f"\nProcessing frame {i+1}/{len(frame_data)}: {frame['timestamp']}")
            
            if prev_frame and board_pts is not None:
                if detect_motion_in_board_region(
                    frame['path'], 
                    prev_frame['path'], 
                    board_pts, 
                    frame['timestamp'],
                    prev_frame['timestamp'],
                    motion_threshold,
                    pixel_diff_threshold,
                    motion_squares_threshold
                ):
                    print(f"✗ Skipped frame at {frame['timestamp']} due to board motion")
                    motion_skipped += 1
                    os.remove(frame['path'])
                    continue
            
            print(f"✓ Kept frame at {frame['timestamp']}")
            stable_frames.append(frame)
            prev_frame = frame
        
        print(f"\nMotion filtering results: {motion_skipped} frames skipped, {len(stable_frames)} frames kept")
    elif board_pts is None:
        print("Warning: No board region detected - skipping motion detection")
        stable_frames = frame_data
        print(f"Keeping all {len(stable_frames)} frames (no board region for motion detection)")
    else:
        print("Motion detection disabled - keeping all frames")
        stable_frames = frame_data
    
    # 8) Remove duplicate positions based on board hash
    if enable_deduplication:
        print("Removing duplicate board positions...")
        unique_frames = []
        seen_hashes = {}
        duplicate_skipped = 0
        
        print(f"Using hash distance threshold: {hash_distance_threshold}")
        
        for i, frame in enumerate(stable_frames):
            print(f"\nAnalyzing frame {i+1}/{len(stable_frames)}: {frame['timestamp']}")
            
            if frame['board_hash'] is None:
                print("  No board hash - keeping frame")
                unique_frames.append(frame)
                continue
            
            print(f"  Board hash: {frame['board_hash']}")
            
            # Check if similar position already exists
            is_duplicate = False
            min_distance = float('inf')
            closest_frame = None
            
            for seen_hash, seen_frame in seen_hashes.items():
                distance = hamming_distance(frame['board_hash'], seen_hash)
                if distance < min_distance:
                    min_distance = distance
                    closest_frame = seen_frame
                
                print(f"    Distance to {seen_frame['timestamp']}: {distance}")
                
                if distance < hash_distance_threshold:
                    is_duplicate = True
                    print(f"  ✗ DUPLICATE: Distance {distance} < threshold {hash_distance_threshold}")
                    print(f"    Similar to frame at {seen_frame['timestamp']}")
                    duplicate_skipped += 1
                    os.remove(frame['path'])
                    break
            
            if not is_duplicate:
                print(f"  ✓ UNIQUE: Minimum distance {min_distance} >= threshold {hash_distance_threshold}")
                seen_hashes[frame['board_hash']] = frame
                unique_frames.append(frame)
        
        print(f"\nDuplication filtering results: {duplicate_skipped} duplicates removed")
    else:
        print("Deduplication disabled - keeping all frames")
        unique_frames = stable_frames
        duplicate_skipped = 0
    print(f"Final frame count: {len(unique_frames)} (from {len(timestamps)} extracted)")
    
    # 9) Save board region info for video_processor
    board_info_path = os.path.join(os.path.dirname(srt_path), 'board_region.json')
    if board_pts is not None:
        board_info = {
            'corner_points': board_pts.tolist(),
            'detected_at_timestamp': timestamps[0] if timestamps else "00:00:10,000",
            'detection_method': 'advanced' if 'advanced' in str(board_pts) else 'standard'
        }
        with open(board_info_path, 'w') as f:
            json.dump(board_info, f, indent=2)
        print(f"Saved board region info to {board_info_path}")
    else:
        # Save info indicating no board region was detected
        board_info = {
            'corner_points': None,
            'detected_at_timestamp': None,
            'detection_method': 'failed',
            'note': 'No chess board region could be detected in the video'
        }
        with open(board_info_path, 'w') as f:
            json.dump(board_info, f, indent=2)
        print(f"Saved board detection failure info to {board_info_path}")
        print("Warning: Processing completed but no board region was detected.")

def extract_timestamps_at_intervals(srt_path, interval=1.0, max_seconds=None):
    """Extract timestamps at regular intervals throughout subtitle blocks, limited by max_seconds."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    time_ranges = re.findall(pattern, content)
    
    timestamps = []
    
    for start_ts, end_ts in time_ranges:
        start_sec = timestamp_to_seconds(start_ts)
        end_sec = timestamp_to_seconds(end_ts)
        
        # Skip if beyond max duration
        if max_seconds and start_sec > max_seconds:
            continue
        
        # Limit end time to max duration
        if max_seconds and end_sec > max_seconds:
            end_sec = max_seconds
        
        # Add before subtitle (0.5s before)
        pre_time = max(0, start_sec - 0.5)
        if not max_seconds or pre_time <= max_seconds:
            timestamps.append(seconds_to_timestamp(pre_time))
        
        # Add during subtitle at intervals
        current = start_sec
        while current <= end_sec and (not max_seconds or current <= max_seconds):
            timestamps.append(seconds_to_timestamp(current))
            current += interval
        
        # Add after subtitle (0.5s after)
        post_time = end_sec + 0.5
        if not max_seconds or post_time <= max_seconds:
            timestamps.append(seconds_to_timestamp(post_time))
    
    # Remove duplicates and sort
    filtered_timestamps = sorted(list(set(timestamps)))
    
    # Additional filter to ensure we don't exceed max_seconds
    if max_seconds:
        filtered_timestamps = [ts for ts in filtered_timestamps if timestamp_to_seconds(ts) <= max_seconds]
    
    print(f"Generated {len(filtered_timestamps)} timestamps (max duration: {max_seconds}s)")
    return filtered_timestamps

def seconds_to_timestamp(seconds):
    """Convert seconds to timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def clean_srt(path):
    """
    Remove subtitle blocks whose text is exactly '[music]'.
    Overwrites the original file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    cleaned = []
    for block in blocks:
        lines = block.splitlines()
        text = ' '.join(lines[2:]).strip().lower() if len(lines) >= 3 else ''
        if text == '[music]':
            continue
        cleaned.append(block)

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned) + "\n")

    print(f"Cleaned: {path}")
    return path

def find_video_for_srt(srt_path):
    dirpath = os.path.dirname(srt_path)
    base = os.path.splitext(os.path.basename(srt_path))[0]
    exts = ['.mp4', '.mkv', '.webm', '.mov', '.flv']
    for ext in exts:
        candidate = os.path.join(dirpath, base + ext)
        if os.path.isfile(candidate):
            return candidate
    for file in os.listdir(dirpath):
        if file.lower().endswith(tuple(exts)):
            return os.path.join(dirpath, file)
    print(f"Warning: no video file found for transcript {srt_path}")
    return None

def timestamp_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(':')
    if ',' in rest:
        s, ms = rest.split(',')
    else:
        s, ms = rest, '0'
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def extract_frame(video_path, ts, imgs_dir, use_gpu=False):
    """
    Worker to extract a single frame at timestamp ts from video_path into imgs_dir.
    """
    sec = timestamp_to_seconds(ts)
    safe_ts = ts.replace(':', '-').replace(',', '.')
    img_path = os.path.join(imgs_dir, f"{safe_ts}.png")

    # Base ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',                # overwrite without asking
        '-hide_banner',
        '-loglevel', 'error'
    ]

    if use_gpu:
        # Decode on GPU into CUDA memory
        cmd += [
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda'
        ]

    # Seek directly to the timestamp
    cmd += ['-ss', str(sec), '-i', video_path]

    if use_gpu:
        # Download the frame back to system memory as yuv420p before encoding PNG
        cmd += [
            '-vf', 'hwdownload,format=yuv420p',
            '-frames:v', '1',
            img_path
        ]
    else:
        # CPU-only extraction
        cmd += ['-frames:v', '1', img_path]

    subprocess.run(cmd, check=True)
    return img_path

def extract_frames(srt_path, video_path, imgs_dir, use_gpu=False):
    os.makedirs(imgs_dir, exist_ok=True)
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    starts = re.findall(r"^(\d{2}:\d{2}:\d{2},\d{3}) -->", content,
                        flags=re.MULTILINE)
    if not starts:
        print(f"No timestamps found in {srt_path}")
        return

    workers = min(len(starts), os.cpu_count() or 4)
    print(f"Extracting {len(starts)} frames using {workers} workers"
          f" (GPU={'yes' if use_gpu else 'no'})...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_frame, video_path, ts, imgs_dir, use_gpu): ts
            for ts in starts
        }
        for fut in as_completed(futures):
            ts = futures[fut]
            try:
                img = fut.result()
                print(f"Extracted {img}")
            except Exception as e:
                print(f"Failed timestamp {ts}: {e}")

def process_file(srt_path, use_gpu=False):
    # 1) clean the .srt
    clean_srt(srt_path)
    # 2) find its video
    video = find_video_for_srt(srt_path)
    if not video:
        return
    # 3) extract frames
    imgs_dir = os.path.join(os.path.dirname(srt_path), 'imgs')
    extract_frames(srt_path, video, imgs_dir, use_gpu)

def main():
    parser = argparse.ArgumentParser(
        description="Clean SRTs, extract frames with board detection"
    )
    parser.add_argument('root_dir', help='Parent directory containing transcript/video subfolders')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--interval', type=float, default=1.0, help='Frame extraction interval in seconds')
    parser.add_argument('--motion-threshold', type=float, default=0.4, help='Overall motion detection threshold')
    parser.add_argument('--motion-squares-threshold', type=int, default=2, help='Number of squares with motion to trigger detection')
    parser.add_argument('--pixel-diff-threshold', type=int, default=40, help='Pixel difference threshold for motion detection')
    parser.add_argument('--hash-threshold', type=int, default=15, help='Hash distance threshold for duplicates (lower=more strict)')
    parser.add_argument('--max-minutes', type=int, default=180, help='Maximum duration to process (minutes)')
    parser.add_argument('--disable-motion', action='store_true', help='Disable motion detection (keep all frames)')
    parser.add_argument('--disable-dedup', action='store_true', help='Disable duplicate detection (keep all unique positions)')
    parser.add_argument('--force-fallback-detection', action='store_true', help='Force use of fallback board detection method')
    parser.add_argument('--detection-times', nargs='+', type=float, default=[10.0, 30.0, 60.0, 90.0], 
                       help='Timestamps (in seconds) to try for board detection')
    
    args = parser.parse_args()
    
    # 0) First, REMOVE any folder that has video(s) but NO .srt
    video_exts = ('.mp4', '.mkv', '.webm', '.mov', '.flv')
    for dirpath, dirs, files in os.walk(args.root_dir, topdown=False):
        if dirpath == args.root_dir:
            continue
        has_video = any(f.lower().endswith(video_exts) for f in files)
        has_srt   = any(f.lower().endswith('.srt')       for f in files)
        if has_video and not has_srt:
            print(f"→ No transcript in '{dirpath}' → deleting entire folder.")
            shutil.rmtree(dirpath)
    
    all_srts = []
    for dirpath, _, files in os.walk(args.root_dir):
        for fname in files:
            if fname.lower().endswith('.srt'):
                all_srts.append(os.path.join(dirpath, fname))

    if not all_srts:
        print(f"No .srt files found under {args.root_dir}. Exiting.")
        return

    # Process each .srt with board detection
    for srt in sorted(all_srts):
        print(f"\nProcessing: {srt}")
        try:
            process_file_with_board_detection(
                srt, 
                use_gpu=args.gpu,
                extract_interval=args.interval,
                motion_threshold=args.motion_threshold,
                motion_squares_threshold=args.motion_squares_threshold,
                pixel_diff_threshold=args.pixel_diff_threshold,
                hash_distance_threshold=args.hash_threshold,
                max_duration_minutes=args.max_minutes,
                enable_motion_detection=not args.disable_motion,
                enable_deduplication=not args.disable_dedup,
                force_fallback_detection=args.force_fallback_detection,
                detection_times=args.detection_times
            )
        except Exception as e:
            print(f"Error with {srt}: {e}")

if __name__ == '__main__':
    main()