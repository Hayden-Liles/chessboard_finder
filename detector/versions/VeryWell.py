#!/usr/bin/env python3
"""
Robust digital chessboard grid detection via color segmentation and rotated-rect extraction.

Approach:
 1. Optionally downscale large images for speed.
 2. Sample random pixels and run k-means to segment into k color clusters.
 3. Pick the 2 largest clusters (by mask area) as the board square colors.
 4. Morphologically clean and extract the largest connected region ⇒ board mask.
 5. Fit a rotated rectangle (`minAreaRect`) around that region ⇒ 4 board corners.
 6. Warp to a square, draw an 8×8 grid, inverse-warp overlay back onto original.

Usage:
    python detect_board.py path/to/image.png --clusters 4 --downscale 800 --debug
"""
import cv2
import numpy as np
import argparse
from itertools import combinations


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def extract_board_region(img, k=4, sample_size=10000, debug=False):
    h, w = img.shape[:2]

    # 1) run exactly as before: sample pixels → k-means → full_labels
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
    best_box   = None
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
            best_box   = cv2.boxPoints(rect).astype(np.float32)

            if debug:
                cv2.imwrite(f'debug_mask_{i}_{j}.png', mask)

    if best_box is None:
        return None

    if debug:
        vis = img.copy()
        for x,y in best_box:
            cv2.circle(vis, (int(x),int(y)), 10, (0,0,255), -1)
        cv2.imwrite('corners_color.png', vis)

    return best_box


def warp_and_draw(img, pts, size=800, thickness=2):
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
    return result


def main():
    parser = argparse.ArgumentParser(description='Chessboard grid by color segmentation.')
    parser.add_argument('image', help='Path to image')
    parser.add_argument('--clusters', type=int, default=4, help='k for k-means')
    parser.add_argument('--downscale', type=int, default=0, help='Max size to downscale longest side')
    parser.add_argument('--debug', action='store_true', help='Dump debug masks and corners')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error loading {args.image}")
        return

    # optional downscale
    if args.downscale > 0:
        h,w = img.shape[:2]
        scale = args.downscale / max(h,w)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    pts = extract_board_region(img, k=args.clusters, debug=args.debug)
    if pts is None:
        print("Failed to find board region via color clustering.")
        return

    out = warp_and_draw(img, pts)
    cv2.imwrite('board_grid.png', out)
    print('Saved board_grid.png')

if __name__ == '__main__':
    main()
