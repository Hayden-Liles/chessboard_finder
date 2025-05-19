import cv2
import numpy as np
import os
import argparse

# Define a custom preprocess_input function since we may not have TensorFlow installed
def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    This function implements the same preprocessing as MobileNetV2:
    - Scales values from [0, 255] to [-1, 1]
    """
    return x / 127.5 - 1.0

"""
Debug script to extract and visualize the exact images fed into the Keras model.
For each square of a warped chessboard image, this script:
 1. Saves the raw BGR square as orig_row_col.png
 2. Applies model preprocessing (resize, BGR->RGB, preprocess_input)
    and saves a visualization of that processed input as proc_row_col.png

Usage:
    python diagnose_model.py path/to/warped_board.png --size 100 --output debug_squares
"""

def main():
    parser = argparse.ArgumentParser(
        description="Extract and visualize model inputs for each chess square"
    )
    parser.add_argument('warped_image', help='Path to the warped chessboard image')
    parser.add_argument('--size', type=int, default=100,
                        help='Input size expected by the model (default: 100)')
    parser.add_argument('--output', default='debug_squares',
                        help='Directory to save debug images')
    args = parser.parse_args()

    # Load the warped image
    img = cv2.imread(args.warped_image)
    if img is None:
        print(f"Error: Could not load image {args.warped_image}")
        return

    h = img.shape[0]
    square_size = h // 8

    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)

    # Loop through each square
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            sq = img[y1:y1+square_size, x1:x1+square_size]

            # Save original square
            orig_path = os.path.join(args.output, f'orig_{row}_{col}.png')
            cv2.imwrite(orig_path, sq)

            # Preprocess for model
            proc = cv2.resize(sq, (args.size, args.size))
            proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32)
            proc_input = preprocess_input(proc_rgb)

            # Convert back to BGR uint8 for visualization
            # MobilenetV2 preprocess_input scales to [-1,1], so map back to [0,255]
            vis = ((proc_input + 1.0) * 127.5).astype(np.uint8)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            proc_path = os.path.join(args.output, f'proc_{row}_{col}.png')
            cv2.imwrite(proc_path, vis_bgr)

    print(f"Saved debug images to '{args.output}'")

if __name__ == '__main__':
    main()