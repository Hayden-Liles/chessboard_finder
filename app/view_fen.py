import tkinter as tk
from tkinter import ttk
import chess
import chess.svg
from PIL import Image, ImageTk
import io
import json
import argparse
import sys
import os
import glob
import re

class ChessFENViewer:
    def __init__(self, fen_positions, image_folder=None):
        self.fen_positions = fen_positions
        self.image_folder = image_folder
        self.current_index = 0
        self.images = {}
        self.pil_images = {}  # Store PIL images separately
        
        # Create main window first
        self.root = tk.Tk()
        self.root.title("Chess FEN Position Viewer with Images")
        self.root.geometry("1400x900")
        self.root.configure(bg='white')
        
        # Make window focusable for key events
        self.root.focus_set()
        
        # Create widgets
        self.setup_widgets()
        
        # Load images after window is created
        if self.image_folder:
            self.load_images()
        
        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)
        
        # Display first position
        self.update_display()
    
    def load_images(self):
        """Load all images from the specified folder"""
        if not os.path.exists(self.image_folder):
            print(f"Warning: Image folder '{self.image_folder}' does not exist.")
            return
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))
        
        print(f"Found {len(image_files)} images in {self.image_folder}")
        
        # Sort images by filename (which should be timestamp-based)
        image_files.sort()
        
        # Load PIL images (don't convert to ImageTk yet)
        for i, image_path in enumerate(image_files):
            try:
                # Extract timestamp from filename for matching
                filename = os.path.basename(image_path)
                timestamp = self.extract_timestamp(filename)
                
                # Load PIL image
                pil_image = Image.open(image_path)
                # Resize to fit display area while maintaining aspect ratio
                pil_image.thumbnail((500, 400), Image.Resampling.LANCZOS)
                
                self.pil_images[i] = {
                    'pil_image': pil_image,
                    'timestamp': timestamp,
                    'filename': filename,
                    'path': image_path
                }
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        
        print(f"Successfully loaded {len(self.pil_images)} images")
    
    def extract_timestamp(self, filename):
        """Extract timestamp from filename like '00-00-07.059.png'"""
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Try to extract timestamp pattern (HH-MM-SS.mmm)
        timestamp_pattern = r'(\d{2})-(\d{2})-(\d{2})\.(\d{3})'
        match = re.search(timestamp_pattern, name_without_ext)
        
        if match:
            hours, minutes, seconds, milliseconds = match.groups()
            # Convert to total milliseconds for easier comparison
            total_ms = int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(milliseconds)
            return total_ms
        
        return None
    
    def setup_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info frame
        info_frame = tk.Frame(main_frame, bg='white')
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Move number label
        self.move_label = tk.Label(info_frame, text="", font=('Arial', 12, 'bold'), bg='white')
        self.move_label.pack()
        
        # FEN label
        self.fen_label = tk.Label(info_frame, text="", font=('Arial', 10), bg='white', wraplength=1300)
        self.fen_label.pack(pady=5)
        
        # Navigation info
        nav_label = tk.Label(info_frame, text="Use ← → arrow keys to navigate", 
                           font=('Arial', 10), bg='white', fg='gray')
        nav_label.pack(pady=5)
        
        # Content frame for board and image
        content_frame = tk.Frame(main_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chess board frame (left side)
        board_container = tk.Frame(content_frame, bg='white')
        board_container.pack(side=tk.LEFT, padx=(0, 20))
        
        board_title = tk.Label(board_container, text="Chess Position", font=('Arial', 12, 'bold'), bg='white')
        board_title.pack(pady=(0, 10))
        
        # Canvas for chess board
        self.canvas = tk.Canvas(board_container, width=600, height=600, bg='white')
        self.canvas.pack()
        
        # Image frame (right side)
        if self.image_folder:
            image_container = tk.Frame(content_frame, bg='white')
            image_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            image_title = tk.Label(image_container, text="Original Image", font=('Arial', 12, 'bold'), bg='white')
            image_title.pack(pady=(0, 10))
            
            # Image display
            self.image_label = tk.Label(image_container, bg='white', text="No image available")
            self.image_label.pack()
            
            # Image info
            self.image_info_label = tk.Label(image_container, text="", font=('Arial', 9), bg='white', fg='gray')
            self.image_info_label.pack(pady=(10, 0))
    
    def draw_board(self, fen):
        """Draw chess board from FEN string"""
        try:
            board = chess.Board(fen)
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Board dimensions
            square_size = 75
            board_size = 8 * square_size
            
            # Colors
            light_color = "#F0D9B5"
            dark_color = "#B58863"
            
            # Draw squares and pieces
            for rank in range(8):
                for file in range(8):
                    x1 = file * square_size
                    y1 = rank * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    
                    # Square color
                    color = light_color if (rank + file) % 2 == 0 else dark_color
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                    
                    # Get piece at this square
                    square = chess.square(file, 7-rank)  # chess library uses different coordinate system
                    piece = board.piece_at(square)
                    
                    if piece:
                        # Draw piece symbol
                        piece_symbol = self.get_piece_symbol(piece)
                        self.canvas.create_text(x1 + square_size//2, y1 + square_size//2, 
                                              text=piece_symbol, font=('Arial', 36), fill='black')
            
            # Draw board border
            self.canvas.create_rectangle(0, 0, board_size, board_size, outline="black", width=2)
            
            # Draw coordinates
            for i in range(8):
                # Files (a-h)
                file_letter = chr(ord('a') + i)
                self.canvas.create_text(i * square_size + square_size//2, board_size + 15, 
                                      text=file_letter, font=('Arial', 12))
                
                # Ranks (1-8)
                rank_number = str(8 - i)
                self.canvas.create_text(-15, i * square_size + square_size//2, 
                                      text=rank_number, font=('Arial', 12))
        
        except Exception as e:
            # If FEN is invalid, show error
            self.canvas.delete("all")
            self.canvas.create_text(300, 300, text=f"Invalid FEN: {str(e)}", 
                                  font=('Arial', 16), fill='red')
    
    def get_piece_symbol(self, piece):
        """Convert chess piece to Unicode symbol"""
        symbols = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',  # White
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'   # Black
        }
        return symbols.get(piece.symbol(), piece.symbol())
    
    def update_display(self):
        """Update the display with current position"""
        if 0 <= self.current_index < len(self.fen_positions):
            current_data = self.fen_positions[self.current_index]
            
            # Handle different data formats
            if isinstance(current_data, dict):
                fen = current_data.get('fen', '')
                move_num = current_data.get('move_num', self.current_index + 1)
                timestamp = current_data.get('timestamp', None)
            elif isinstance(current_data, (list, tuple)) and len(current_data) >= 2:
                fen = current_data[0]
                move_num = current_data[1]
                timestamp = current_data[2] if len(current_data) > 2 else None
            else:
                fen = str(current_data)
                move_num = self.current_index + 1
                timestamp = None
            
            # Update labels
            self.move_label.config(text=f"Position {self.current_index + 1}/{len(self.fen_positions)} - Move: {move_num}")
            self.fen_label.config(text=f"FEN: {fen}")
            
            # Draw board
            self.draw_board(fen)
            
            # Update image if available
            if self.image_folder and hasattr(self, 'image_label'):
                self.update_image_display(timestamp)
    
    def update_image_display(self, timestamp=None):
        """Update the image display"""
        # Try to find matching image
        image_index = None
        
        if timestamp:
            # If we have timestamp info, try to find closest match
            image_index = self.find_closest_image_by_timestamp(timestamp)
        else:
            # If no timestamp, assume images are in same order as positions
            if self.current_index < len(self.pil_images):
                image_index = self.current_index
        
        if image_index is not None and image_index in self.pil_images:
            image_data = self.pil_images[image_index]
            
            # Convert PIL image to ImageTk on demand (cache by index)
            if image_index not in self.images:
                try:
                    tk_image = ImageTk.PhotoImage(image_data['pil_image'])
                    self.images[image_index] = tk_image
                except Exception as e:
                    print(f"Error converting image to Tkinter format: {e}")
                    self.image_label.config(image="", text="Error loading image")
                    self.image_info_label.config(text="")
                    return
            
            tk_image = self.images[image_index]
            self.image_label.config(image=tk_image, text="")
            self.image_info_label.config(text=f"File: {image_data['filename']}")
        else:
            # Show placeholder
            self.image_label.config(image="", text="No matching image found")
            self.image_info_label.config(text="")
    
    def find_closest_image_by_timestamp(self, target_timestamp):
        """Find the image with timestamp closest to target"""
        if not self.pil_images:
            return None
        
        closest_index = None
        min_diff = float('inf')
        
        for index, image_data in self.pil_images.items():
            if image_data['timestamp'] is not None:
                diff = abs(image_data['timestamp'] - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_index = index
        
        return closest_index
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.keysym == 'Right':
            if self.current_index < len(self.fen_positions) - 1:
                self.current_index += 1
                self.update_display()
        elif event.keysym == 'Left':
            if self.current_index > 0:
                self.current_index -= 1
                self.update_display()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def load_positions_from_json(file_path):
    """Load FEN positions from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common key names for the positions array
            for key in ['positions', 'fen_positions', 'moves', 'game', 'data']:
                if key in data:
                    return data[key]
            # If no common key found, return the whole dict as single position
            return [data]
        else:
            print(f"Error: Unexpected JSON structure in {file_path}")
            return []
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        return []
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Chess FEN Position Viewer with Images')
    parser.add_argument('json_file', 
                       help='Path to JSON file containing FEN positions')
    parser.add_argument('--images', '-i', 
                       help='Path to folder containing images')
    parser.add_argument('--key', '-k', 
                       help='Specific key in JSON to use for positions (optional)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File '{args.json_file}' does not exist.")
        sys.exit(1)
    
    # Load positions
    positions = load_positions_from_json(args.json_file)
    
    # If specific key was provided, try to use it
    if args.key:
        try:
            with open(args.json_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and args.key in data:
                positions = data[args.key]
            else:
                print(f"Warning: Key '{args.key}' not found in JSON file. Using auto-detection.")
        except Exception as e:
            print(f"Warning: Could not use specified key '{args.key}': {e}")
    
    if not positions:
        print("Error: No positions found in the JSON file.")
        print("\nExpected JSON formats:")
        print("1. Array of objects: [{'fen': '...', 'move_num': 1, 'timestamp': 123456}, ...]")
        print("2. Array of arrays: [['fen_string', move_num, timestamp], ...]")
        print("3. Array of strings: ['fen_string1', 'fen_string2', ...]")
        print("4. Object with positions array: {'positions': [...], 'other_data': '...'}")
        sys.exit(1)
    
    print(f"Loaded {len(positions)} positions from {args.json_file}")
    if args.images:
        print(f"Using images from: {args.images}")
    print("Use ← → arrow keys to navigate between positions")
    print("Close the window or press Ctrl+C to exit")
    
    try:
        viewer = ChessFENViewer(positions, args.images)
        viewer.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()