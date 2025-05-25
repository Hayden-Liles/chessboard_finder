#!/usr/bin/env python3
"""
Tool to detect and fix mid-move positions in FEN sequences.
These occur when video frames are captured while pieces are being physically moved.
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Set

try:
    import chess
except ImportError:
    print("Error: python-chess library is required. Install it with: pip install python-chess")
    sys.exit(1)

def count_pieces(board_fen: str) -> Dict[str, int]:
    """
    Count pieces on the board from a FEN position.
    Returns a dictionary with piece counts.
    """
    piece_counts = {}
    for char in board_fen:
        if char.isalpha():
            piece_counts[char] = piece_counts.get(char, 0) + 1
    return piece_counts

def get_piece_positions(board_fen: str) -> Dict[str, Set[str]]:
    """
    Get positions of all pieces on the board.
    Returns a dictionary mapping piece types to sets of squares.
    """
    piece_positions = {}
    ranks = board_fen.split('/')
    
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                square = chr(ord('a') + file_idx) + str(8 - rank_idx)
                if char not in piece_positions:
                    piece_positions[char] = set()
                piece_positions[char].add(square)
                file_idx += 1
    
    return piece_positions

def detect_mid_move_position(prev_fen: str, curr_fen: str, next_fen: Optional[str] = None) -> Tuple[bool, str]:
    """
    Detect if curr_fen represents a mid-move position.
    Returns (is_mid_move, description).
    """
    try:
        prev_board = chess.Board(prev_fen)
        curr_board = chess.Board(curr_fen)
    except:
        return False, "Invalid FEN"
    
    prev_board_fen = prev_board.board_fen()
    curr_board_fen = curr_board.board_fen()
    
    prev_counts = count_pieces(prev_board_fen)
    curr_counts = count_pieces(curr_board_fen)
    
    # Check for pieces that disappeared
    disappeared_pieces = []
    for piece, count in prev_counts.items():
        curr_count = curr_counts.get(piece, 0)
        if curr_count < count:
            disappeared_pieces.append((piece, count - curr_count))
    
    # Check for pieces that appeared
    appeared_pieces = []
    for piece, count in curr_counts.items():
        prev_count = prev_counts.get(piece, 0)
        if count > prev_count:
            appeared_pieces.append((piece, count - prev_count))
    
    # Mid-move detection criteria
    if disappeared_pieces and not appeared_pieces:
        # Piece picked up but not placed
        if len(disappeared_pieces) == 1 and disappeared_pieces[0][1] == 1:
            piece_type = disappeared_pieces[0][0]
            
            # If we have the next position, check if the piece appears there
            if next_fen:
                try:
                    next_board = chess.Board(next_fen)
                    next_board_fen = next_board.board_fen()
                    next_counts = count_pieces(next_board_fen)
                    
                    # Check if the disappeared piece reappears
                    if next_counts.get(piece_type, 0) == prev_counts.get(piece_type, 0):
                        return True, f"Mid-move: {piece_type} picked up but not placed"
                except:
                    pass
            else:
                return True, f"Mid-move: {piece_type} picked up but not placed"
    
    elif appeared_pieces and not disappeared_pieces:
        # Piece appeared without being picked up (rare but possible)
        if len(appeared_pieces) == 1 and appeared_pieces[0][1] == 1:
            return True, f"Mid-move: {appeared_pieces[0][0]} appeared without origin"
    
    # Check for impossible captures (piece disappeared but no opponent piece captured)
    if disappeared_pieces:
        # In a normal capture, one piece disappears and is replaced by an opponent piece
        # If multiple pieces disappear or pieces of both colors disappear, it's likely mid-move
        white_disappeared = sum(1 for p, _ in disappeared_pieces if p.isupper())
        black_disappeared = sum(1 for p, _ in disappeared_pieces if p.islower())
        
        if white_disappeared > 0 and black_disappeared > 0:
            return True, "Mid-move: Pieces of both colors disappeared"
    
    return False, "Normal position"

def fix_mid_move_positions(fen_sequence: List[str]) -> Tuple[List[str], List[str], List[int]]:
    """
    Fix mid-move positions in a FEN sequence.
    Returns (fixed_sequence, messages, removed_indices).
    """
    if len(fen_sequence) < 2:
        return fen_sequence, [], []
    
    fixed_sequence = []
    messages = []
    removed_indices = []
    
    # Always keep the first position
    fixed_sequence.append(fen_sequence[0])
    
    i = 1
    while i < len(fen_sequence):
        prev_fen = fixed_sequence[-1]  # Use the last kept position
        curr_fen = fen_sequence[i]
        next_fen = fen_sequence[i + 1] if i + 1 < len(fen_sequence) else None
        
        is_mid_move, description = detect_mid_move_position(prev_fen, curr_fen, next_fen)
        
        if is_mid_move:
            messages.append(f"Position {i+1}: {description} - Removing")
            removed_indices.append(i)
            # Skip this position
        else:
            # Check if this is a valid move from the previous position
            try:
                prev_board = chess.Board(prev_fen)
                curr_board = chess.Board(curr_fen)
                
                # Check if there's a legal move between positions
                move_found = False
                for move in prev_board.legal_moves:
                    test_board = prev_board.copy()
                    test_board.push(move)
                    if test_board.board_fen() == curr_board.board_fen():
                        move_found = True
                        break
                
                if not move_found and i + 1 < len(fen_sequence):
                    # Maybe current + next form a complete move
                    next_board = chess.Board(fen_sequence[i + 1])
                    for move in prev_board.legal_moves:
                        test_board = prev_board.copy()
                        test_board.push(move)
                        if test_board.board_fen() == next_board.board_fen():
                            # Skip current and use next
                            messages.append(f"Position {i+1}: Intermediate position - Skipping")
                            removed_indices.append(i)
                            i += 1
                            continue
                
                fixed_sequence.append(curr_fen)
                
            except Exception as e:
                # Keep the position if we can't analyze it
                fixed_sequence.append(curr_fen)
                messages.append(f"Position {i+1}: Could not analyze ({e})")
        
        i += 1
    
    return fixed_sequence, messages, removed_indices

def clean_game_positions(game_data: Dict, verbose: bool = True) -> Tuple[int, int]:
    """
    Clean mid-move positions from a game.
    Returns (original_count, cleaned_count).
    """
    analysis_points = game_data.get("stockfish_analysis_points", [])
    if not analysis_points:
        return 0, 0
    
    # Extract FEN sequence and metadata
    original_data = []
    for point in analysis_points:
        if point.get("fen"):
            original_data.append({
                'fen': point.get("fen", ""),
                'metadata': {k: v for k, v in point.items() if k != 'fen'}
            })
    
    if not original_data:
        return 0, 0
    
    original_fens = [item['fen'] for item in original_data]
    original_count = len(original_fens)
    
    if verbose:
        print(f"    Original positions: {original_count}")
    
    # Fix mid-move positions
    fixed_fens, messages, removed_indices = fix_mid_move_positions(original_fens)
    
    if verbose and messages:
        print(f"    Mid-move detection:")
        for msg in messages[:5]:
            print(f"      {msg}")
        if len(messages) > 5:
            print(f"      ... and {len(messages) - 5} more")
    
    # Create new analysis points
    new_points = []
    kept_indices = [i for i in range(len(original_data)) if i not in removed_indices]
    
    for new_idx, old_idx in enumerate(kept_indices):
        point = original_data[old_idx]
        new_point = {
            "fen": point['fen'],
            "move_num": new_idx + 1,
            **point['metadata']
        }
        new_points.append(new_point)
    
    game_data["stockfish_analysis_points"] = new_points
    cleaned_count = len(new_points)
    
    if verbose:
        print(f"    Cleaned positions: {cleaned_count} (removed {original_count - cleaned_count})")
    
    return original_count, cleaned_count

def process_compiled_game_data(file_path: str, dry_run: bool = False) -> bool:
    """
    Process a compiled_game_data.json file to clean mid-move positions.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            games_data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    if not games_data:
        print("No games found in file")
        return False
    
    print(f"Processing {len(games_data)} games for mid-move positions...")
    
    total_original = 0
    total_cleaned = 0
    games_modified = 0
    
    for i, game_data in enumerate(games_data):
        game_id = game_data.get("game_id", f"game_{i+1}")
        print(f"\n  Game: {game_id}")
        
        try:
            original_count, cleaned_count = clean_game_positions(game_data, verbose=True)
            total_original += original_count
            total_cleaned += cleaned_count
            
            if original_count != cleaned_count:
                games_modified += 1
                
        except Exception as e:
            print(f"    ✗ Error processing game: {e}")
    
    print(f"\n  Summary:")
    print(f"    Total positions: {total_original} → {total_cleaned}")
    print(f"    Positions removed: {total_original - total_cleaned}")
    print(f"    Games modified: {games_modified}/{len(games_data)}")
    
    if dry_run:
        print("\n  DRY RUN - No changes saved")
        return True
    
    # Save the cleaned data
    backup_path = file_path + ".backup_midmove"
    try:
        # Create backup
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"\n  Created backup: {backup_path}")
        
        # Save cleaned data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(games_data, f, indent=4)
        
        print(f"  ✓ Saved cleaned data")
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving cleaned data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Detect and remove mid-move positions from chess game data.",
        epilog="Mid-move positions occur when frames are captured while pieces are being moved."
    )
    parser.add_argument("path", 
                        type=str, 
                        help="Path to compiled_game_data.json file or directory containing such files")
    parser.add_argument("--dry-run", 
                        action="store_true",
                        help="Show what would be changed without modifying files")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file
        process_compiled_game_data(args.path, dry_run=args.dry_run)
    elif os.path.isdir(args.path):
        # Directory - find all compiled_game_data.json files
        processed_count = 0
        for item in os.listdir(args.path):
            item_path = os.path.join(args.path, item)
            if os.path.isdir(item_path):
                compiled_data_path = os.path.join(item_path, "compiled_game_data.json")
                if os.path.exists(compiled_data_path):
                    print(f"\n{'='*60}")
                    print(f"Processing: {item}")
                    print(f"{'='*60}")
                    if process_compiled_game_data(compiled_data_path, dry_run=args.dry_run):
                        processed_count += 1
        
        print(f"\n{'='*70}")
        print(f"Completed: Processed {processed_count} files")
        print(f"{'='*70}")
    else:
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()