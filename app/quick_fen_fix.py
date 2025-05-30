"""
Enhanced tool that fixes turn indicators AND reconstructs missing positions.
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Tuple, Optional

try:
    import chess
except ImportError:
    print("Error: python-chess library is required. Install it with: pip install python-chess")
    sys.exit(1)

def analyze_game_sequence(fen_sequence: List[str]) -> Dict:
    """
    Analyze a FEN sequence to understand the game flow and identify issues.
    """
    analysis = {
        'total_positions': len(fen_sequence),
        'valid_positions': 0,
        'invalid_positions': [],
        'direct_transitions': 0,
        'missing_moves': [],
        'turn_issues': 0
    }
    
    valid_fens = []
    for i, fen in enumerate(fen_sequence):
        try:
            board = chess.Board(fen)
            valid_fens.append((i, fen, board))
            analysis['valid_positions'] += 1
        except:
            analysis['invalid_positions'].append(i)
    
    # Analyze transitions between consecutive valid positions
    for i in range(len(valid_fens) - 1):
        idx1, fen1, board1 = valid_fens[i]
        idx2, fen2, board2 = valid_fens[i + 1]
        
        # Check if there's a direct legal move
        direct_move_found = False
        for move in board1.legal_moves:
            test_board = board1.copy()
            test_board.push(move)
            if test_board.board_fen() == board2.board_fen():
                direct_move_found = True
                break
        
        if direct_move_found:
            analysis['direct_transitions'] += 1
        else:
            # Check how many moves apart they are
            piece_diff = abs(sum(1 for c in board1.board_fen() if c.isalpha()) - 
                           sum(1 for c in board2.board_fen() if c.isalpha()))
            
            analysis['missing_moves'].append({
                'from_index': idx1,
                'to_index': idx2,
                'from_fen': fen1,
                'to_fen': fen2,
                'estimated_missing': max(1, piece_diff // 2)
            })
    
    return analysis

def find_intermediate_positions(from_board: chess.Board, to_board: chess.Board, max_depth: int = 3) -> List[chess.Board]:
    """
    Try to find intermediate positions between two board states.
    Returns a list of boards representing the path from from_board to to_board.
    """
    from_fen = from_board.fen()
    to_fen = to_board.fen()
    
    # Quick check: if boards are one move apart
    for move in from_board.legal_moves:
        test_board = from_board.copy()
        test_board.push(move)
        if test_board.board_fen() == to_board.board_fen():
            return [test_board]
    
    # Analyze what changed to guide the search
    from_pieces = {}
    to_pieces = {}
    
    # Count pieces by type and color
    for square in chess.SQUARES:
        from_piece = from_board.piece_at(square)
        to_piece = to_board.piece_at(square)
        
        if from_piece:
            key = (from_piece.piece_type, from_piece.color)
            from_pieces[key] = from_pieces.get(key, 0) + 1
            
        if to_piece:
            key = (to_piece.piece_type, to_piece.color)
            to_pieces[key] = to_pieces.get(key, 0) + 1
    
    # Determine which color likely moved by checking piece changes
    white_changed = False
    black_changed = False
    
    for key in set(list(from_pieces.keys()) + list(to_pieces.keys())):
        from_count = from_pieces.get(key, 0)
        to_count = to_pieces.get(key, 0)
        if from_count != to_count:
            piece_type, color = key
            if color:  # White
                white_changed = True
            else:  # Black
                black_changed = True
    
    # Try to find a path using BFS with limited depth
    from collections import deque
    
    queue = deque([(from_board, [])])
    visited = {from_board.board_fen()}
    
    while queue:
        current_board, path = queue.popleft()
        
        if len(path) >= max_depth:
            continue
        
        # Prioritize moves based on whose pieces changed
        moves = list(current_board.legal_moves)
        
        # Sort moves to prioritize the color that needs to move
        if white_changed and not black_changed and not current_board.turn:
            # Black's turn but only white pieces changed - deprioritize
            continue
        elif black_changed and not white_changed and current_board.turn:
            # White's turn but only black pieces changed - deprioritize  
            continue
            
        for move in moves:
            new_board = current_board.copy()
            new_board.push(move)
            board_fen = new_board.board_fen()
            
            if board_fen == to_board.board_fen():
                # Found the target
                return path + [new_board]
            
            # Check if this move gets us closer to the target
            # by seeing if it matches any of the changes we need
            move_is_promising = False
            
            # Check if the move affects squares that differ between from and to
            from_square = move.from_square
            to_square = move.to_square
            
            if (from_board.piece_at(from_square) != to_board.piece_at(from_square) or
                from_board.piece_at(to_square) != to_board.piece_at(to_square)):
                move_is_promising = True
            
            if move_is_promising and board_fen not in visited:
                visited.add(board_fen)
                queue.append((new_board, path + [new_board]))
    
    return []

def reconstruct_missing_positions(fen_sequence: List[str]) -> Tuple[List[str], List[str]]:
    """
    Reconstruct missing positions in a FEN sequence.
    Returns the enhanced sequence and messages about what was done.
    """
    messages = []
    enhanced_sequence = []
    
    for i, fen in enumerate(fen_sequence):
        try:
            board = chess.Board(fen)
            enhanced_sequence.append(fen)
            
            # If not the last position, check if we need intermediate positions
            if i < len(fen_sequence) - 1:
                next_fen = fen_sequence[i + 1]
                try:
                    next_board = chess.Board(next_fen)
                    
                    # Check if there's a direct move
                    direct_move = False
                    for move in board.legal_moves:
                        test_board = board.copy()
                        test_board.push(move)
                        if test_board.board_fen() == next_board.board_fen():
                            direct_move = True
                            break
                    
                    if not direct_move:
                        # Intelligent reconstruction based on turn order
                        reconstructed = smart_reconstruct_positions(board, next_board)
                        
                        if reconstructed:
                            for j, (inter_board, move_desc) in enumerate(reconstructed):
                                inter_fen = inter_board.fen()
                                enhanced_sequence.append(inter_fen)
                                messages.append(f"Inserted missing position after {i+1}: {move_desc}")
                        else:
                            messages.append(f"Could not reconstruct path between positions {i+1} and {i+2}")
                            
                except Exception as e:
                    messages.append(f"Error processing next position {i+2}: {e}")
                    
        except Exception as e:
            messages.append(f"Invalid FEN at position {i+1}: {e}")
            enhanced_sequence.append(fen)  # Keep the original even if invalid
    
    return enhanced_sequence, messages

def smart_reconstruct_positions(from_board: chess.Board, to_board: chess.Board) -> List[Tuple[chess.Board, str]]:
    """
    Intelligently reconstruct missing positions between two board states.
    Returns a list of (board, move_description) tuples.
    """
    # First, analyze what changed between the positions
    from_pieces = {}
    to_pieces = {}
    
    # Map each square to its piece
    for square in chess.SQUARES:
        from_piece = from_board.piece_at(square)
        to_piece = to_board.piece_at(square)
        
        if from_piece:
            from_pieces[square] = from_piece
        if to_piece:
            to_pieces[square] = to_piece
    
    # Find differences
    changed_squares = set()
    for square in chess.SQUARES:
        from_piece = from_pieces.get(square)
        to_piece = to_pieces.get(square)
        if from_piece != to_piece:
            changed_squares.add(square)
    
    # Determine whose turn it is and what moves are needed
    current_board = from_board.copy()
    path = []
    
    # Try to find a sequence of moves that leads to the target
    max_moves = 3  # Limit the search
    moves_found = find_move_sequence(current_board, to_board, changed_squares, max_moves)
    
    for move in moves_found:
        current_board.push(move)
        move_desc = f"{current_board.san(move)} ({chess.square_name(move.from_square)}-{chess.square_name(move.to_square)})"
        path.append((current_board.copy(), move_desc))
    
    # Remove the last position if it matches the target (to avoid duplication)
    if path and path[-1][0].board_fen() == to_board.board_fen():
        path = path[:-1]
    
    return path

def find_move_sequence(from_board: chess.Board, to_board: chess.Board, changed_squares: set, max_depth: int) -> List[chess.Move]:
    """
    Find a sequence of moves that transforms from_board to to_board.
    Uses the changed_squares hint to prioritize relevant moves.
    """
    from collections import deque
    
    # BFS to find the shortest sequence
    queue = deque([(from_board, [])])
    visited = {from_board.fen()}
    
    while queue:
        current_board, moves = queue.popleft()
        
        if len(moves) >= max_depth:
            continue
        
        # Get all legal moves
        legal_moves = list(current_board.legal_moves)
        
        # Prioritize moves that affect changed squares
        prioritized_moves = []
        other_moves = []
        
        for move in legal_moves:
            if move.from_square in changed_squares or move.to_square in changed_squares:
                prioritized_moves.append(move)
            else:
                other_moves.append(move)
        
        # Try prioritized moves first
        for move in prioritized_moves + other_moves:
            new_board = current_board.copy()
            new_board.push(move)
            
            # Check if we reached the target
            if new_board.board_fen() == to_board.board_fen():
                return moves + [move]
            
            # Continue searching
            new_fen = new_board.fen()
            if new_fen not in visited and len(moves) + 1 < max_depth:
                visited.add(new_fen)
                queue.append((new_board, moves + [move]))
    
    return []

def fix_turn_indicators_with_context(fen_sequence: List[str]) -> Tuple[List[str], List[str]]:
    """
    Fix turn indicators using game context and proper move counting.
    """
    if not fen_sequence:
        return [], []
    
    fixed_fens = []
    messages = []
    
    for i, fen in enumerate(fen_sequence):
        if not fen or not fen.strip():
            continue
        
        parts = fen.split(' ')
        if len(parts) < 2:
            continue
        
        original_turn = parts[1]
        
        # Determine correct turn based on position in sequence
        # Starting position is white's turn, then alternates
        correct_turn = 'w' if i % 2 == 0 else 'b'
        
        parts[1] = correct_turn
        
        # Fix other FEN components
        if len(parts) >= 3 and parts[2] not in ['K', 'Q', 'k', 'q', 'KQ', 'Kk', 'Qq', 'KQk', 'KQq', 'Kkq', 'Qkq', 'KQkq', '-']:
            parts[2] = 'KQkq'
        
        if len(parts) >= 4 and parts[3] != '-':
            if len(parts[3]) != 2 or parts[3][0] not in 'abcdefgh' or parts[3][1] not in '36':
                parts[3] = '-'
        
        # Fix move counters
        while len(parts) < 6:
            if len(parts) == 2:
                parts.append('KQkq')
            elif len(parts) == 3:
                parts.append('-')
            elif len(parts) == 4:
                parts.append('0')
            elif len(parts) == 5:
                # Calculate fullmove number based on position
                fullmove = (i // 2) + 1
                parts.append(str(fullmove))
        
        # Update halfmove clock (simplified - reset on pawn move or capture)
        if len(parts) >= 5:
            parts[4] = '0'  # Simplified
            
        # Update fullmove number
        if len(parts) >= 6:
            fullmove = (i // 2) + 1
            parts[5] = str(fullmove)
        
        fixed_fen = ' '.join(parts)
        
        # Validate the fixed FEN
        try:
            chess.Board(fixed_fen)
            fixed_fens.append(fixed_fen)
            
            if original_turn != correct_turn:
                messages.append(f"Fixed turn indicator at position {i+1}: {original_turn} → {correct_turn}")
            
        except Exception as e:
            messages.append(f"Could not fix FEN at position {i+1}: {e}")
            continue
    
    return fixed_fens, messages

def enhanced_repair_game(game_data: Dict, fix_missing: bool = True) -> int:
    """
    Enhanced repair that fixes turn indicators AND reconstructs missing positions.
    """
    analysis_points = game_data.get("stockfish_analysis_points", [])
    if not analysis_points:
        return 0

    # Extract FEN sequence and preserve metadata
    original_data = []
    for point in analysis_points:
        if point.get("fen"):
            original_data.append({
                'fen': point.get("fen", ""),
                'metadata': {k: v for k, v in point.items() if k != 'fen'}
            })

    if not original_data:
        return 0

    original_fens = [item['fen'] for item in original_data]
    print(f"    Original FENs: {len(original_fens)}")

    # Step 1: Reconstruct missing positions if requested
    if fix_missing:
        enhanced_fens, reconstruction_messages = reconstruct_missing_positions(original_fens)
        if reconstruction_messages:
            print(f"    Position reconstruction:")
            for msg in reconstruction_messages[:5]:  # Show first 5 messages
                print(f"      {msg}")
            if len(reconstruction_messages) > 5:
                print(f"      ... and {len(reconstruction_messages) - 5} more changes")
    else:
        enhanced_fens = original_fens

    # Step 2: Fix turn indicators
    fixed_fens, fix_messages = fix_turn_indicators_with_context(enhanced_fens)
    print(f"    After repair: {len(fixed_fens)} positions")

    if fix_messages:
        turn_fixes = [msg for msg in fix_messages if "Fixed turn indicator" in msg]
        if turn_fixes:
            print(f"    Turn indicators fixed: {len(turn_fixes)}")

    # Step 3: Analyze the final sequence
    final_analysis = analyze_game_sequence(fixed_fens)
    print(f"    Valid positions: {final_analysis['valid_positions']}/{final_analysis['total_positions']}")
    print(f"    Direct transitions: {final_analysis['direct_transitions']}")
    
    remaining_gaps = len(final_analysis['missing_moves'])
    if remaining_gaps > 0:
        print(f"    Remaining gaps: {remaining_gaps}")

    # Create new analysis points with fixed FENs
    new_points = []
    metadata_index = 0
    
    for i, fen in enumerate(fixed_fens):
        # Try to use original metadata if available
        if metadata_index < len(original_data):
            # Check if this position matches an original position
            orig_board_fen = original_data[metadata_index]['fen'].split()[0]
            curr_board_fen = fen.split()[0]
            
            if orig_board_fen == curr_board_fen:
                # Use original metadata
                metadata = original_data[metadata_index]['metadata'].copy()
                metadata_index += 1
            else:
                # This is an inserted position
                metadata = {
                    "eval_cp": 0,
                    "best_move_san": "",
                    "pv": [],
                    "local_commentary_window": "[Reconstructed position]",
                    "video_time_start_str": "",
                    "video_time_end_str": ""
                }
        else:
            # Beyond original data
            metadata = {
                "eval_cp": 0,
                "best_move_san": "",
                "pv": [],
                "local_commentary_window": "[Reconstructed position]",
                "video_time_start_str": "",
                "video_time_end_str": ""
            }

        new_point = {
            "fen": fen,
            "move_num": i + 1,
            **metadata
        }
        new_points.append(new_point)

    game_data["stockfish_analysis_points"] = new_points
    return len(fixed_fens)

def repair_compiled_game_data(file_path: str, fix_missing: bool = True) -> bool:
    """
    Repair compiled game data with optional missing position reconstruction.
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
    
    repair_type = "Enhanced repair (turn indicators + missing positions)" if fix_missing else "Simple repair (turn indicators only)"
    print(f"{repair_type} for {len(games_data)} games...")
    
    total_repaired = 0
    for i, game_data in enumerate(games_data):
        game_id = game_data.get("game_id", f"game_{i+1}")
        print(f"  Repairing game: {game_id}")
        
        try:
            repaired_count = enhanced_repair_game(game_data, fix_missing)
            if repaired_count > 0:
                total_repaired += 1
                print(f"    ✓ Repaired sequence: {repaired_count} positions")
            else:
                print(f"    ⚠ No positions to repair")
        except Exception as e:
            print(f"    ✗ Error repairing game: {e}")
    
    # Save the repaired data
    backup_path = file_path + ".backup_enhanced" if fix_missing else file_path + ".backup_simple"
    try:
        # Create backup
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"  Created backup: {backup_path}")
        
        # Save repaired data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(games_data, f, indent=4)
        
        print(f"  ✓ Saved repair results")
        print(f"  ✓ Repaired {total_repaired}/{len(games_data)} games")
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving repaired data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced repair: fix turn indicators and reconstruct missing positions.")
    parser.add_argument("path", 
                        type=str, 
                        help="Path to compiled_game_data.json file or directory containing such files")
    parser.add_argument("--no-missing", 
                        action="store_true",
                        help="Skip missing position reconstruction (only fix turn indicators)")
    
    args = parser.parse_args()
    
    fix_missing = not args.no_missing
    
    if os.path.isfile(args.path):
        # Single file
        repair_compiled_game_data(args.path, fix_missing)
    elif os.path.isdir(args.path):
        # Directory - find all compiled_game_data.json files
        repaired_count = 0
        for item in os.listdir(args.path):
            item_path = os.path.join(args.path, item)
            if os.path.isdir(item_path):
                compiled_data_path = os.path.join(item_path, "compiled_game_data.json")
                if os.path.exists(compiled_data_path):
                    print(f"\nRepairing: {item}")
                    if repair_compiled_game_data(compiled_data_path, fix_missing):
                        repaired_count += 1
        
        print(f"\n{'='*60}")
        print(f"Repair completed for {repaired_count} files")
        print(f"{'='*60}")
    else:
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()