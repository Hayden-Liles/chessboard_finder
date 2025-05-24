import os
import json
import sys
import argparse
from typing import List, Dict, Optional, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("Error: python-chess library is required. Install it with: pip install python-chess")
    sys.exit(1)

def clean_fen_sequence(fen_list: List[str]) -> List[str]:
    """
    Remove consecutive duplicate FENs and return a cleaned sequence.
    """
    if not fen_list:
        return []
    
    cleaned = [fen_list[0]]
    for fen in fen_list[1:]:
        if fen != cleaned[-1]:
            cleaned.append(fen)
    
    return cleaned

def normalize_fen_for_comparison(fen: str) -> str:
    """
    Normalize FEN for comparison by removing move counters and en passant.
    This helps with FEN detection inconsistencies.
    """
    parts = fen.split(' ')
    if len(parts) >= 4:
        # Keep board, turn, castling but normalize en passant and move counters
        return f"{parts[0]} {parts[1]} {parts[2]} -"
    return fen

def is_valid_fen(fen: str) -> bool:
    """
    Check if a FEN string represents a valid chess position.
    """
    try:
        board = chess.Board(fen)
        return True
    except (ValueError, AssertionError):
        return False

def fix_fen_turn_indicator(fen: str, expected_turn: str) -> str:
    """
    Fix the turn indicator in a FEN string.
    """
    parts = fen.split(' ')
    if len(parts) >= 2:
        parts[1] = expected_turn
        return ' '.join(parts)
    return fen

def find_move_between_positions(from_board: chess.Board, to_fen: str) -> Optional[chess.Move]:
    """
    Find the move that transforms from_board to the position in to_fen.
    Returns None if no legal move is found.
    """
    try:
        target_board = chess.Board(to_fen)
    except (ValueError, AssertionError):
        return None
    
    # Try all legal moves
    for move in from_board.legal_moves:
        test_board = from_board.copy()
        try:
            test_board.push(move)
            
            # Compare normalized board positions
            if normalize_fen_for_comparison(test_board.fen()) == normalize_fen_for_comparison(to_fen):
                return move
                
            # Also try exact board comparison
            if test_board.board_fen() == target_board.board_fen():
                return move
                
        except (ValueError, AssertionError):
            continue
    
    return None

def reconstruct_moves_from_fens(fen_sequence: List[str]) -> Tuple[Optional[chess.pgn.Game], List[str]]:
    """
    Reconstruct chess moves from a sequence of FEN positions with improved error handling.
    """
    if not fen_sequence:
        return None, ["Empty FEN sequence"]
    
    errors = []
    cleaned_fens = clean_fen_sequence(fen_sequence)
    
    if len(cleaned_fens) < 2:
        return None, ["Need at least 2 different positions to reconstruct moves"]
    
    # Filter out invalid FENs
    valid_fens = []
    for i, fen in enumerate(cleaned_fens):
        if is_valid_fen(fen):
            valid_fens.append(fen)
        else:
            errors.append(f"Invalid FEN at position {i}: {fen}")
    
    if len(valid_fens) < 2:
        return None, errors + ["Not enough valid FEN positions to reconstruct moves"]
    
    # Create a new game
    game = chess.pgn.Game()
    node = game
    
    # Set up the starting position
    try:
        board = chess.Board(valid_fens[0])
        if valid_fens[0] != chess.STARTING_FEN:
            game.setup(board)
    except (ValueError, AssertionError) as e:
        errors.append(f"Invalid starting FEN: {valid_fens[0]} - {e}")
        return None, errors
    
    successful_moves = 0
    skipped_positions = 0
    
    for i in range(1, len(valid_fens)):
        current_fen = valid_fens[i]
        
        # Try to fix turn indicator if needed
        expected_turn = 'w' if board.turn else 'b'
        fixed_fen = fix_fen_turn_indicator(current_fen, expected_turn)
        
        move = find_move_between_positions(board, fixed_fen)
        
        if move:
            try:
                # Verify the move is legal before adding it
                if move in board.legal_moves:
                    board.push(move)
                    node = node.add_variation(move)
                    successful_moves += 1
                else:
                    errors.append(f"Move {move} not in legal moves at position {i}")
                    skipped_positions += 1
            except (ValueError, AssertionError) as e:
                errors.append(f"Error applying move {move} at position {i}: {e}")
                skipped_positions += 1
        else:
            # Try with opposite turn in case turn indicator is wrong
            opposite_turn = 'b' if expected_turn == 'w' else 'w'
            alt_fixed_fen = fix_fen_turn_indicator(current_fen, opposite_turn)
            move = find_move_between_positions(board, alt_fixed_fen)
            
            if move and move in board.legal_moves:
                try:
                    board.push(move)
                    node = node.add_variation(move)
                    successful_moves += 1
                except (ValueError, AssertionError):
                    errors.append(f"Could not find valid move to position {i}: {current_fen}")
                    skipped_positions += 1
            else:
                errors.append(f"Could not find valid move to position {i}: {current_fen}")
                skipped_positions += 1
                
                # Try to continue by setting the board to the target position
                try:
                    target_board = chess.Board(fixed_fen)
                    if target_board.is_valid():
                        board = target_board
                        errors.append(f"Jumped to position {i} due to missing move")
                except:
                    # Can't continue from this position
                    break
    
    if successful_moves == 0:
        return None, errors + ["No moves were successfully reconstructed"]
    
    # Add game metadata
    game.headers["Event"] = "Video Analysis"
    game.headers["Result"] = "*"  # Unknown result
    
    if skipped_positions > 0:
        errors.append(f"Skipped {skipped_positions} positions due to move reconstruction issues")
    
    return game, errors

def generate_pgn_for_game(game_data: Dict) -> Tuple[str, List[str]]:
    """
    Generate PGN string for a single game from its analysis points.
    """
    analysis_points = game_data.get("stockfish_analysis_points", [])
    
    if not analysis_points:
        return "", ["No analysis points found"]
    
    # Extract FEN sequence
    fen_sequence = []
    for point in analysis_points:
        fen = point.get("fen", "")
        if fen and fen.strip():
            fen_sequence.append(fen.strip())
    
    if not fen_sequence:
        return "", ["No valid FEN positions found"]
    
    print(f"    Attempting to reconstruct {len(fen_sequence)} positions...")
    
    # Reconstruct the game
    game, errors = reconstruct_moves_from_fens(fen_sequence)
    
    if game is None:
        return "", errors
    
    # Add metadata from the game data
    game.headers["Site"] = "Video Analysis"
    game.headers["White"] = "Player 1"
    game.headers["Black"] = "Player 2"
    game.headers["Round"] = game_data.get("game_id", "Unknown")
    game.headers["Date"] = "????.??.??"
    
    # Convert to PGN string with error handling
    try:
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn_string = game.accept(exporter)
        return pgn_string, errors
    except Exception as e:
        return "", errors + [f"Error generating PGN string: {e}"]

def process_compiled_game_data(compiled_data_path: str) -> bool:
    """
    Process a compiled_game_data.json file and generate PGNs for all games.
    """
    if not os.path.exists(compiled_data_path):
        print(f"File not found: {compiled_data_path}")
        return False
    
    try:
        with open(compiled_data_path, 'r', encoding='utf-8') as f:
            games_data = json.load(f)
    except Exception as e:
        print(f"Error reading {compiled_data_path}: {e}")
        return False
    
    if not games_data:
        print(f"No games found in {compiled_data_path}")
        return False
    
    success_count = 0
    total_games = len(games_data)
    
    for i, game_data in enumerate(games_data):
        game_id = game_data.get("game_id", f"game_{i+1}")
        print(f"  Processing game: {game_id}")
        
        try:
            pgn_string, errors = generate_pgn_for_game(game_data)
            
            if pgn_string and pgn_string.strip():
                game_data["pgn"] = pgn_string
                success_count += 1
                print(f"    ✓ PGN generated successfully")
                
                if errors:
                    print(f"    ⚠ Warnings: {len(errors)}")
                    for error in errors[:2]:  # Show first 2 errors
                        print(f"      - {error}")
                    if len(errors) > 2:
                        print(f"      ... and {len(errors) - 2} more warnings")
            else:
                print(f"    ✗ Failed to generate PGN")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if len(errors) > 3:
                    print(f"      ... and {len(errors) - 3} more errors")
                game_data["pgn"] = ""  # Set empty PGN
                
        except Exception as e:
            print(f"    ✗ Exception during PGN generation: {e}")
            game_data["pgn"] = ""
    
    # Save the updated data
    try:
        with open(compiled_data_path, 'w', encoding='utf-8') as f:
            json.dump(games_data, f, indent=4)
        print(f"  Updated file saved: {compiled_data_path}")
        print(f"  Success rate: {success_count}/{total_games} games")
        return True
    except Exception as e:
        print(f"  Error saving updated file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate PGN data for chess games from FEN analysis.")
    parser.add_argument("parent_directory_path", 
                        type=str, 
                        help="The full path to the parent directory containing video project folders.")
    
    args = parser.parse_args()
    parent_of_categories_full_path = os.path.abspath(args.parent_directory_path)

    if not os.path.isdir(parent_of_categories_full_path):
        print(f"Error: Parent directory not found: {parent_of_categories_full_path}")
        sys.exit(1)

    print(f"Scanning for compiled game data files in: {parent_of_categories_full_path}")
    
    processed_count = 0
    total_success = 0
    total_games = 0
    
    for video_project_folder_name in os.listdir(parent_of_categories_full_path):
        video_project_path = os.path.join(parent_of_categories_full_path, video_project_folder_name)
        
        if os.path.isdir(video_project_path):
            compiled_data_path = os.path.join(video_project_path, "compiled_game_data.json")
            
            if os.path.exists(compiled_data_path):
                print(f"\nProcessing: {video_project_folder_name}")
                if process_compiled_game_data(compiled_data_path):
                    processed_count += 1
                    
                    # Count successes for summary
                    try:
                        with open(compiled_data_path, 'r', encoding='utf-8') as f:
                            games_data = json.load(f)
                        for game_data in games_data:
                            total_games += 1
                            if game_data.get("pgn", "").strip():
                                total_success += 1
                    except:
                        pass
            else:
                print(f"  Skipping {video_project_folder_name}: compiled_game_data.json not found")
    
    print(f"\n" + "="*60)
    print(f"PGN generation completed!")
    print(f"Processed {processed_count} video projects")
    if total_games > 0:
        success_rate = (total_success / total_games) * 100
        print(f"Overall success rate: {total_success}/{total_games} games ({success_rate:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()