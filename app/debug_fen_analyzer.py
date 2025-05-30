#!/usr/bin/env python3
"""
Debug script to analyze FEN sequences and identify issues.
Run this to understand what's happening with your FEN data before PGN generation.
"""

import os
import json
import sys
import argparse

try:
    import chess
except ImportError:
    print("Error: python-chess library is required. Install it with: pip install python-chess")
    sys.exit(1)

def analyze_fen_validity(fen_sequence):
    """Analyze which FENs are valid/invalid and why."""
    print(f"\n=== FEN Validity Analysis ===")
    print(f"Total FENs: {len(fen_sequence)}")
    
    valid_count = 0
    invalid_fens = []
    
    for i, fen in enumerate(fen_sequence):
        try:
            board = chess.Board(fen)
            valid_count += 1
        except Exception as e:
            invalid_fens.append((i, fen, str(e)))
    
    print(f"Valid FENs: {valid_count}")
    print(f"Invalid FENs: {len(invalid_fens)}")
    
    if invalid_fens:
        print(f"\nFirst few invalid FENs:")
        for i, (idx, fen, error) in enumerate(invalid_fens[:5]):
            print(f"  {idx}: {fen}")
            print(f"      Error: {error}")
        if len(invalid_fens) > 5:
            print(f"  ... and {len(invalid_fens) - 5} more invalid FENs")

def analyze_fen_sequence_changes(fen_sequence):
    """Analyze how positions change between consecutive FENs."""
    print(f"\n=== Position Change Analysis ===")
    
    valid_fens = []
    for fen in fen_sequence:
        try:
            chess.Board(fen)
            valid_fens.append(fen)
        except:
            continue
    
    if len(valid_fens) < 2:
        print("Not enough valid FENs to analyze changes")
        return
    
    print(f"Analyzing {len(valid_fens)} valid positions...")
    
    # Remove consecutive duplicates
    unique_fens = [valid_fens[0]]
    for fen in valid_fens[1:]:
        if fen != unique_fens[-1]:
            unique_fens.append(fen)
    
    print(f"Unique positions: {len(unique_fens)}")
    
    # Analyze differences
    move_found_count = 0
    no_move_count = 0
    illegal_transitions = []
    
    for i in range(len(unique_fens) - 1):
        try:
            board1 = chess.Board(unique_fens[i])
            board2 = chess.Board(unique_fens[i + 1])
            
            # Try to find a legal move
            move_found = False
            for move in board1.legal_moves:
                test_board = board1.copy()
                test_board.push(move)
                if test_board.board_fen() == board2.board_fen():
                    move_found = True
                    break
            
            if move_found:
                move_found_count += 1
            else:
                no_move_count += 1
                illegal_transitions.append((i, unique_fens[i], unique_fens[i + 1]))
                
        except Exception as e:
            illegal_transitions.append((i, unique_fens[i], unique_fens[i + 1], str(e)))
    
    print(f"Legal transitions: {move_found_count}")
    print(f"Problematic transitions: {no_move_count}")
    
    if illegal_transitions:
        print(f"\nFirst few problematic transitions:")
        for i, transition in enumerate(illegal_transitions[:3]):
            if len(transition) == 3:
                idx, fen1, fen2 = transition
                print(f"  Transition {idx} -> {idx + 1}:")
                print(f"    From: {fen1}")
                print(f"    To:   {fen2}")
            else:
                idx, fen1, fen2, error = transition
                print(f"  Transition {idx} -> {idx + 1} (Error: {error}):")
                print(f"    From: {fen1}")
                print(f"    To:   {fen2}")

def analyze_turn_indicators(fen_sequence):
    """Analyze turn indicators in FEN sequence."""
    print(f"\n=== Turn Indicator Analysis ===")
    
    turn_stats = {'w': 0, 'b': 0, 'other': 0}
    consecutive_same_turn = 0
    max_consecutive = 0
    current_consecutive = 1
    
    valid_fens = []
    for fen in fen_sequence:
        try:
            chess.Board(fen)
            valid_fens.append(fen)
        except:
            continue
    
    if not valid_fens:
        print("No valid FENs to analyze")
        return
    
    prev_turn = None
    for fen in valid_fens:
        parts = fen.split(' ')
        if len(parts) >= 2:
            turn = parts[1]
            turn_stats[turn] = turn_stats.get(turn, 0) + 1
            
            if turn == prev_turn:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
            prev_turn = turn
        else:
            turn_stats['other'] += 1
    
    print(f"Turn distribution: White: {turn_stats.get('w', 0)}, Black: {turn_stats.get('b', 0)}, Other: {turn_stats.get('other', 0)}")
    print(f"Max consecutive same turn: {max_consecutive}")
    
    if turn_stats.get('w', 0) == len(valid_fens) or turn_stats.get('b', 0) == len(valid_fens):
        print("⚠️  WARNING: All positions have the same turn indicator - this suggests FEN generation issues")

def debug_specific_game(game_data):
    """Debug a specific game's FEN sequence."""
    game_id = game_data.get("game_id", "Unknown")
    print(f"\n{'='*60}")
    print(f"DEBUGGING GAME: {game_id}")
    print(f"{'='*60}")
    
    analysis_points = game_data.get("stockfish_analysis_points", [])
    print(f"Analysis points: {len(analysis_points)}")
    
    if not analysis_points:
        print("No analysis points found")
        return
    
    fen_sequence = [point.get("fen", "") for point in analysis_points if point.get("fen")]
    print(f"FEN positions extracted: {len(fen_sequence)}")
    
    if not fen_sequence:
        print("No FEN positions found")
        return
    
    # Show first few FENs
    print(f"\nFirst few FENs:")
    for i, fen in enumerate(fen_sequence[:5]):
        print(f"  {i}: {fen}")
    if len(fen_sequence) > 5:
        print(f"  ... and {len(fen_sequence) - 5} more")
    
    analyze_fen_validity(fen_sequence)
    analyze_turn_indicators(fen_sequence)
    analyze_fen_sequence_changes(fen_sequence)

def main():
    parser = argparse.ArgumentParser(description="Debug FEN sequences in compiled game data.")
    parser.add_argument("compiled_data_path", 
                        type=str, 
                        help="Path to compiled_game_data.json file")
    parser.add_argument("--game-index", 
                        type=int, 
                        default=0,
                        help="Index of game to debug (default: 0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.compiled_data_path):
        print(f"{args.compiled_data_path}")
        print(f"Error: File not found: {args.compiled_data_path}")
        sys.exit(1)
    
    try:
        with open(args.compiled_data_path, 'r', encoding='utf-8') as f:
            games_data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    if not games_data:
        print("No games found in file")
        sys.exit(1)
    
    print(f"Found {len(games_data)} games in file")
    
    if args.game_index >= len(games_data):
        print(f"Error: Game index {args.game_index} out of range (0-{len(games_data)-1})")
        sys.exit(1)
    
    # Debug specific game
    debug_specific_game(games_data[args.game_index])
    
    # Also show summary for all games
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR ALL GAMES")
    print(f"{'='*60}")
    
    total_positions = 0
    total_games_with_positions = 0
    
    for i, game_data in enumerate(games_data):
        analysis_points = game_data.get("stockfish_analysis_points", [])
        fen_count = len([p for p in analysis_points if p.get("fen")])
        if fen_count > 0:
            total_games_with_positions += 1
            total_positions += fen_count
    
    print(f"Games with FEN data: {total_games_with_positions}/{len(games_data)}")
    print(f"Total FEN positions: {total_positions}")
    if total_games_with_positions > 0:
        print(f"Average positions per game: {total_positions / total_games_with_positions:.1f}")

if __name__ == "__main__":
    main()