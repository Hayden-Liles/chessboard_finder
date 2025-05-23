#!/usr/bin/env python3
"""
Simple tool that ONLY fixes turn indicators without trying to reconstruct missing moves.
This avoids the complexity that was causing errors.
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Tuple

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

def fix_turn_indicators_only(fen_sequence: List[str]) -> Tuple[List[str], List[str]]:
    """
    Fix ONLY the turn indicators in FEN strings, without trying to reconstruct moves.
    """
    if not fen_sequence:
        return [], []
    
    fixed_fens = []
    messages = []
    current_turn = 'w'  # Start with white
    
    for i, fen in enumerate(fen_sequence):
        if not fen or not fen.strip():
            continue
        
        parts = fen.split(' ')
        if len(parts) < 2:
            continue
        
        original_turn = parts[1]
        parts[1] = current_turn
        
        # Fix other basic FEN components if needed
        if len(parts) >= 3 and parts[2] not in ['K', 'Q', 'k', 'q', 'KQ', 'Kk', 'Qq', 'KQk', 'KQq', 'Kkq', 'Qkq', 'KQkq', '-']:
            parts[2] = 'KQkq'
        
        if len(parts) >= 4 and parts[3] != '-':
            if len(parts[3]) != 2 or parts[3][0] not in 'abcdefgh' or parts[3][1] not in '36':
                parts[3] = '-'
        
        # Ensure we have 6 parts
        while len(parts) < 6:
            if len(parts) == 2:
                parts.append('KQkq')
            elif len(parts) == 3:
                parts.append('-')
            elif len(parts) == 4:
                parts.append('0')
            elif len(parts) == 5:
                parts.append('1')
        
        fixed_fen = ' '.join(parts)
        
        # Validate the fixed FEN
        try:
            chess.Board(fixed_fen)
            fixed_fens.append(fixed_fen)
            
            if original_turn != current_turn:
                messages.append(f"Fixed turn indicator at position {i+1}: {original_turn} → {current_turn}")
            
            # Alternate turn for next position
            current_turn = 'b' if current_turn == 'w' else 'w'
            
        except Exception as e:
            messages.append(f"Could not fix FEN at position {i+1}: {e}")
            continue
    
    return fixed_fens, messages

def simple_repair_game(game_data: Dict, report_analysis: bool = True) -> int:
    """
    Simple repair that only fixes turn indicators and reports on missing moves.
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
    
    # Analyze the sequence to report issues
    if report_analysis:
        analysis = analyze_game_sequence(original_fens)
        print(f"    Valid positions: {analysis['valid_positions']}/{analysis['total_positions']}")
        print(f"    Direct transitions: {analysis['direct_transitions']}")
        if analysis['missing_moves']:
            print(f"    Positions with missing moves: {len(analysis['missing_moves'])}")
            # Show first few missing move gaps
            for i, gap in enumerate(analysis['missing_moves'][:3]):
                estimated = gap['estimated_missing']
                from_idx = gap['from_index']
                to_idx = gap['to_index']
                print(f"      Gap {i+1}: Position {from_idx+1} → {to_idx+1} (estimated {estimated} missing moves)")
    
    # Fix turn indicators only
    fixed_fens, fix_messages = fix_turn_indicators_only(original_fens)
    print(f"    After turn fix: {len(fixed_fens)} positions")
    
    if fix_messages:
        turn_fixes = [msg for msg in fix_messages if "Fixed turn indicator" in msg]
        if turn_fixes:
            print(f"    Turn indicators fixed: {len(turn_fixes)}")
    
    # Create new analysis points with fixed FENs
    new_points = []
    for i, fen in enumerate(fixed_fens):
        # Use original metadata if available
        if i < len(original_data):
            metadata = original_data[i]['metadata'].copy()
        else:
            metadata = {
                "eval_cp": 0,
                "best_move_san": "",
                "pv": [],
                "local_commentary_window": "",
                "video_time_start_str": "",
                "video_time_end_str": ""
            }
        
        new_point = {
            "fen": fen,
            "move_num": i + 1,
            **metadata
        }
        new_points.append(new_point)
    
    # Add back any non-FEN points
    non_fen_points = [point for point in analysis_points if not point.get("fen")]
    new_points.extend(non_fen_points)
    
    game_data["stockfish_analysis_points"] = new_points
    return len(fixed_fens)

def simple_repair_compiled_game_data(file_path: str, report_analysis: bool = True) -> bool:
    """
    Simple repair that only fixes turn indicators.
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
    
    print(f"Simple repair (turn indicators only) for {len(games_data)} games...")
    
    total_repaired = 0
    for i, game_data in enumerate(games_data):
        game_id = game_data.get("game_id", f"game_{i+1}")
        print(f"  Repairing game: {game_id}")
        
        try:
            repaired_count = simple_repair_game(game_data, report_analysis)
            if repaired_count > 0:
                total_repaired += 1
                print(f"    ✓ Repaired sequence: {repaired_count} positions")
            else:
                print(f"    ⚠ No positions to repair")
        except Exception as e:
            print(f"    ✗ Error repairing game: {e}")
    
    # Save the repaired data
    backup_path = file_path + ".backup_simple"
    try:
        # Create backup
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"  Created backup: {backup_path}")
        
        # Save repaired data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(games_data, f, indent=4)
        
        print(f"  ✓ Saved simple repair results")
        print(f"  ✓ Repaired {total_repaired}/{len(games_data)} games")
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving repaired data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple repair: fix turn indicators only, report missing moves.")
    parser.add_argument("path", 
                        type=str, 
                        help="Path to compiled_game_data.json file or directory containing such files")
    parser.add_argument("--no-analysis", 
                        action="store_true",
                        help="Skip missing move analysis reporting")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # Single file
        simple_repair_compiled_game_data(args.path, report_analysis=not args.no_analysis)
    elif os.path.isdir(args.path):
        # Directory - find all compiled_game_data.json files
        repaired_count = 0
        for item in os.listdir(args.path):
            item_path = os.path.join(args.path, item)
            if os.path.isdir(item_path):
                compiled_data_path = os.path.join(item_path, "compiled_game_data.json")
                if os.path.exists(compiled_data_path):
                    print(f"\nSimple repair: {item}")
                    if simple_repair_compiled_game_data(compiled_data_path, report_analysis=not args.no_analysis):
                        repaired_count += 1
        
        print(f"\n{'='*60}")
        print(f"Simple repair completed for {repaired_count} files")
        print(f"{'='*60}")
    else:
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

if __name__ == "__main__":
    main()