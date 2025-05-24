import os
import json
import re
from datetime import timedelta
import sys
import argparse # Import argparse for command-line arguments

# --- Constants for Game Segmentation ---
STARTING_FEN_PIECES = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
# Tunable: min duration for a starting FEN block to be considered a new game
MIN_DURATION_FOR_NEW_GAME_START_FEN = timedelta(seconds=0)
# Tunable: how many past distinct FENs to check for rewind
REWIND_DETECTION_HISTORY_LENGTH = 10

def srt_time_to_timedelta(time_str):
    """Converts an SRT time string (HH:MM:SS,mmm) to a timedelta object."""
    if not time_str: return timedelta(0) # Handle empty or None time_str
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_str)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    print(f"Warning: Invalid SRT time format encountered: {time_str}. Returning zero timedelta.")
    return timedelta(0)

def parse_srt_for_full_transcript(srt_file_path):
    """Parses an SRT file and returns the concatenated text of all entries."""
    if not os.path.exists(srt_file_path):
        print(f"SRT file not found for full transcript: {srt_file_path}")
        return ""
    
    texts = []
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = re.compile(r'\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\s*\n(.*?)(?=\n\n\d+\s*\n|\Z)', re.DOTALL | re.MULTILINE)
    for match in pattern.finditer(content):
        texts.append(match.group(1).strip())
    
    return " ".join(texts).replace('\n', ' ').strip()

def compile_game_data(video_dir_path):
    """
    Reads commentary_data.json and the SRT file for a video,
    segments into games, and saves compiled_game_data.json.
    """
    commentary_json_path = os.path.join(video_dir_path, "commentary_data.json")
    if not os.path.exists(commentary_json_path):
        print(f"  Skipping {video_dir_path}: commentary_data.json not found.")
        return

    video_dir_basename = os.path.basename(video_dir_path)
    srt_file_path = os.path.join(video_dir_path, f"{video_dir_basename}.srt")

    try:
        with open(commentary_json_path, 'r', encoding='utf-8') as f:
            # This is the list of FEN blocks (analysis_point_object from video_processor.py)
            fen_blocks_from_json = json.load(f) 
    except Exception as e:
        print(f"  Error reading {commentary_json_path}: {e}")
        return

    if not fen_blocks_from_json:
        print(f"  No FEN data in {commentary_json_path}. Skipping.")
        return

    # Ensure fen_blocks have necessary time information for duration heuristic
    # The video_processor.py should ideally output 'video_time_start_str' and 'video_time_end_str'
    # For now, we'll assume they are present or handle their absence.
    processed_fen_blocks = []
    for block_json in fen_blocks_from_json:
        start_td = srt_time_to_timedelta(block_json.get("video_time_start_str")) # Get with default if missing
        end_td = srt_time_to_timedelta(block_json.get("video_time_end_str"))
        duration_td = end_td - start_td if end_td > start_td else timedelta(seconds=0)
        
        processed_block = {**block_json, 'start_td': start_td, 'end_td': end_td, 'duration_td': duration_td}
        processed_fen_blocks.append(processed_block)

    full_transcript_text = parse_srt_for_full_transcript(srt_file_path)

    all_games_data = []
    current_game_points = []
    game_counter = 0 
    current_game_distinct_fens_history = []

    for i, fen_block in enumerate(processed_fen_blocks):
        is_potential_new_game_start = fen_block['fen'].startswith(STARTING_FEN_PIECES)
        is_true_new_game_trigger = False

        if is_potential_new_game_start:
            is_true_new_game_trigger = True 
            if fen_block['duration_td'] < MIN_DURATION_FOR_NEW_GAME_START_FEN:
                is_true_new_game_trigger = False
            
            if is_true_new_game_trigger and (i + 1) < len(processed_fen_blocks):
                next_fen_block = processed_fen_blocks[i+1]
                if not next_fen_block['fen'].startswith(STARTING_FEN_PIECES):
                    if next_fen_block['fen'] in current_game_distinct_fens_history[-REWIND_DETECTION_HISTORY_LENGTH:]:
                        is_true_new_game_trigger = False
        
        if is_true_new_game_trigger:
            if current_game_points: 
                if game_counter == 0: game_counter = 1
                game_object = {
                    "game_id": f"{video_dir_basename}_game_{game_counter}",
                    "pgn": "", 
                    "full_transcript": full_transcript_text,
                    "stockfish_analysis_points": current_game_points
                }
                all_games_data.append(game_object)
                game_counter += 1
                current_game_points = []
                current_game_distinct_fens_history = []
            elif game_counter == 0 : 
                game_counter = 1
        elif not is_potential_new_game_start and game_counter == 0:
            game_counter = 1 

        # The fen_block itself is already structured like a stockfish_analysis_point
        # We just need to ensure it has the required fields (fen, commentary, placeholders)
        # And remove the temporary 'start_td', 'end_td', 'duration_td' if not needed in final output
        analysis_point = {
            "fen": fen_block['fen'],
            "move_num": fen_block.get("move_num", 0),
            "eval_cp": fen_block.get("eval_cp", 0),
            "best_move_san": fen_block.get("best_move_san", ""),
            "pv": fen_block.get("pv", []),
            "local_commentary_window": fen_block.get("local_commentary_window", "")
        }
        current_game_points.append(analysis_point)

        if not current_game_distinct_fens_history or fen_block['fen'] != current_game_distinct_fens_history[-1]:
            current_game_distinct_fens_history.append(fen_block['fen'])

    if current_game_points:
        if game_counter == 0: game_counter = 1
        game_object = {
            "game_id": f"{video_dir_basename}_game_{game_counter}",
            "pgn": "", 
            "full_transcript": full_transcript_text,
            "stockfish_analysis_points": current_game_points
        }
        all_games_data.append(game_object)

    output_compiled_json_path = os.path.join(video_dir_path, "compiled_game_data.json")
    try:
        with open(output_compiled_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_games_data, f, indent=4)
        print(f"  Successfully compiled game data for {video_dir_basename} -> {output_compiled_json_path}")
    except Exception as e:
        print(f"  Error writing compiled_game_data.json for {video_dir_basename}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Compile game data from video FEN analysis and transcripts.")
    parser.add_argument("parent_directory_path", 
                        type=str, 
                        help="The full path to the parent directory containing video categories (e.g., /path/to/transcript_video_grabber/videos).")
    
    args = parser.parse_args()
    parent_of_categories_full_path = os.path.abspath(args.parent_directory_path)

    if not os.path.isdir(parent_of_categories_full_path):
        print(f"Error: Parent directory of video categories not found: {parent_of_categories_full_path}")
        sys.exit(1)

    # The parent_of_categories_full_path is now assumed to be a directory
    # directly containing video project folders.
    print(f"Scanning for video project folders in: {parent_of_categories_full_path}")
    for video_project_folder_name in os.listdir(parent_of_categories_full_path):
        video_project_path = os.path.join(parent_of_categories_full_path, video_project_folder_name)
        
        # Check if the item is a directory and assume it's a video project folder
        if os.path.isdir(video_project_path):
            print(f"  Processing video project: {video_project_folder_name}")
            compile_game_data(video_project_path)
            
    print("All game data compilation finished.")