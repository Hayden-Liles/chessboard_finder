import os
import json
import re
from datetime import timedelta
import sys # To modify path for importing warped_to_fen

import multiprocessing # Added for parallel image processing
# Add the directory of warped_to_fen.py to sys.path if it's not in the same directory
# Assuming warped_to_fen.py is in the same directory as video_processor.py
# If not, you might need to adjust this path.
# For example, if warped_to_fen.py is one level up:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(SCRIPT_DIR)
# sys.path.append(PARENT_DIR) # Or specific path to 'app' directory

try:
    from warped_to_fen import get_fen_from_image_or_dir
except ImportError as e:
    print(f"Error importing warped_to_fen: {e}")
    print("Please ensure warped_to_fen.py is accessible in your Python path.")
    sys.exit(1)

def srt_time_to_timedelta(time_str):
    """Converts an SRT time string (HH:MM:SS,mmm) to a timedelta object."""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_str)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    raise ValueError(f"Invalid SRT time format: {time_str}")

def img_filename_to_timedelta(filename_stem):
    """Converts an image filename stem (HH-MM-SS.mmm) to a timedelta object."""
    # Replace hyphens with colons and dot with comma for consistency with srt_time_to_timedelta
    time_str_srt_format = filename_stem.replace('-', ':').replace('.', ',')
    return srt_time_to_timedelta(time_str_srt_format)

def timedelta_to_srt_time_str(td_object):
    """Converts a timedelta object back to an SRT time string (HH:MM:SS,mmm)."""
    if td_object is None:
        return "00:00:00,000"
    total_seconds = int(td_object.total_seconds())
    milliseconds = int(td_object.microseconds / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_srt(srt_file_path):
    """Parses an SRT file and returns a list of transcript entries."""
    if not os.path.exists(srt_file_path):
        print(f"SRT file not found: {srt_file_path}")
        return []
    
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = []
    # Regex to capture index, timestamps, and text
    # SRT blocks can have multi-line text
    pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\n\d+\s*\n|\Z)', re.DOTALL | re.MULTILINE)
    
    for match in pattern.finditer(content):
        index = int(match.group(1))
        start_time_str = match.group(2)
        end_time_str = match.group(3)
        text = match.group(4).strip()
        
        entries.append({
            'id_in_file': index, # Original index from SRT file
            'start_td': srt_time_to_timedelta(start_time_str),
            'end_td': srt_time_to_timedelta(end_time_str),
            'text': text
        })
    return entries

def group_fen_blocks(fen_observations):
    """Groups consecutive identical FEN observations into blocks."""
    if not fen_observations:
        return []

    grouped_blocks = []
    current_block = None

    for obs in fen_observations:
        if current_block is None:
            current_block = {
                'fen': obs['fen'],
                'start_td': obs['timestamp_td'],
                'end_td': obs['timestamp_td'],
                'image_timestamps_str': [obs['timestamp_str']]
            }
        elif obs['fen'] == current_block['fen']:
            current_block['end_td'] = obs['timestamp_td'] # Update end time
            current_block['image_timestamps_str'].append(obs['timestamp_str'])
        else:
            grouped_blocks.append(current_block)
            current_block = {
                'fen': obs['fen'],
                'start_td': obs['timestamp_td'],
                'end_td': obs['timestamp_td'],
                'image_timestamps_str': [obs['timestamp_str']]
            }
    
    if current_block: # Add the last block
        grouped_blocks.append(current_block)
        
    return grouped_blocks

def _process_single_image_for_video_wrapper(args):
    """
    Wrapper function for get_fen_from_image_or_dir to be used with multiprocessing.Pool.map.
    It takes a tuple of arguments, calls the FEN detection, and returns
    the original image filename stem (timestamp) and the detected FEN.
    """
    image_path, model_path_arg, debug_arg, kmeans_sample_size_arg, kmeans_iterations_arg, morph_kernel_size_arg = args
    
    # When get_fen_from_image_or_dir is called with a single image path,
    # its own num_workers parameter is not used for parallelism here.
    # The parallelism is handled by the Pool in generate_commentary_json.
    fen_string = get_fen_from_image_or_dir(image_path, model_path_arg, debug=debug_arg, 
                                           kmeans_sample_size=kmeans_sample_size_arg,
                                           kmeans_iterations=kmeans_iterations_arg,
                                           morph_kernel_size=morph_kernel_size_arg)
    timestamp_str = os.path.splitext(os.path.basename(image_path))[0]
    return timestamp_str, fen_string

def generate_commentary_json_improved(video_dir_path, model_path, output_json_path, num_workers=None):
    """
    Enhanced version that tracks position history for better FEN generation.
    """
    imgs_dir = os.path.join(video_dir_path, "imgs")
    video_dir_basename = os.path.basename(video_dir_path)
    expected_srt_filename = f"{video_dir_basename}.srt"
    srt_file_path = os.path.join(video_dir_path, expected_srt_filename)

    if not os.path.exists(srt_file_path):
        print(f"Expected SRT file not found: {srt_file_path}")
        return

    if not os.path.isdir(imgs_dir):
        print(f"Images directory not found: {imgs_dir}")
        return

    image_files = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith('.png')])
    if not image_files:
        print(f"No images found in {imgs_dir}")
        return

    # Process images to get FEN observations (same as before)
    default_kmeans_sample_size = 5000
    default_kmeans_iterations = 30
    default_morph_kernel_size = 15

    tasks = []
    for img_file in image_files:
        image_path = os.path.join(imgs_dir, img_file)
        tasks.append((
            image_path, model_path, False,
            default_kmeans_sample_size, default_kmeans_iterations, default_morph_kernel_size
        ))

    actual_num_workers = num_workers if num_workers is not None else os.cpu_count()
    if actual_num_workers is None:
        actual_num_workers = 1 
        print("Warning: os.cpu_count() returned None, defaulting to 1 worker.")
    
    print(f"Processing {len(image_files)} images from {imgs_dir} with {actual_num_workers} worker(s)...")
    fen_observations = []

    with multiprocessing.Pool(processes=actual_num_workers) as pool:
        results = pool.map(_process_single_image_for_video_wrapper, tasks)

    for i, (timestamp_str, fen_string) in enumerate(results):
        original_img_file = image_files[i]
        if fen_string and isinstance(fen_string, str):
            try:
                timestamp_td = img_filename_to_timedelta(timestamp_str)
                fen_observations.append({
                    'timestamp_str': timestamp_str,
                    'fen': fen_string,
                    'timestamp_td': timestamp_td
                })
            except ValueError as e:
                print(f"    Could not parse timestamp from filename stem {timestamp_str}: {e}")
        else:
            print(f"    No valid FEN detected for {original_img_file}")

    if not fen_observations:
        print("No FENs were successfully extracted from images.")
        return

    print("Grouping FEN observations...")
    grouped_fen_blocks = group_fen_blocks(fen_observations)
    if not grouped_fen_blocks:
        print("No FEN blocks created.")
        return

    print(f"Parsing SRT file: {srt_file_path}...")
    srt_data = parse_srt(srt_file_path)
    if not srt_data:
        print("SRT data is empty or could not be parsed.")
    
    for i, entry in enumerate(srt_data):
        entry['srt_list_idx'] = i

    final_json_output = []
    print("Associating transcripts and building commentary windows...")
    
    # Track move numbers for better FEN generation
    move_counter = 0
    previous_fen = None
    
    for fen_block in grouped_fen_blocks:
        fen_start_td = fen_block['start_td']
        fen_end_td = fen_block['end_td']
        current_fen = fen_block['fen']
        
        # Increment move counter if position changed
        if previous_fen and previous_fen != current_fen:
            move_counter += 1
        elif previous_fen is None:
            move_counter = 1
        
        primary_srt_entries_info = []
        for srt_entry in srt_data:
            if srt_entry['start_td'] < fen_end_td and srt_entry['end_td'] > fen_start_td:
                primary_srt_entries_info.append(srt_entry)
        
        commentary_text = ""
        if primary_srt_entries_info and srt_data:
            min_primary_idx = min(e['srt_list_idx'] for e in primary_srt_entries_info)
            max_primary_idx = max(e['srt_list_idx'] for e in primary_srt_entries_info)
            
            window_start_idx = max(0, min_primary_idx - 4)
            window_end_idx = min(len(srt_data) - 1, max_primary_idx + 4)
            
            commentary_texts = [srt_data[i]['text'] for i in range(window_start_idx, window_end_idx + 1)]
            commentary_text = " ".join(commentary_texts).replace('\n', ' ').strip()

        # Enhanced output object with move tracking
        output_object = {
            "fen": current_fen,
            "video_time_start_str": timedelta_to_srt_time_str(fen_start_td),
            "video_time_end_str": timedelta_to_srt_time_str(fen_end_td),
            "move_num": move_counter,  # Now properly tracked
            "eval_cp": 0,
            "best_move_san": "",
            "pv": [],
            "local_commentary_window": commentary_text
        }
        final_json_output.append(output_object)
        previous_fen = current_fen

    print(f"Writing output to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, indent=4)
    
    print("Processing complete.")


if __name__ == "__main__":
    # --- Configuration ---
    # Base directory where 'test_videos' and 'detector' (for model) might be relative to.
    # If running this script from 'app' directory:
    BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up two levels from app/
    
    # Example video directory relative to BASE_PROJECT_DIR
    # Adjust this to your actual video directory name
    video_folder_name = "Blitz, Bishops, and History _ Develop Your Instincts _ 5-Minute Blitz _ GM Naroditsky's DYI Speedrun"
    video_directory = os.path.join(BASE_PROJECT_DIR, "app", "test_videos", video_folder_name)
    
    # Model path relative to BASE_PROJECT_DIR
    model_file_path = os.path.join(BASE_PROJECT_DIR, "detector", "chess_piece_model.keras")
    
    # Output JSON path
    output_file_path = os.path.join(video_directory, "commentary_data.json")

    # --- Safety Checks ---
    if not os.path.isdir(video_directory):
        print(f"Error: Video directory not found: {video_directory}")
        print("Please check the 'video_folder_name' and 'BASE_PROJECT_DIR' configuration.")
        sys.exit(1)
        
    if not os.path.exists(model_file_path):
        print(f"Error: Model file not found: {model_file_path}")
        print("Please check the 'model_file_path' configuration.")
        sys.exit(1)

    # --- Run Processing ---
    # You can specify the number of workers, e.g., num_workers=4
    # If num_workers is None, it will try to use os.cpu_count()
    # If num_workers is 1, it will run sequentially (useful for debugging)
    generate_commentary_json_improved(video_directory, model_file_path, output_file_path, num_workers=None)


    # --- Example of how to run for another video (if you have one) ---
    # video_folder_name_2 = "Another Video Folder Name"
    # video_directory_2 = os.path.join(BASE_PROJECT_DIR, "app", "test_videos", video_folder_name_2)
    # output_file_path_2 = os.path.join(video_directory_2, "commentary_data.json")
    # if os.path.isdir(video_directory_2):
    #     generate_commentary_json(video_directory_2, model_file_path, output_file_path_2)
    # else:
    #     print(f"Skipping second example, directory not found: {video_directory_2}")
