import os
import json
import re
from datetime import timedelta
import sys
import time
from tqdm import tqdm
import psutil

# Memory and performance monitoring
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Set multiprocessing start method to avoid CUDA issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Import the optimized warped_to_fen functions
try:
    from warped_to_fen_optimized import get_fen_from_image_or_dir_optimized, OptimizedChessProcessor
except ImportError:
    try:
        from warped_to_fen import get_fen_from_image_or_dir
        print("⚠️  Using legacy warped_to_fen module (slower performance)")
        get_fen_from_image_or_dir_optimized = None
        OptimizedChessProcessor = None
    except ImportError as e:
        print(f"❌ Error importing FEN processing modules: {e}")
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
    pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\n\d+\s*\n|\Z)', re.DOTALL | re.MULTILINE)
    
    for match in pattern.finditer(content):
        index = int(match.group(1))
        start_time_str = match.group(2)
        end_time_str = match.group(3)
        text = match.group(4).strip()
        
        entries.append({
            'id_in_file': index,
            'start_td': srt_time_to_timedelta(start_time_str),
            'end_td': srt_time_to_timedelta(end_time_str),
            'text': text
        })
    return entries

def group_fen_blocks(fen_observations, progress_bar=None):
    """Groups consecutive identical FEN observations into blocks with progress tracking."""
    if not fen_observations:
        return []

    grouped_blocks = []
    current_block = None

    if progress_bar:
        progress_bar.set_description("Grouping FEN blocks")

    for i, obs in enumerate(fen_observations):
        if current_block is None:
            current_block = {
                'fen': obs['fen'],
                'start_td': obs['timestamp_td'],
                'end_td': obs['timestamp_td'],
                'image_timestamps_str': [obs['timestamp_str']],
                'image_filenames': [obs['timestamp_str'] + '.png']
            }
        elif obs['fen'] == current_block['fen']:
            current_block['end_td'] = obs['timestamp_td']
            current_block['image_timestamps_str'].append(obs['timestamp_str'])
            current_block['image_filenames'].append(obs['timestamp_str'] + '.png')
        else:
            grouped_blocks.append(current_block)
            current_block = {
                'fen': obs['fen'],
                'start_td': obs['timestamp_td'],
                'end_td': obs['timestamp_td'],
                'image_timestamps_str': [obs['timestamp_str']],
                'image_filenames': [obs['timestamp_str'] + '.png']
            }
        
        if progress_bar and i % 10 == 0:
            progress_bar.update(10)
    
    if current_block:
        grouped_blocks.append(current_block)
        
    return grouped_blocks

class ProgressTracker:
    """Enhanced progress tracking with ETA and performance metrics"""
    
    def __init__(self, total_images):
        self.total_images = total_images
        self.start_time = time.time()
        self.processed_images = 0
        self.successful_extractions = 0
        self.memory_start = get_memory_usage()
        
    def update(self, successful=True):
        self.processed_images += 1
        if successful:
            self.successful_extractions += 1
            
    def get_stats(self):
        elapsed = time.time() - self.start_time
        if self.processed_images > 0:
            avg_time_per_image = elapsed / self.processed_images
            eta = avg_time_per_image * (self.total_images - self.processed_images)
        else:
            eta = 0
            
        current_memory = get_memory_usage()
        memory_diff = current_memory - self.memory_start
        
        return {
            'processed': self.processed_images,
            'successful': self.successful_extractions,
            'success_rate': (self.successful_extractions / max(1, self.processed_images)) * 100,
            'elapsed': elapsed,
            'eta': eta,
            'memory_mb': current_memory,
            'memory_diff_mb': memory_diff,
            'images_per_second': self.processed_images / max(0.1, elapsed)
        }

def generate_commentary_json_ultra_optimized(video_dir_path, model_path, output_json_path, 
                                           batch_size=64, progress_callback=None):
    """
    Ultra-optimized version with batch processing, progress tracking, and performance monitoring.
    
    Args:
        video_dir_path (str): Path to video directory containing imgs/ and .srt file
        model_path (str): Path to PyTorch model (.pth file)
        output_json_path (str): Path where to save the commentary JSON
        batch_size (int): Batch size for neural network inference (higher = faster)
        progress_callback (callable): Optional callback for progress updates
    """
    print("🚀 Starting ultra-optimized video processing...")
    print(f"📁 Video directory: {video_dir_path}")
    print(f"🤖 Model file: {model_path}")
    print(f"📊 Batch size: {batch_size}")
    print(f"💾 Initial memory usage: {get_memory_usage():.1f} MB")
    print("-" * 60)
    
    # Validate paths
    imgs_dir = os.path.join(video_dir_path, "imgs")
    video_dir_basename = os.path.basename(video_dir_path)
    expected_srt_filename = f"{video_dir_basename}.srt"
    srt_file_path = os.path.join(video_dir_path, expected_srt_filename)

    if not os.path.exists(srt_file_path):
        print(f"❌ Expected SRT file not found: {srt_file_path}")
        return

    if not os.path.isdir(imgs_dir):
        print(f"❌ Images directory not found: {imgs_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ PyTorch model file not found: {model_path}")
        return

    # Get image files
    image_files = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith('.png')])
    if not image_files:
        print(f"❌ No images found in {imgs_dir}")
        return

    print(f"📸 Found {len(image_files)} images to process")
    
    # Initialize progress tracking
    progress_tracker = ProgressTracker(len(image_files))
    
    # Process images using optimized batch processing
    print("\n🔄 Phase 1: Chess Position Extraction")
    print("=" * 50)
    
    start_time = time.time()
    
    def progress_update(message):
        if progress_callback:
            progress_callback(message)
        stats = progress_tracker.get_stats()
        print(f"   {message} | Memory: {stats['memory_mb']:.1f}MB | Speed: {stats['images_per_second']:.1f} img/s")
    
    # Use optimized processing if available
    if get_fen_from_image_or_dir_optimized:
        try:
            results = get_fen_from_image_or_dir_optimized(
                path_to_process=imgs_dir,
                model_path=model_path,
                debug=False,
                batch_size=batch_size,
                progress_callback=progress_update
            )
        except Exception as e:
            print(f"⚠️  Optimized processing failed: {e}")
            print("🔄 Falling back to legacy processing...")
            results = get_fen_from_image_or_dir(imgs_dir, model_path, debug=False, num_workers=4)
    else:
        # Fallback to legacy processing
        results = get_fen_from_image_or_dir(imgs_dir, model_path, debug=False, num_workers=4)
    
    processing_time = time.time() - start_time
    
    # Collect FEN observations with progress bar
    print("\n🔄 Phase 2: Processing Results")
    print("=" * 50)
    
    fen_observations = []
    successful_extractions = 0
    
    with tqdm(total=len(image_files), desc="Processing results", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for img_file in image_files:
            fen_string = results.get(img_file) if isinstance(results, dict) else None
            
            if fen_string and isinstance(fen_string, str):
                try:
                    timestamp_str = os.path.splitext(img_file)[0]
                    timestamp_td = img_filename_to_timedelta(timestamp_str)
                    fen_observations.append({
                        'timestamp_str': timestamp_str,
                        'fen': fen_string,
                        'timestamp_td': timestamp_td
                    })
                    successful_extractions += 1
                    progress_tracker.update(successful=True)
                except ValueError as e:
                    progress_tracker.update(successful=False)
            else:
                progress_tracker.update(successful=False)
            
            pbar.update(1)
            
            # Update description with stats
            if pbar.n % 50 == 0:
                stats = progress_tracker.get_stats()
                pbar.set_description(f"Processing results (Success: {stats['success_rate']:.1f}%)")

    # Performance summary
    stats = progress_tracker.get_stats()
    print(f"\n📊 Extraction Performance Summary:")
    print(f"   ⚡ Total time: {processing_time:.1f} seconds")
    print(f"   🎯 Success rate: {stats['success_rate']:.1f}% ({successful_extractions}/{len(image_files)})")
    print(f"   🏃 Processing speed: {stats['images_per_second']:.1f} images/second")
    print(f"   💾 Memory usage: {stats['memory_mb']:.1f} MB (Δ{stats['memory_diff_mb']:+.1f} MB)")

    if not fen_observations:
        print("\n❌ No FENs were successfully extracted from images.")
        print("This could indicate:")
        print("  - PyTorch model compatibility issues")
        print("  - Model class mapping mismatch")
        print("  - Board detection problems")
        print("  - Model file corruption")
        return

    # Group FEN blocks with progress
    print(f"\n🔄 Phase 3: Grouping {len(fen_observations)} FEN observations")
    print("=" * 50)
    
    with tqdm(total=len(fen_observations), desc="Grouping FEN blocks") as pbar:
        grouped_fen_blocks = group_fen_blocks(fen_observations, pbar)

    print(f"✅ Created {len(grouped_fen_blocks)} position blocks")

    # Parse SRT file
    print(f"\n🔄 Phase 4: Processing Transcript")
    print("=" * 50)
    
    print(f"📝 Parsing SRT file: {os.path.basename(srt_file_path)}")
    srt_data = parse_srt(srt_file_path)
    
    if not srt_data:
        print("⚠️  SRT data is empty or could not be parsed.")
    else:
        print(f"✅ Loaded {len(srt_data)} transcript entries")
    
    for i, entry in enumerate(srt_data):
        entry['srt_list_idx'] = i

    # Build final output with progress
    print(f"\n🔄 Phase 5: Building Commentary Data")
    print("=" * 50)
    
    final_json_output = []
    move_counter = 0
    previous_fen = None
    
    with tqdm(total=len(grouped_fen_blocks), desc="Building commentary", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for block_idx, fen_block in enumerate(grouped_fen_blocks):
            fen_start_td = fen_block['start_td']
            fen_end_td = fen_block['end_td']
            current_fen = fen_block['fen']
            
            # Increment move counter if position changed
            if previous_fen and previous_fen != current_fen:
                move_counter += 1
            elif previous_fen is None:
                move_counter = 1
            
            # Find overlapping SRT entries
            primary_srt_entries_info = []
            for srt_entry in srt_data:
                if srt_entry['start_td'] < fen_end_td and srt_entry['end_td'] > fen_start_td:
                    primary_srt_entries_info.append(srt_entry)
            
            # Build commentary window
            commentary_text = ""
            if primary_srt_entries_info and srt_data:
                min_primary_idx = min(e['srt_list_idx'] for e in primary_srt_entries_info)
                max_primary_idx = max(e['srt_list_idx'] for e in primary_srt_entries_info)
                
                # Expand window to include surrounding context
                window_start_idx = max(0, min_primary_idx - 4)
                window_end_idx = min(len(srt_data) - 1, max_primary_idx + 4)
                
                commentary_texts = [srt_data[i]['text'] for i in range(window_start_idx, window_end_idx + 1)]
                commentary_text = " ".join(commentary_texts).replace('\n', ' ').strip()

            # Enhanced output object
            output_object = {
                "fen": current_fen,
                "video_time_start_str": timedelta_to_srt_time_str(fen_start_td),
                "video_time_end_str": timedelta_to_srt_time_str(fen_end_td),
                "move_num": move_counter,
                "eval_cp": 0,
                "best_move_san": "",
                "pv": [],
                "local_commentary_window": commentary_text,
                "source_images": fen_block['image_filenames'],
                "image_timestamps": fen_block['image_timestamps_str'],
                "block_duration_seconds": (fen_end_td - fen_start_td).total_seconds(),
                "images_in_block": len(fen_block['image_filenames'])
            }
            final_json_output.append(output_object)
            previous_fen = current_fen
            
            pbar.update(1)
            
            # Update progress description
            if (block_idx + 1) % 10 == 0:
                pbar.set_description(f"Building commentary (Move {move_counter})")

    # Save output
    print(f"\n🔄 Phase 6: Saving Results")
    print("=" * 50)
    
    print(f"💾 Writing output to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Calculate total video duration
    total_duration = sum(block['block_duration_seconds'] for block in final_json_output)
    
    # Enhanced metadata
    final_stats = progress_tracker.get_stats()
    output_data = {
        "metadata": {
            "total_positions": len(final_json_output),
            "total_images_processed": len(image_files),
            "successful_extractions": successful_extractions,
            "extraction_success_rate": f"{(successful_extractions/len(image_files)*100):.1f}%",
            "model_file": os.path.basename(model_path),
            "video_directory": os.path.basename(video_dir_path),
            "total_video_duration_seconds": total_duration,
            "processing_time_seconds": processing_time,
            "processing_speed_fps": final_stats['images_per_second'],
            "model_type": "PyTorch_Optimized" if get_fen_from_image_or_dir_optimized else "PyTorch_Legacy",
            "batch_size": batch_size,
            "memory_peak_mb": final_stats['memory_mb'],
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_metrics": {
                "images_per_second": final_stats['images_per_second'],
                "memory_efficiency_mb_per_image": final_stats['memory_diff_mb'] / len(image_files),
                "time_per_image_seconds": processing_time / len(image_files)
            }
        },
        "positions": final_json_output
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Final summary
    print("\n" + "🎉 PROCESSING COMPLETE! 🎉".center(60))
    print("=" * 60)
    print(f"📊 Final Summary:")
    print(f"   📍 Total positions extracted: {len(final_json_output)}")
    print(f"   📸 Images processed: {len(image_files)}")
    print(f"   🎯 Success rate: {(successful_extractions/len(image_files)*100):.1f}%")
    print(f"   ⏱️  Total processing time: {processing_time:.1f} seconds")
    print(f"   🏃 Average processing speed: {final_stats['images_per_second']:.1f} images/second")
    print(f"   💾 Peak memory usage: {final_stats['memory_mb']:.1f} MB")
    print(f"   🎬 Video duration covered: {total_duration:.1f} seconds")
    print(f"   💾 Output saved to: {output_json_path}")
    
    # Performance recommendations
    if final_stats['images_per_second'] < 2.0:
        print(f"\n💡 Performance Tips:")
        print(f"   - Consider increasing batch_size (current: {batch_size})")
        print(f"   - Ensure CUDA is available for GPU acceleration")
        print(f"   - Check if system has sufficient RAM")
    
    if successful_extractions / len(image_files) < 0.8:
        print(f"\n⚠️  Quality Warning: Success rate below 80%")
        print(f"   Consider checking:")
        print(f"   - Model compatibility with training data")
        print(f"   - Image quality and board visibility")
        print(f"   - Model file integrity")
    
    print("=" * 60)
    return output_data

# Backward compatibility function
def generate_commentary_json_improved(video_dir_path, model_path, output_json_path, num_workers=None):
    """Legacy function maintained for backward compatibility"""
    batch_size = 32 if num_workers != 1 else 16  # Smaller batch for single worker
    return generate_commentary_json_ultra_optimized(
        video_dir_path=video_dir_path,
        model_path=model_path,
        output_json_path=output_json_path,
        batch_size=batch_size
    )

if __name__ == "__main__":
    # Configuration with improved error handling
    BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuration
    video_folder_name = "2"
    video_directory = os.path.join(BASE_PROJECT_DIR, "app", "test_videos", video_folder_name)
    model_file_path = os.path.join(BASE_PROJECT_DIR, "app", "chess_model.pth")
    output_file_path = os.path.join(video_directory, "commentary_data.json")

    # Safety checks with better error messages
    if not os.path.isdir(video_directory):
        print(f"❌ Error: Video directory not found: {video_directory}")
        print("Please check the 'video_folder_name' and directory structure.")
        sys.exit(1)
        
    if not os.path.exists(model_file_path):
        print(f"❌ Error: PyTorch model file not found: {model_file_path}")
        sys.exit(1)

    # PyTorch environment check
    try:
        import torch
        print(f"✅ PyTorch detected: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("ℹ️  CUDA not available, using CPU (slower performance)")
    except ImportError:
        print("❌ Error: PyTorch not installed!")
        sys.exit(1)

    # Test with single image first
    print("\n🧪 Quick compatibility test...")
    test_imgs_dir = os.path.join(video_directory, "imgs")
    if os.path.exists(test_imgs_dir):
        test_imgs = [f for f in os.listdir(test_imgs_dir) if f.lower().endswith('.png')][:1]
        if test_imgs:
            test_img_path = os.path.join(test_imgs_dir, test_imgs[0])
            print(f"Testing with: {test_imgs[0]}")
            try:
                if get_fen_from_image_or_dir_optimized:
                    test_fen = get_fen_from_image_or_dir_optimized(
                        test_img_path, model_file_path, debug=False, batch_size=1,
                        progress_callback=lambda x: None
                    )
                else:
                    from warped_to_fen import get_fen_from_image_or_dir
                    test_fen = get_fen_from_image_or_dir(test_img_path, model_file_path, debug=False)
                
                if test_fen:
                    print(f"✅ Test successful! FEN: {str(test_fen)[:50]}...")
                else:
                    print("⚠️  Test returned None - proceeding anyway")
            except Exception as e:
                print(f"❌ Test failed: {e}")
                print("Proceeding with batch processing anyway...")

    # Determine optimal batch size based on available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb > 16:
        optimal_batch_size = 128
    elif available_memory_gb > 8:
        optimal_batch_size = 64
    elif available_memory_gb > 4:
        optimal_batch_size = 32
    else:
        optimal_batch_size = 16
    
    print(f"\n🎯 Optimal batch size for {available_memory_gb:.1f}GB RAM: {optimal_batch_size}")
    
    # Run ultra-optimized processing
    try:
        print(f"\n{'🚀 STARTING ULTRA-OPTIMIZED PROCESSING 🚀':^60}")
        result = generate_commentary_json_ultra_optimized(
            video_directory, 
            model_file_path, 
            output_file_path, 
            batch_size=optimal_batch_size
        )
        
        if result:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📋 Check the results in: {output_file_path}")
        
    except Exception as e:
        print(f"\n❌ Ultra-optimized processing failed: {e}")
        print("🔄 Falling back to legacy processing...")
        try:
            generate_commentary_json_improved(video_directory, model_file_path, output_file_path, num_workers=4)
        except Exception as e2:
            print(f"❌ Legacy processing also failed: {e2}")
            print("Please check your setup and try again.")