#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- helper functions (unchanged) ---

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def clean_srt(path):
    """
    Remove subtitle blocks whose text is exactly '[music]'.
    Overwrites the original file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    cleaned = []
    for block in blocks:
        lines = block.splitlines()
        text = ' '.join(lines[2:]).strip().lower() if len(lines) >= 3 else ''
        if text == '[music]':
            continue
        cleaned.append(block)

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned) + "\n")

    print(f"Cleaned: {path}")
    return path

def find_video_for_srt(srt_path):
    dirpath = os.path.dirname(srt_path)
    base = os.path.splitext(os.path.basename(srt_path))[0]
    exts = ['.mp4', '.mkv', '.webm', '.mov', '.flv']
    for ext in exts:
        candidate = os.path.join(dirpath, base + ext)
        if os.path.isfile(candidate):
            return candidate
    for file in os.listdir(dirpath):
        if file.lower().endswith(tuple(exts)):
            return os.path.join(dirpath, file)
    print(f"Warning: no video file found for transcript {srt_path}")
    return None

def timestamp_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(':')
    if ',' in rest:
        s, ms = rest.split(',')
    else:
        s, ms = rest, '0'
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def extract_frame(video_path, ts, imgs_dir, use_gpu=False):
    """
    Worker to extract a single frame at timestamp ts from video_path into imgs_dir.
    """
    sec = timestamp_to_seconds(ts)
    safe_ts = ts.replace(':', '-').replace(',', '.')
    img_path = os.path.join(imgs_dir, f"{safe_ts}.png")

    # Base ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',                # overwrite without asking
        '-hide_banner',
        '-loglevel', 'error'
    ]

    if use_gpu:
        # Decode on GPU into CUDA memory
        cmd += [
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda'
        ]

    # Seek directly to the timestamp
    cmd += ['-ss', str(sec), '-i', video_path]

    if use_gpu:
        # Download the frame back to system memory as yuv420p before encoding PNG
        cmd += [
            '-vf', 'hwdownload,format=yuv420p',
            '-frames:v', '1',
            img_path
        ]
    else:
        # CPU-only extraction
        cmd += ['-frames:v', '1', img_path]

    subprocess.run(cmd, check=True)
    return img_path

def extract_frames(srt_path, video_path, imgs_dir, use_gpu=False):
    os.makedirs(imgs_dir, exist_ok=True)
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    starts = re.findall(r"^(\d{2}:\d{2}:\d{2},\d{3}) -->", content,
                        flags=re.MULTILINE)
    if not starts:
        print(f"No timestamps found in {srt_path}")
        return

    workers = min(len(starts), os.cpu_count() or 4)
    print(f"Extracting {len(starts)} frames using {workers} workers"
          f" (GPU={'yes' if use_gpu else 'no'})...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_frame, video_path, ts, imgs_dir, use_gpu): ts
            for ts in starts
        }
        for fut in as_completed(futures):
            ts = futures[fut]
            try:
                img = fut.result()
                print(f"Extracted {img}")
            except Exception as e:
                print(f"Failed timestamp {ts}: {e}")

def process_file(srt_path, use_gpu=False):
    # 1) clean the .srt
    clean_srt(srt_path)
    # 2) find its video
    video = find_video_for_srt(srt_path)
    if not video:
        return
    # 3) extract frames
    imgs_dir = os.path.join(os.path.dirname(srt_path), 'imgs')
    extract_frames(srt_path, video, imgs_dir, use_gpu)

# --- main driver ---

def main():
    parser = argparse.ArgumentParser(
        description="Clean SRTs, extract frames, and remove any video-folder"
                    " that has no transcript."
    )
    parser.add_argument(
        'root_dir',
        help='Parent directory containing transcript/video subfolders'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU acceleration via ffmpeg (CUDA)'
    )
    args = parser.parse_args()

    # 0) First, REMOVE any folder that has video(s) but NO .srt
    video_exts = ('.mp4', '.mkv', '.webm', '.mov', '.flv')
    for dirpath, dirs, files in os.walk(args.root_dir, topdown=False):
        if dirpath == args.root_dir:
            continue
        has_video = any(f.lower().endswith(video_exts) for f in files)
        has_srt   = any(f.lower().endswith('.srt')       for f in files)
        if has_video and not has_srt:
            print(f"→ No transcript in '{dirpath}' → deleting entire folder.")
            shutil.rmtree(dirpath)

    # 1) Gather all remaining .srt files
    all_srts = []
    for dirpath, _, files in os.walk(args.root_dir):
        for fname in files:
            if fname.lower().endswith('.srt'):
                all_srts.append(os.path.join(dirpath, fname))

    if not all_srts:
        print(f"No .srt files found under {args.root_dir}. Exiting.")
        return

    # 2) Process each .srt: clean + extract frames
    for srt in sorted(all_srts):
        print(f"\nProcessing: {srt}")
        try:
            process_file(srt, use_gpu=args.gpu)
        except Exception as e:
            print(f"Error with {srt}: {e}")

if __name__ == '__main__':
    main()
