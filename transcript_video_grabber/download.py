import argparse
import os
import re
import json
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from concurrent.futures import ThreadPoolExecutor, as_completed


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to an SRT-style timestamp: HH:MM:SS,mmm
    """
    millis = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def sanitize_url(url: str) -> str:
    """
    Normalize any YouTube video URL to 'https://www.youtube.com/watch?v=VIDEO_ID'
    """
    parsed = urlparse(url)
    vid = None
    # youtu.be/VIDEO_ID
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/")
    # youtube.com/watch?v=VIDEO_ID
    elif "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        vid = qs.get("v", [None])[0]
    if not vid or len(vid) != 11:
        raise ValueError(f"Could not extract a valid YouTube ID from '{url}'")
    return f"https://www.youtube.com/watch?v={vid}"


def sanitize_filename(name: str) -> str:
    """
    Remove or replace characters that are invalid in filenames.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def download_transcript_by_id(video_id: str, title: str, output_dir: str, languages=None):
    """
    Fetch and save the transcript (SRT) for a given video ID, naming it after the video title.
    """
    safe_title = sanitize_filename(title)
    try:
        if languages:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except NoTranscriptFound:
        print(f"No transcript found for '{title}' ({video_id})")
        return
    except TranscriptsDisabled:
        print(f"Transcripts disabled for '{title}' ({video_id})")
        return
    except Exception as e:
        print(f"Error fetching transcript for '{title}' ({video_id}): {e}")
        return

    srt_path = os.path.join(output_dir, f"{safe_title}.srt")
    with open(srt_path, 'w', encoding='utf-8') as f:
        for idx, entry in enumerate(transcript, start=1):
            start, duration = entry['start'], entry.get('duration', 0)
            text = entry['text'].replace('\n', ' ')
            end = start + duration
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")
    print(f"Transcript saved: {srt_path}")


def process_single_video(url: str, resolution: str, base_dir: str, languages=None):
    """
    Download a single YouTube video and its transcript into its own folder.
    """
    clean_url = sanitize_url(url)
    height = int(resolution.rstrip('p'))
    # Get metadata first
    ydl_opts_info = {'quiet': True, 'skip_download': True}
    with YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(clean_url, download=False)
    title = info.get('title', info['id'])
    safe_title = sanitize_filename(title)
    video_dir = os.path.join(base_dir, safe_title)
    os.makedirs(video_dir, exist_ok=True)

    # Download video + audio up to requested height with concurrent fragments
    ydl_opts = {
        'format': f"mp4[height<={height}]+bestaudio/best[height<={height}]",
        'outtmpl': os.path.join(video_dir, f"{safe_title}.%(ext)s"),
        'quiet': False,
        'noprogress': False,
        'concurrent_fragment_downloads': 4,
    }
    print(f"Downloading '{title}' to {video_dir}…")
    with YoutubeDL(ydl_opts) as ydl:
        downloaded_info = ydl.extract_info(clean_url, download=True)
    print(f"Video downloaded: {downloaded_info.get('title', downloaded_info['id'])}")

    # Download transcript
    download_transcript_by_id(downloaded_info['id'], title, video_dir, languages)


def process_playlist(url: str, resolution: str, base_dir: str, languages=None):
    """
    Download all videos in a playlist and their transcripts into a playlist-named folder.
    """
    # Extract playlist metadata
    ydl_opts_info = {'quiet': True, 'skip_download': True}
    with YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
    playlist_title = info.get('title', info.get('id', 'playlist'))
    safe_pl_title = sanitize_filename(playlist_title)
    playlist_dir = os.path.join(base_dir, safe_pl_title)
    os.makedirs(playlist_dir, exist_ok=True)
    entries = info.get('entries', [])
    print(f"Processing playlist '{playlist_title}' with {len(entries)} videos…")

    # Download multiple videos concurrently
    urls = [
        entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}"
        for entry in entries
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_single_video, vurl, resolution, playlist_dir, languages): vurl for vurl in urls}
        for fut in as_completed(futures):
            vurl = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error downloading {vurl}: {e}")


def is_playlist_url(url: str) -> bool:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return 'list' in qs


def main():
    parser = argparse.ArgumentParser(
        description="Download videos (or playlists) and transcripts from a JSON list."
    )
    parser.add_argument('input', help="Path to the JSON file containing YouTube links.")
    parser.add_argument('-r', '--resolution', default='480p', help="Max video resolution, e.g. 360p, 480p, 720p")
    parser.add_argument('-o', '--output', help="Directory to save all downloads")
    parser.add_argument('-l', '--languages', nargs='+', help="Preferred transcript languages, e.g. en es")
    args = parser.parse_args()

    # Load links from JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    base_dir = args.output or os.getcwd()
    os.makedirs(base_dir, exist_ok=True)

    for item in data:
        link = item.get('link')
        if not link:
            continue
        print("\n===\nProcessing link:", link)
        try:
            if is_playlist_url(link):
                process_playlist(link, args.resolution, base_dir, args.languages)
            else:
                process_single_video(link, args.resolution, base_dir, args.languages)
        except Exception as e:
            print(f"Error handling '{link}': {e}")

if __name__ == '__main__':
    main()
