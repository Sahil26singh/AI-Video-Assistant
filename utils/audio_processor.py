import os
import shutil
import glob
import subprocess
import wave

# Ensure ffmpeg path is available to yt-dlp and ffmpeg subprocesses
if not shutil.which("ffmpeg"):
    possible_paths = glob.glob(r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*FFmpeg*\*\bin") + \
                     glob.glob(r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*FFmpeg*")
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "ffmpeg.exe")):
            os.environ["PATH"] = path + os.path.pathsep + os.environ.get("PATH", "")
            break

import yt_dlp
import tempfile

DOWNLOAD_DIR = 'downloads'

def extract_video_id(url: str) -> str:
    import re
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def get_youtube_transcript(url: str) -> str:
    """
    Fetch YouTube transcripts directly via yt-dlp mobile client simulation.
    Bypasses datacenter/cloud IP blocks (AWS, Streamlit Cloud) without needing third-party scrapers.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return ""

    ydl_opts = {
        "skip_download": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        },
        "quiet": True,
        "no_warnings": True,
    }
    try:
        import requests

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or {}
            auto = info.get("automatic_captions") or {}
            if not subs and not auto:
                return ""

            # Candidate tracks in priority order:
            # 1. Manual human subtitles (matching en or hi first, then any other)
            # 2. Native auto-captions (where YouTube hasn't auto-translated with 'tlang=')
            # 3. Fallback: Any remaining auto-captions
            candidates = []
            for l in ["en", "en-US", "en-GB", "hi"]:
                if l in subs and "chat" not in l.lower():
                    candidates.append((l, subs[l]))
            for l, fmts in subs.items():
                if "chat" not in l.lower() and (l, fmts) not in candidates:
                    candidates.append((l, fmts))

            for l, fmts in auto.items():
                if "chat" in l.lower():
                    continue
                is_translated = any("tlang=" in f.get("url", "") for f in fmts)
                if not is_translated:
                    candidates.insert(0 if l in ["en", "hi"] else len(candidates), (l, fmts))

            for l in ["en", "hi"]:
                if l in auto and "chat" not in l.lower() and (l, auto[l]) not in candidates:
                    candidates.append((l, auto[l]))

            for lang_code, formats in candidates:
                if "chat" in lang_code.lower():
                    continue
                json_fmt = next((f for f in formats if f.get("ext") == "json3"), formats[0] if formats else None)
                if not json_fmt or "url" not in json_fmt:
                    continue
                if "live_chat" in json_fmt.get("url", ""):
                    continue

                try:
                    resp = requests.get(json_fmt["url"], timeout=10)
                    if resp.status_code != 200:
                        continue

                    if json_fmt.get("ext") == "json3":
                        data = resp.json()
                        lines = []
                        for ev in data.get("events", []):
                            for seg in ev.get("segs", []):
                                t = seg.get("utf8", "").strip()
                                if t and t != "\n":
                                    lines.append(t)
                        text = " ".join(lines).strip()
                        if text:
                            print(f"[Transcript API] Successfully fetched direct transcript in '{lang_code}' ({len(text)} chars).")
                            return text
                    elif resp.text:
                        import re
                        cleaned = re.sub(r"<[^>]+>", " ", resp.text)
                        cleaned = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d+ --> \d{2}:\d{2}:\d{2}[.,]\d+", " ", cleaned)
                        text = " ".join(cleaned.split()).strip()
                        if text:
                            print(f"[Transcript API] Successfully parsed caption stream in '{lang_code}' ({len(text)} chars).")
                            return text
                except Exception:
                    continue
    except Exception as e:
        print(f"[Transcript API] Direct transcript extraction failed for {video_id}: {e}")
    return ""


def download_youtube_audio(url: str, output_dir: str = None) -> str:
    target_dir = output_dir or DOWNLOAD_DIR
    os.makedirs(target_dir, exist_ok=True)
    output_template = os.path.join(target_dir, "%(id)s.%(ext)s")
    
    ffmpeg_exe = shutil.which("ffmpeg")
    ffmpeg_dir = os.path.dirname(ffmpeg_exe) if ffmpeg_exe else None

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")
        expected_wav = os.path.join(target_dir, f"{video_id}.wav")

        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback to prepare_filename replacement
        raw_name = ydl.prepare_filename(info)
        base, _ = os.path.splitext(raw_name)
        wav_name = base + ".wav"
        if os.path.exists(wav_name):
            return wav_name
        return expected_wav


def convert_to_wav(input_path: str, output_dir: str = None) -> str:
    """Convert any audio/video file to 16kHz mono WAV format using FFmpeg directly (low RAM streaming)."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_converted.wav")
    else:
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-vn",                 # Drop video stream
        "-ac", "1",            # Convert to mono
        "-ar", "16000",        # Resample to 16 kHz
        "-c:a", "pcm_s16le",   # Standard 16-bit uncompressed WAV
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 8, output_dir: str = None) -> tuple:
    """Split WAV audio into chunks using FFmpeg segmenter without loading data into RAM."""
    chunk_seconds = chunk_minutes * 60
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_pattern = os.path.join(output_dir, f"{base_name}_chunk_%03d.wav")
        search_pattern = os.path.join(output_dir, f"{base_name}_chunk_[0-9][0-9][0-9].wav")
    else:
        base = os.path.splitext(wav_path)[0]
        output_pattern = f"{base}_chunk_%03d.wav"
        search_pattern = f"{base}_chunk_[0-9][0-9][0-9].wav"

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    # Segment the file on disk directly
    cmd = [
        ffmpeg, "-y", "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        output_pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    raw_chunks = sorted(glob.glob(search_pattern))

    # Filter out empty or tiny trailing fraction chunks
    chunks = []
    for c in raw_chunks:
        if os.path.getsize(c) > 1024 or not chunks:
            chunks.append(c)
        else:
            try:
                os.remove(c)
            except OSError:
                pass

    if not chunks:
        chunks = [wav_path]

    # Calculate duration from WAV header in <1ms without decoding audio
    total_duration_sec = 0.0
    try:
        with wave.open(wav_path, "rb") as wf:
            total_duration_sec = wf.getnframes() / float(wf.getframerate())
    except Exception:
        pass

    return chunks, total_duration_sec


def process_input(source: str, work_dir: str = None) -> tuple:
    """
    Process input audio/video (YouTube URL or local file path) into 16kHz mono WAV chunks.
    All intermediate files (download, converted WAV, chunks) are placed in work_dir
    (or an isolated temporary directory) to avoid cluttering the user's files or filling up disk.
    """
    temp_dir = work_dir or tempfile.mkdtemp(prefix="aivideo_")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if source.startswith("http://") or source.startswith("https://"):
            print("Detected YouTube URL. Downloading audio...")
            raw_download = download_youtube_audio(source, output_dir=temp_dir)
            print("Converting audio to standard 16kHz mono WAV...")
            wav_path = convert_to_wav(raw_download, output_dir=temp_dir)
            # Remove intermediate raw download once converted to 16kHz WAV
            if os.path.abspath(raw_download) != os.path.abspath(wav_path) and os.path.exists(raw_download):
                try:
                    os.remove(raw_download)
                except OSError:
                    pass
        else:
            print("Detected local file. Converting to 16kHz mono WAV in temp directory...")
            wav_path = convert_to_wav(source, output_dir=temp_dir)

        print("Chunking audio...")
        chunks, duration_sec = chunk_audio(wav_path, chunk_minutes=8, output_dir=temp_dir)
        print(f"Audio ready — {len(chunks)} chunk(s) created (duration: {duration_sec:.1f}s).")
        return chunks, duration_sec
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def cleanup_temp_files(chunks: list = None, temp_dir: str = None) -> None:
    """
    Safely delete chunk files, intermediate WAVs, and any dedicated temporary directory
    created during processing. Safe to call multiple times or when files are already removed.
    """
    dirs_to_remove = set()
    if temp_dir and os.path.isdir(temp_dir):
        dirs_to_remove.add(os.path.abspath(temp_dir))

    if chunks:
        for chunk in chunks:
            if not chunk:
                continue
            parent = os.path.abspath(os.path.dirname(chunk))
            base_dir = os.path.basename(parent)
            # If the chunk is inside our designated temporary directory, collect it for removal
            if "aivideo_" in base_dir or "vid_upload_" in base_dir:
                dirs_to_remove.add(parent)
            else:
                try:
                    if os.path.exists(chunk):
                        os.remove(chunk)
                except OSError:
                    pass

    for d in dirs_to_remove:
        try:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

