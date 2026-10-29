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

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

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
                if l in subs:
                    candidates.append((l, subs[l]))
            for l, fmts in subs.items():
                if (l, fmts) not in candidates:
                    candidates.append((l, fmts))

            for l, fmts in auto.items():
                is_translated = any("tlang=" in f.get("url", "") for f in fmts)
                if not is_translated:
                    candidates.insert(0 if l in ["en", "hi"] else len(candidates), (l, fmts))

            for l in ["en", "hi"]:
                if l in auto and (l, auto[l]) not in candidates:
                    candidates.append((l, auto[l]))

            for lang_code, formats in candidates:
                json_fmt = next((f for f in formats if f.get("ext") == "json3"), formats[0] if formats else None)
                if not json_fmt or "url" not in json_fmt:
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


def download_youtube_audio(url: str) -> str:
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
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
        expected_wav = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")

        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback to prepare_filename replacement
        raw_name = ydl.prepare_filename(info)
        base, _ = os.path.splitext(raw_name)
        wav_name = base + ".wav"
        return wav_name


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV format using FFmpeg directly (low RAM streaming)."""
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


def chunk_audio(wav_path: str, chunk_minutes: int = 8) -> tuple:
    """Split WAV audio into chunks using FFmpeg segmenter without loading data into RAM."""
    chunk_seconds = chunk_minutes * 60
    base = os.path.splitext(wav_path)[0]
    output_pattern = f"{base}_chunk_%03d.wav"
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

    raw_chunks = sorted(glob.glob(f"{base}_chunk_[0-9][0-9][0-9].wav"))

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


def process_input(source: str) -> tuple:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        raw_download = download_youtube_audio(source)
        print("Converting audio to standard 16kHz mono WAV...")
        wav_path = convert_to_wav(raw_download)
    else:
        print("Detected local file. Converting to 16kHz mono WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks, duration_sec = chunk_audio(wav_path, chunk_minutes=8)
    print(f"Audio ready — {len(chunks)} chunk(s) created (duration: {duration_sec:.1f}s).")
    return chunks, duration_sec
