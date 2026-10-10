import os
import shutil
import glob

# Ensure ffmpeg path is available to pydub and yt-dlp
if not shutil.which("ffmpeg"):
    possible_paths = glob.glob(r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*FFmpeg*\*\bin") + \
                     glob.glob(r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*FFmpeg*")
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "ffmpeg.exe")):
            os.environ["PATH"] = path + os.path.pathsep + os.environ.get("PATH", "")
            break

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def extract_video_id(url: str) -> str:
    import re
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def get_youtube_transcript(url: str) -> str:
    """
    Fetch YouTube transcript using a 3-layer strategy:
      1. youtube-transcript-api (fastest, works on localhost)
      2. Invidious public API (bypasses YouTube IP bans on cloud servers)
      3. Additional Invidious instances as further fallbacks
    Returns plain transcript text, or "" if all methods fail.
    """
    import requests

    video_id = extract_video_id(url)
    if not video_id:
        return ""

    # ── Layer 1: youtube-transcript-api ──────────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

        ytta = YouTubeTranscriptApi()
        snippets = None

        # Try all available transcripts (manual + auto-generated, any language)
        try:
            t_list = ytta.list(video_id)
            for t in t_list:
                try:
                    snippets = t.fetch()
                    if snippets:
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Explicit language fallback
        if not snippets:
            for lang_code in ["en", "hi", "en-US", "en-GB"]:
                try:
                    snippets = ytta.fetch(video_id, languages=[lang_code])
                    if snippets:
                        break
                except Exception:
                    continue

        if snippets:
            text = " ".join(
                getattr(s, "text", "") if not isinstance(s, dict) else s.get("text", "")
                for s in snippets
            ).strip()
            if text:
                print(f"[Transcript API] Successfully fetched direct YouTube transcript ({len(text)} chars).")
                return text

    except Exception as e:
        print(f"[Transcript API] Layer 1 failed for {video_id}: {e}")

    # ── Layer 2: Invidious public API (works on AWS/cloud IPs) ───────────────
    # Invidious is an open-source YouTube frontend running on non-AWS servers,
    # so it bypasses the IP ban that cloud-hosted apps face with YouTube directly.
    INVIDIOUS_INSTANCES = [
        "https://invidious.nerdvpn.de",
        "https://invidious.privacyredirect.com",
        "https://inv.nadeko.net",
        "https://invidious.lunar.icu",
        "https://iv.melmac.space",
        "https://invidious.perennialte.ch",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Video-Assistant/1.0)",
        "Accept": "application/json",
    }

    for instance in INVIDIOUS_INSTANCES:
        try:
            api_url = f"{instance}/api/v1/captions/{video_id}"
            resp = requests.get(api_url, headers=headers, timeout=8)
            if resp.status_code != 200:
                continue

            data = resp.json()
            captions = data.get("captions", [])
            if not captions:
                print(f"[Invidious] No captions found on {instance} for {video_id}.")
                break  # No captions exist — no point trying other instances

            # Prefer English, then auto-generated, then first available
            chosen = None
            for cap in captions:
                label = cap.get("label", "").lower()
                if "english" in label and "auto" not in label:
                    chosen = cap
                    break
            if not chosen:
                for cap in captions:
                    if "english" in cap.get("label", "").lower():
                        chosen = cap
                        break
            if not chosen:
                chosen = captions[0]

            caption_url = f"{instance}{chosen['url']}&format=json"
            cap_resp = requests.get(caption_url, headers=headers, timeout=10)
            if cap_resp.status_code != 200:
                continue

            lines = cap_resp.json()
            text = " ".join(
                line.get("text", "").strip()
                for line in lines
                if line.get("text", "").strip()
            ).strip()

            if text:
                print(f"[Invidious] Fetched transcript via {instance} ({len(text)} chars).")
                return text

        except Exception as e:
            print(f"[Invidious] Instance {instance} failed: {e}")
            continue

    print(f"[Transcript] All transcript methods exhausted for {video_id}.")
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
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path: str, chunk_minutes: int = 8) -> tuple:
    audio = AudioSegment.from_wav(wav_path)
    total_duration_sec = len(audio) / 1000.0
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        if len(chunk) < 500 and len(chunks) > 0:
            # Skip tiny trailing fraction chunks if we already have chunks
            continue
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    if not chunks and len(audio) > 0:
        chunk_path = f"{wav_path}_chunk_0.wav"
        audio.export(chunk_path, format="wav")
        chunks.append(chunk_path)

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

