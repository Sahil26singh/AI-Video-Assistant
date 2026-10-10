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
    Attempt to fetch official or auto-generated YouTube transcripts directly.
    Works instantaneously (under 1 second), uses zero audio bandwidth,
    and reliably works on cloud platforms (AWS / Streamlit Cloud) without 403 Forbidden.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return ""

    try:
        import requests
        from youtube_transcript_api import YouTubeTranscriptApi
        import http.cookiejar

        # Mimic standard browser headers so cloud datacenters (AWS) are not immediately flagged
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
        })

        ytta = YouTubeTranscriptApi(http_client=session)

        snippets = None
        # Try fetching any available transcript (manual, auto-generated, any language)
        if hasattr(ytta, "list"):
            try:
                t_list = ytta.list(video_id)
                first_t = next(iter(t_list), None)
                if first_t:
                    snippets = first_t.fetch()
            except Exception:
                pass

        if not snippets:
            if hasattr(ytta, "fetch"):
                snippets = ytta.fetch(video_id)
            else:
                snippets = YouTubeTranscriptApi.get_transcript(video_id)

        text = " ".join(getattr(s, "text", "") if not isinstance(s, dict) else s.get("text", "") for s in snippets)
        cleaned = text.strip()
        if cleaned:
            print(f"[Transcript API] Successfully fetched direct YouTube transcript ({len(cleaned)} chars).")
            return cleaned
    except Exception as e:
        print(f"[Transcript API] Direct transcript not available for {video_id}: {e}")
    return ""


def download_youtube_audio(url: str) -> str:
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    ffmpeg_exe = shutil.which("ffmpeg")
    ffmpeg_dir = os.path.dirname(ffmpeg_exe) if ffmpeg_exe else None

    # Check for cookies file (needed for Streamlit Cloud / cloud datacenter IPs)
    cookie_file = None
    if os.path.exists("cookies.txt"):
        cookie_file = "cookies.txt"
    else:
        cookies_content = os.getenv("YOUTUBE_COOKIES")
        if not cookies_content:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "YOUTUBE_COOKIES" in st.secrets:
                    cookies_content = st.secrets["YOUTUBE_COOKIES"]
            except Exception:
                pass
        if cookies_content and isinstance(cookies_content, str):
            clean_content = cookies_content.strip()
            # Strip enclosing quotes if accidentally passed
            if clean_content.startswith('"""') and clean_content.endswith('"""'):
                clean_content = clean_content[3:-3].strip()
            elif clean_content.startswith("'''") and clean_content.endswith("'''"):
                clean_content = clean_content[3:-3].strip()

            # Ensure proper Netscape cookie file header
            if not clean_content.startswith("# Netscape") and not clean_content.startswith("# HTTP"):
                clean_content = "# Netscape HTTP Cookie File\n" + clean_content

            # Only write if it contains valid YouTube cookie data
            if ".youtube.com" in clean_content:
                cookie_file = os.path.join(DOWNLOAD_DIR, "yt_cookies.txt")
                with open(cookie_file, "w", encoding="utf-8") as f:
                    f.write(clean_content + "\n")

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

    if cookie_file and os.path.exists(cookie_file):
        try:
            import http.cookiejar
            cj = http.cookiejar.MozillaCookieJar(cookie_file)
            cj.load(ignore_discard=True, ignore_expires=True)
            ydl_opts["cookiefile"] = cookie_file
            print(f"[yt-dlp] Using YouTube cookies from {cookie_file} ({len(cj)} cookies loaded).")
        except Exception as ce:
            print(f"[yt-dlp] Skipping {cookie_file} (invalid cookie format: {ce}).")
    else:
        print("[yt-dlp] Note: No cookies provided. Cloud hosting IPs may require YOUTUBE_COOKIES.")

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

