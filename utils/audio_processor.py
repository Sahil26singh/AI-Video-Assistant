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
            all_subs = {**auto, **subs}
            if not all_subs:
                return ""

            target_lang = None
            for l in ["en", "en-US", "en-GB", "hi"]:
                if l in all_subs:
                    target_lang = l
                    break
            if not target_lang:
                target_lang = next(iter(all_subs.keys()))

            formats = all_subs.get(target_lang, [])
            json_fmt = next((f for f in formats if f.get("ext") == "json3"), None)
            if not json_fmt and formats:
                json_fmt = formats[0]

            if not json_fmt or "url" not in json_fmt:
                return ""

            resp = requests.get(json_fmt["url"], timeout=10)
            if resp.status_code != 200:
                return ""

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
                    print(f"[Transcript API] Successfully fetched direct transcript ({len(text)} chars).")
                    return text
            elif resp.text:
                import re
                cleaned = re.sub(r"<[^>]+>", " ", resp.text)
                cleaned = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d+ --> \d{2}:\d{2}:\d{2}[.,]\d+", " ", cleaned)
                text = " ".join(cleaned.split()).strip()
                if text:
                    print(f"[Transcript API] Successfully parsed caption stream ({len(text)} chars).")
                    return text
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

