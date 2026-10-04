import os
import time
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Sarvam STT-Translate API configuration
SARVAM_PIECE_SECONDS = 25
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _call_groq_api(audio_path: str, retries: int = 3) -> str:
    """Call Groq Whisper transcription API with retry logic."""
    client = get_groq_client()
    for attempt in range(retries):
        try:
            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), f.read()),
                    model=GROQ_MODEL,
                    response_format="text",
                    temperature=0.0,
                )
            text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
            return text
        except Exception as e:
            print(f"Groq API attempt {attempt+1}/{retries} failed on {audio_path}: {e}")
            if attempt == retries - 1:
                raise e
            time.sleep(2 * (attempt + 1))
    return ""


def transcribe_chunk_groq(chunk_path: str) -> str:
    """
    Transcribe audio chunk using Groq Cloud's ultra-fast Whisper Large v3.
    Automatically splits files larger than 20MB to stay well within Groq's 25MB limit.
    """
    try:
        audio = AudioSegment.from_file(chunk_path)
        if len(audio) < 300:  # Less than 300ms is too short
            return ""
    except Exception as e:
        print(f"Warning: Could not read audio chunk {chunk_path}: {e}")
        return ""

    file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
    if file_size_mb <= 20:
        return _call_groq_api(chunk_path)

    # If chunk is oversized (>20MB), slice into 5-minute sub-pieces
    print(f"⚠️ Chunk {chunk_path} is {file_size_mb:.1f}MB (>20MB). Splitting into smaller pieces for Groq...")
    sub_ms = 5 * 60 * 1000
    combined_text = ""
    for idx, start in enumerate(range(0, len(audio), sub_ms)):
        sub_piece = audio[start: start + sub_ms]
        sub_path = f"{chunk_path}_sub_{idx}.wav"
        sub_piece.export(sub_path, format="wav")
        try:
            piece_text = _call_groq_api(sub_path)
            combined_text += piece_text + " "
        finally:
            if os.path.exists(sub_path):
                os.remove(sub_path)

    return combined_text.strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route chunk to Groq Whisper or Sarvam depending on language choice:
    - english  -> Groq Whisper Large v3 (Fast API)
    - hinglish -> Sarvam AI (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_groq(chunk_path)


def transcribe_all(chunks: list, language: str = "english", on_progress=None) -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Groq Whisper (whisper-large-v3)"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks, 1):
        print(f"Transcribing chunk {i}/{len(chunks)}...")
        if on_progress:
            try:
                on_progress(i, len(chunks))
            except Exception:
                pass
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")

    cleaned = full_transcript.strip()
    if not cleaned:
        raise ValueError("No audible speech could be detected in the provided audio. Please make sure the audio file contains audible conversation.")
    return cleaned
