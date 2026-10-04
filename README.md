# AI Video Assistant

> **AI-powered Meeting & Video Assistant** that transcribes YouTube videos or local media (Whisper / Sarvam AI), generates structured summaries & action items, and features interactive RAG chat (ChromaDB + Groq/Mistral). Includes Streamlit Web UI and CLI.

---

## Overview

The application processes video or audio input through a multi-stage pipeline:
1. **Audio Acquisition & Normalization**: Downloads audio from YouTube URLs via `yt-dlp` or processes local uploads (`.mp4`, `.mp3`, `.wav`, etc.), standardizing them into 16 kHz mono WAV chunks using `pydub` and FFmpeg.
2. **Speech-to-Text**: Transcribes audio using either Groq Cloud's `whisper-large-v3` (with chunking and backoff retries) or Sarvam AI (`saaras:v2.5` for Hindi/Hinglish speech-to-text translation).
3. **Structured Intelligence Extraction**: Generates an executive summary, action items table (with assignees and deadlines), key takeaways, open questions, and a video title in a single pass.
4. **Interactive RAG Chat**: Indexes transcript chunks into an ephemeral, in-memory ChromaDB vector store using HuggingFace embeddings (`all-MiniLM-L6-v2`) and answers questions via LangChain LCEL with token streaming.

---

## Architecture

```
AI-Video-Assistant/
├── app.py                   # Streamlit web dashboard
├── main.py                  # Command-line interface
├── requirements.txt         # Project dependencies
├── .env                     # API keys and environment variables
├── core/
│   ├── transcriber.py       # Groq Whisper and Sarvam AI transcription
│   ├── summarizer.py        # Map-reduce summarization and title generator
│   ├── extractor.py         # Structured extraction (action items, takeaways, questions)
│   ├── vector_store.py      # In-memory ChromaDB and cached HuggingFace embeddings
│   └── rag_engine.py        # LCEL retrieval chain with token streaming
└── utils/
    └── audio_processor.py   # YouTube download, 16 kHz normalization, and chunking
```

---

## Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend** | Streamlit | Web interface with custom styling and streaming responses |
| **Orchestration** | LangChain LCEL | Composable runnables for extraction and retrieval |
| **STT (Global)** | Groq Whisper (`whisper-large-v3`) | Ultra-fast cloud speech-to-text |
| **STT (Indic)** | Sarvam AI (`saaras:v2.5`) | Hindi and Hinglish speech translation to English |
| **Primary LLM** | Groq Cloud (`openai/gpt-oss-120b`) | Primary inference engine for summaries and RAG |
| **Alternative LLM** | Mistral AI (`mistral-small-latest`) | Used when no Groq key is configured |
| **Vector Store** | ChromaDB | Ephemeral in-memory collections per session |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Local CPU sentence embeddings |
| **Audio Processing** | `pydub`, `yt-dlp`, FFmpeg | Format conversion, normalization, and chunking |

---

## Prerequisites

- **Python**: 3.10 or higher
- **FFmpeg**: Must be installed and accessible on your system `PATH`.
  - Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sahil26singh/AI-Video-Assistant.git
   cd AI-Video-Assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### Key Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Used for Groq Whisper transcription and primary LLM generation |
| `SARVAM_API_KEY` | Optional | Required only when selecting Hindi / Hinglish transcription |
| `MISTRAL_API_KEY` | Optional | Used when no Groq API key is configured |

---

## Usage

### Web Interface (Streamlit)

Launch the web app:
```bash
streamlit run app.py
```

1. Choose the input source (YouTube URL or local file upload).
2. Select the transcription engine (English / Global or Hinglish / Hindi).
3. Click **Run analysis**.
4. View structured tabs for the summary, action items table, takeaways, and full transcript.
5. Use the chat section below to query the video transcript with streaming responses.

### Command-Line Interface (CLI)

Run the terminal pipeline:
```bash
python main.py
```

Provide a YouTube URL or path to a local media file when prompted, and enter interactive Q&A mode once processing finishes.

---

## License

This project is licensed under the MIT License.
