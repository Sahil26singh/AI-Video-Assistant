# 🎬 AI Video Assistant

> **AI-Powered Video Intelligence & RAG Chat** — Transcribe, Summarize, Extract Action Items & Chat with any Video.

An end-to-end AI video intelligence system that processes YouTube videos or local audio/video files, transcribes speech with ultra-fast cloud STT, generates structured multi-section summaries & action items, and enables **Interactive RAG Chat** with live token streaming.

---

## 🌟 Key Features

- **Multi-Source Audio Acquisition**:
  - Download and extract audio directly from YouTube URLs via `yt-dlp`.
  - Upload local audio/video files (`.mp4`, `.mp3`, `.wav`, `.m4a`, `.mkv`, `.webm`, `.aac`, `.ogg`, `.flac`).
  - Automatic conversion to standardized 16kHz mono WAV chunks with exact duration measurement (`pydub` + `ffmpeg`).
  - Isolated temporary upload directories with automatic cleanup upon completion.

- **High-Speed Speech-to-Text (STT)**:
  - **English / Global**: Ultra-fast Groq Whisper Large v3 (`whisper-large-v3`) with automatic payload-splitting protection (<20MB chunks) and exponential backoff retry.
  - **Hinglish / Hindi**: Sarvam AI STT-Translate API (`saaras:v2.5`), converting spoken Hindi/Hinglish directly into English text.
  - Live chunk-by-chunk transcription progress indicators in the UI.

- **Automated Video Intelligence & Extraction**:
  - 🏷️ **Video Title**: Short, descriptive title generated from transcript context.
  - 📋 **Executive Summary**: Structured, bullet-pointed multi-part executive overview.
  - ✅ **Action Items & Task Owners**: Parsed into a structured table (`Task | Owner | Deadline`).
  - 🔑 **Key Takeaways**: Core conclusions, insights, and key takeaways.
  - ❓ **Questions & Follow-ups**: Unresolved questions, open topics, and recommended next steps.

- **Interactive RAG Chat**:
  - Chat directly with the video content using LangChain LCEL chains.
  - In-memory ephemeral ChromaDB vector collections with unique session UUIDs (zero disk residue, instant memory recycling).
  - Cached HuggingFace embeddings (`all-MiniLM-L6-v2`) on CPU.
  - Real-time token streaming with Groq (`openai/gpt-oss-120b`) / Mistral AI (`mistral-small-latest`).

- **Export & Reporting**:
  - ⬇️ Download full transcript as `.txt`.
  - ⬇️ Download complete structured intelligence report as Markdown (`.md`).

- **Dual Interfaces**:
  - 🎨 **Web UI**: Modern Streamlit dashboard with custom typography (`Inter` / `Syne`), metrics cards, Material icons, and responsive layout.
  - 🖥️ **CLI Mode**: Fast terminal entry point via `python main.py`.

---

## 🏗️ Project Structure

```
AI-Video-Assistant/
├── app.py                   # Streamlit Web Application (Modern Dark UI)
├── main.py                  # Terminal CLI Pipeline
├── requirements.txt         # Project Dependencies
├── .env                     # API Keys & Model Configurations
├── core/
│   ├── transcriber.py       # Groq Whisper & Sarvam AI STT with progress callbacks
│   ├── summarizer.py        # Map-reduce video summarizer & title generator
│   ├── extractor.py         # Single-pass structured extractor (Action Items, Takeaways, Questions)
│   ├── vector_store.py      # In-memory ChromaDB vector store & cached HuggingFace embeddings
│   └── rag_engine.py        # RAG pipeline with grounded answering & token streaming
└── utils/
    └── audio_processor.py   # YouTube download, 16kHz mono normalization & chunking
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg**: Ensure `ffmpeg` is available on your system path (e.g., via WinGet, Chocolatey, or Homebrew).

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Sahil26singh/AI-Video-Assistant.git
cd AI-Video-Assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

| Key | Description | Required For |
|---|---|---|
| `GROQ_API_KEY` | Groq Cloud API key | Fast Whisper STT, LLM summarization & RAG |
| `SARVAM_API_KEY` | Sarvam AI API key | Hinglish / Hindi speech-to-text translation |
| `MISTRAL_API_KEY` | Mistral AI API key | LLM provider used when no Groq key is configured |

---

## 🖥️ Running the Application

### Option A: Streamlit Web Dashboard (Recommended)

```bash
streamlit run app.py
```

1. Select your input source (**YouTube URL** or **Upload File**).
2. Choose your transcription engine (**English / Global** or **Hinglish / Hindi**).
3. Click **⚡ Run Analysis**.
4. Explore executive summaries, action items table, key takeaways, and chat interactively with the video below.

### Option B: Terminal CLI Interface

```bash
python main.py
```

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://python.langchain.com/) (LCEL)
- **STT Engines**: [Groq Whisper Large v3](https://console.groq.com/) & [Sarvam AI](https://sarvam.ai/)
- **LLMs**: Groq (`openai/gpt-oss-120b`) / Mistral AI (`mistral-small-latest`, used when no Groq key is configured)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Ephemeral in-memory collections)
- **Embeddings**: HuggingFace [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Audio Processing**: [pydub](https://github.com/jiaaro/pydub), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [FFmpeg](https://ffmpeg.org/)
