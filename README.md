# 🎬 AI Video Assistant

An end-to-end AI-powered meeting intelligence system that transcribes YouTube videos or local audio/video files, auto-generates structured summaries & action items, and lets you **Chat with your Meeting** using Retrieval-Augmented Generation (RAG).

---

## 🌟 Key Features

- **Audio Acquisition**: Download YouTube videos/audio (`yt-dlp`) or process local audio/video files (`pydub` + `ffmpeg`).
- **Speech-to-Text Transcription**: 
  - English: OpenAI Whisper (local execution)
  - Hinglish / Hindi: Sarvam AI STT Translation API
- **Meeting Summarisation & Extraction**:
  - Auto-generated Meeting Title
  - Executive Bullet-point Summary
  - Action Items & Task Owners
  - Key Decisions
  - Open Questions / Follow-ups
- **Interactive RAG Chat**: Ask questions directly about the video transcript powered by ChromaDB vector store and Groq / Mistral AI LLMs.
- **Dual Interfaces**:
  - 🎨 **Web UI**: Modern Streamlit application with live progress indicators & custom styling.
  - 🖥️ **CLI Mode**: Fast terminal interface.

---

## 🏗️ Project Architecture

```
AI-Video-Assistant/
├── app.py                   # Streamlit Web Application
├── main.py                  # Terminal CLI Pipeline
├── Requirements.txt         # Project Dependencies
├── .env                     # API Keys & Configuration
├── core/
│   ├── transcriber.py       # Whisper & Sarvam STT engine
│   ├── summarizer.py        # Title generation & summarization chain
│   ├── extractor.py         # Action item, decision & question extraction
│   ├── vector_store.py      # ChromaDB & HuggingFace embeddings
│   └── rag_engine.py        # RAG pipeline for meeting chat
└── utils/
    └── audio_processor.py   # YouTube download, audio conversion & chunking
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg**: Installed on system path or via WinGet / Homebrew.

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/AI-Video-Assistant.git
cd AI-Video-Assistant

# Create & activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r Requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

### 4. Running the Application

#### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```

#### Option B: Terminal CLI Interface
```bash
python main.py
```

---

## 🛠️ Tech Stack

- **Framework**: LangChain (LCEL)
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Providers**: Groq (`openai/gpt-oss-120b`) / Mistral AI (`mistral-small-latest`)
- **Speech-to-Text**: OpenAI Whisper / Sarvam AI
- **Frontend**: Streamlit
