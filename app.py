import os
import re
import shutil
import base64
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FAVICON_PATH = os.path.join(ASSETS_DIR, "favicon.png") if os.path.exists(os.path.join(ASSETS_DIR, "favicon.png")) else "🎬"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon=FAVICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Refined & Modern) ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --surface: #FFFFFF;
    --surface-2: #F3EFE9;
    --border: #E7E1D8;
    --border-hover: #D5CDC1;
    --text: #26231F;
    --text-muted: #6B645B;
    --accent: #6D28D9;
    --accent-soft: #EDE9FE;
    --accent-2: #0E7490;
    --success: #15803D;
    --warning: #B45309;
}

/* ── Typography & Global Styles (Applied to .stApp to protect Streamlit icon fonts) ── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
}

/* ── Top Decoration Bar ── */
[data-testid="stDecoration"] {
    background: #6D28D9 !important;
    height: 3px !important;
}

code, pre, .mono-text {
    font-family: 'JetBrains Mono', monospace !important;
}

code {
    color: #6D28D9 !important;
    background: #EDE9FE !important;
    border-radius: 6px;
    padding: 0.1rem 0.4rem;
}

/* ── Hero Title (Main Page, Centered & Prominent) ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 4.5vw, 3.4rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    text-align: center;
    background: linear-gradient(135deg, #26231F 0%, #6D28D9 55%, #0E7490 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* ── Sidebar Branding ── */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.25rem;
    line-height: 1.2;
}

.sidebar-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    margin-bottom: 0.75rem;
}

/* ── Bordered Containers (Cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(38, 35, 31, 0.05);
    transition: all 0.2s ease-in-out;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 12px rgba(38, 35, 31, 0.08);
}

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-purple {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(109, 40, 217, 0.2);
}

.badge-cyan {
    background: #E0F2FE;
    color: var(--accent-2);
    border: 1px solid rgba(14, 116, 144, 0.2);
}

.badge-green {
    background: #DCFCE7;
    color: var(--success);
    border: 1px solid rgba(21, 128, 61, 0.2);
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0 16px;
    background-color: var(--surface);
    border: 1px solid var(--border);
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background-color: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 3px rgba(38, 35, 31, 0.04);
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    font-variant-numeric: tabular-nums lining-nums !important;
    color: var(--text) !important;
}

/* ── Sidebar Radio Options Spacing & Divider ── */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 12px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    margin-bottom: 6px !important;
    padding: 3px 0 !important;
    cursor: pointer;
}

[data-testid="stSidebar"] hr {
    margin: 0.75rem 0 !important;
}

/* ── Status Widget ── */
[data-testid="stStatusWidget"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(38, 35, 31, 0.05) !important;
    padding: 0.6rem 0.85rem !important;
}

[data-testid="stStatusWidget"] p {
    line-height: 1.65 !important;
    margin-bottom: 0.35rem !important;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-2); }
::-webkit-scrollbar-thumb { background: #D5CDC1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #BDB4A5; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "selected_source_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helper Functions ────────────────────────────────────────────────────────────
def card(title: str, body_md: str):
    """Render structured markdown inside a native bordered container."""
    with st.container(border=True):
        st.caption(title.upper())
        st.markdown(body_md)

def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.selected_source_name = ""
    st.rerun()

def format_duration(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins == 0:
        return f"{secs}s"
    return f"{mins}m {secs:02d}s"

def short_youtube_name(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return f"YouTube · {m.group(1)}" if m else "YouTube"

def get_logo_base64() -> str:
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

# Script detector for Indic / Arabic / Cyrillic (Devanagari, Arabic, Cyrillic, Bengali)
NON_LATIN_PATTERN = re.compile(r"[\u0900-\u097F\u0600-\u06FF\u0400-\u04FF\u0980-\u09FF]")

def looks_non_english(text: str, threshold: float = 0.12) -> bool:
    """Check if text contains a high proportion of Indic / non-Latin characters."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 80:
        return False
    return (sum(1 for c in letters if NON_LATIN_PATTERN.search(c)) / len(letters)) > threshold

def actions_to_df(text: str) -> pd.DataFrame:
    """Parse pipe-separated action items into a clean pandas DataFrame."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "none found" in line.lower() or "no action items" in line.lower():
            continue
        # Remove numbered or bullet prefixes (e.g., '1.', '1)', '-', '*')
        cleaned = re.sub(r"^\s*(\d+[\.\)]|\-|\*)\s*", "", line)
        parts = [p.strip() for p in cleaned.split("|")]
        if len(parts) >= 3:
            rows.append({"Task": parts[0], "Owner": parts[1], "Deadline": parts[2]})
        elif len(parts) == 2:
            rows.append({"Task": parts[0], "Owner": parts[1], "Deadline": "Not specified"})
        elif len(parts) == 1 and parts[0]:
            rows.append({"Task": parts[0], "Owner": "Viewer", "Deadline": "Not specified"})
    return pd.DataFrame(rows)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        col_logo, col_brand = st.columns([1, 3], vertical_alignment="center")
        col_logo.image(LOGO_PATH, width=54)
        with col_brand:
            st.markdown('<div class="sidebar-brand">AI Video Assistant</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-sub" style="margin-bottom:0">Video Intelligence & RAG Chat</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-brand">AI Video Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Video Intelligence & RAG Chat</div>', unsafe_allow_html=True)

    if st.session_state.result:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        if st.button("Start new analysis", icon=":material/refresh:", width="stretch", type="secondary"):
            reset_session()

    st.markdown("---")

    st.markdown("**1. Select input source**")
    source_type = st.radio(
        "Source Type",
        ["YouTube URL", "Upload File"],
        horizontal=True,
        label_visibility="collapsed"
    )

    youtube_url = ""
    uploaded_file = None
    if source_type == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube Video Link",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any public YouTube video link."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload from your device",
            type=["mp3", "wav", "mp4", "m4a", "mkv", "webm", "aac", "ogg", "flac"],
            help="Click 'Browse files' to choose an audio/video file from your computer."
        )

    st.markdown("**2. Transcription engine**")
    language = st.radio(
        "Transcription Engine",
        ["english", "hinglish"],
        format_func=lambda x: "English / Global (Groq Whisper)" if x == "english" else "Hinglish / Hindi (Sarvam AI)",
        help="Use Groq Whisper for ultra-fast English and multi-language transcription, or Sarvam AI for mixed Hindi/Hinglish audio.",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    submit_button = st.button("Run analysis", icon=":material/play_arrow:", width="stretch", type="primary")

# ─── Main Content Area ───────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Automated Transcription · Multi-Section Summaries · Interactive RAG Chat</div>', unsafe_allow_html=True)

# ─── Execute Pipeline When Form is Submitted ────────────────────────────────────
if submit_button:
    source_target = None
    source_display_name = ""
    temp_dir_to_clean = None

    if source_type == "YouTube URL":
        if not youtube_url.strip():
            st.error("Please provide a valid YouTube video URL.")
        else:
            source_target = youtube_url.strip()
            source_display_name = short_youtube_name(source_target)
    else:
        if not uploaded_file:
            st.error("Please select an audio or video file to upload.")
        else:
            temp_dir_to_clean = tempfile.mkdtemp(prefix="vid_upload_")
            saved_file_path = os.path.join(temp_dir_to_clean, uploaded_file.name)
            with open(saved_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_target = saved_file_path
            source_display_name = uploaded_file.name

    if source_target:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.selected_source_name = source_display_name

        try:
            with st.status("Processing video intelligence pipeline...", expanded=True) as status:
                st.markdown(":material/graphic_eq: **Step 1/5 · Audio preparation** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Normalizing and chunking audio</span>", unsafe_allow_html=True)
                chunks, duration_sec = process_input(source_target)

                step_2_elem = st.empty()
                step_2_elem.markdown(f":material/transcribe: **Step 2/5 · Transcription** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Transcribing chunk 1 of {len(chunks)} via {language.capitalize()}</span>", unsafe_allow_html=True)

                def update_progress(curr, total):
                    step_2_elem.markdown(f":material/transcribe: **Step 2/5 · Transcription** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Transcribing chunk {curr} of {total} via {language.capitalize()}</span>", unsafe_allow_html=True)

                transcript = transcribe_all(chunks, language, on_progress=update_progress)

                st.markdown(":material/title: **Step 3/5 · Title generation** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Writing concise video title</span>", unsafe_allow_html=True)
                title = generate_title(transcript)

                st.markdown(":material/summarize: **Step 4/5 · Content extraction** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Generating executive summary & key takeaways</span>", unsafe_allow_html=True)
                summary = summarize(transcript)
                extracted = extract_all(transcript)
                action_items, decisions, questions = (
                    extracted["action_items"],
                    extracted["key_decisions"],
                    extracted["open_questions"],
                )

                st.markdown(":material/hub: **Step 5/5 · Knowledge indexing** &nbsp;—&nbsp; <span style='color:var(--text-muted); font-size:0.88rem;'>Building ChromaDB vector embeddings for interactive chat</span>", unsafe_allow_html=True)
                rag_chain = build_rag_chain(transcript)

                status.update(label="Analysis complete · Video intelligence ready", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "chunks_count": len(chunks),
                "duration_sec": duration_sec,
                "language": language,
            }
            st.toast("Video analyzed successfully!", icon=":material/check_circle:")
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            with st.expander("View error details"):
                st.exception(e)
        finally:
            if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
                shutil.rmtree(temp_dir_to_clean, ignore_errors=True)

# ─── Display Results & Tabs ─────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Language check heuristic (accurate regex for non-Latin scripts)
    if r["language"] == "english" and looks_non_english(r["transcript"]):
        st.warning("High proportion of Indic / non-Latin characters detected in the transcript. If this recording contains Hindi or Hinglish dialogue, consider re-running with 'Hinglish / Hindi (Sarvam AI)' selected for optimal transcription.")

    # Header Card with Title
    with st.container(border=True):
        st.caption("SESSION TITLE")
        st.markdown(f"### {r['title']}")
        if st.session_state.selected_source_name:
            st.caption(f"Source: {st.session_state.selected_source_name}")

    # Metrics Row
    words_count = len(r["transcript"].split())
    char_count = len(r["transcript"])

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Words", f"{words_count:,}")
    with m_col2:
        st.metric("Characters", f"{char_count:,}")
    with m_col3:
        st.metric("Audio chunks", f"{r.get('chunks_count', 1)}")
    with m_col4:
        st.metric("Duration", format_duration(r.get("duration_sec", 0)))

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # Main Tabs
    tab_sum, tab_act, tab_dec, tab_q, tab_trans = st.tabs([
        "Summary",
        "Action items",
        "Key takeaways",
        "Questions & follow-ups",
        "Transcript & export"
    ])

    with tab_sum:
        card("Executive summary", r["summary"])

    with tab_act:
        df_actions = actions_to_df(r["action_items"])
        if not df_actions.empty:
            with st.container(border=True):
                st.caption("ACTION ITEMS & TASK OWNERS")
                st.dataframe(df_actions, hide_index=True, width="stretch")
        else:
            card("Action items & task owners", r["action_items"])

    with tab_dec:
        card("Key takeaways", r["key_decisions"])

    with tab_q:
        card("Questions & follow-ups", r["open_questions"])

    with tab_trans:
        with st.container(border=True):
            st.caption("FULL TRANSCRIPT")
            st.container(height=420).text(r["transcript"])

        # Export Buttons
        report_markdown = f"""# {r['title']}

Generated by AI Video Assistant
Source: {st.session_state.selected_source_name}

---

## Executive Summary
{r['summary']}

---

## Action Items & Owners
{r['action_items']}

---

## Key Takeaways
{r['key_decisions']}

---

## Questions & Follow-ups
{r['open_questions']}

---

## Full Transcript
{r['transcript']}
"""
        exp_col1, exp_col2, _ = st.columns([2, 2, 4])
        with exp_col1:
            st.download_button(
                label="Download transcript (.txt)",
                icon=":material/download:",
                data=r["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
                width="stretch"
            )
        with exp_col2:
            st.download_button(
                label="Download report (.md)",
                icon=":material/download:",
                data=report_markdown,
                file_name="video_report.md",
                mime="text/markdown",
                width="stretch"
            )

    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ─── Interactive Chat Section (Below Summary & Tabs) ───────────────────────────
    with st.container(border=True):
        st.markdown("### Interactive chat with video")
        st.caption("Ask questions, explore specific discussion points, or request further synthesis based on the video transcript.")

        # Quick question suggestion buttons
        st.markdown("**Suggested questions:**")
        q_cols = st.columns(3)
        suggested_prompt = None

        if q_cols[0].button("Key takeaways?", width="stretch"):
            suggested_prompt = "What were the key takeaways from this video?"
        if q_cols[1].button("3-bullet summary", width="stretch"):
            suggested_prompt = "Provide a concise 3-bullet point executive overview of the video."
        if q_cols[2].button("Unanswered questions", width="stretch"):
            suggested_prompt = "What questions or issues remained unresolved or need follow-up?"

        if st.session_state.chat_history:
            if st.button("Clear conversation", icon=":material/delete:", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

        # Render message history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Determine user query
        chat_input = st.chat_input("Ask anything about this video...")
        query_to_process = chat_input or suggested_prompt

        if query_to_process:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": query_to_process})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query_to_process)
                with st.chat_message("assistant"):
                    try:
                        # Attempt live token streaming with LCEL
                        response_stream = r["rag_chain"].stream(query_to_process)
                        assistant_response = st.write_stream(response_stream)
                    except Exception:
                        # Fallback invocation
                        assistant_response = ask_question(r["rag_chain"], query_to_process)
                        st.markdown(assistant_response)

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.rerun()

    # ─── New Analysis / Reset Button Below Chat ──────────────────────────────────
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        if st.button("Start new analysis", icon=":material/refresh:", width="stretch", type="secondary"):
            reset_session()

else:
    # Empty State (Initial Screen)
    logo_b64 = get_logo_base64()
    logo_img_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:110px; height:110px; border-radius:22px; margin-bottom:1.5rem; box-shadow: 0 4px 14px rgba(38,35,31,0.08);" alt="App Logo" />' if logo_b64 else ''

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:4rem 2rem; text-align:center; background:var(--surface); border:1px solid var(--border); border-radius:16px; box-shadow:0 1px 3px rgba(38,35,31,0.06); margin-top:1.5rem;">
        {logo_img_html}
        <div style="font-family:'Syne', sans-serif; font-size:1.6rem; font-weight:700; color:var(--text); margin-bottom:0.5rem;">
            Ready to transcribe & analyze
        </div>
        <div style="color:var(--text-muted); font-size:0.92rem; max-width:480px; line-height:1.6; margin-bottom:1.5rem;">
            Enter a YouTube link or upload a local audio/video file in the sidebar, select your language, and click <strong>Run analysis</strong>.
        </div>
        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; justify-content:center;">
            <span class="badge badge-purple">Whisper / Sarvam STT</span>
            <span class="badge badge-cyan">Structured summaries</span>
            <span class="badge badge-green">ChromaDB RAG chat</span>
        </div>
    </div>
    """, unsafe_allow_html=True)