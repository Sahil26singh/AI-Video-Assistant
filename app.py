import os
import re
import shutil
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input, get_youtube_transcript
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FAVICON, LOGO = os.path.join(ASSETS, "favicon.png"), os.path.join(ASSETS, "logo.png")

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon=FAVICON if os.path.exists(FAVICON) else "🎬",
    layout="wide",
)

# ─── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');
:root { 
    --text: #1E1B18; 
    --muted: #6B645B; 
    --border: #E5DFD7; 
    --card-bg: #FFFFFF;
    --metric-bg: #F4EFEA;
    --accent: #6D28D9; 
    --soft: #EDE9FE; 
}

.stApp { font-family:'Inter',sans-serif; color:var(--text); }
h1,h2,h3,h4,h5 { font-family:'Syne',sans-serif !important; letter-spacing:-0.02em; color:var(--text) !important; }
[data-testid="stDecoration"] { background:var(--accent) !important; height:3px !important; }
code { color:var(--accent) !important; background:var(--soft) !important; border-radius:6px; }

/* hero */
.hero { 
    font: 800 clamp(2.2rem, 4.2vw, 3.2rem)/1.15 'Syne', sans-serif; 
    text-align: center;
    color: var(--text);
    letter-spacing: -0.025em;
}
.hero-sub { text-align:center; color:var(--muted); font-size:.95rem; margin:.4rem 0 1.5rem; }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card-bg); border:1px solid var(--border);
    border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.08); transition:.2s; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:var(--accent); box-shadow:0 4px 12px rgba(0,0,0,.15); }

/* plain section labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.15rem 0 0.5rem 0;
    line-height: 1.4;
}
.section-label .sub {
    display: block;
    font-size: 0.8rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: normal;
    color: var(--muted);
    opacity: 0.85;
    margin-top: 2px;
}

/* headings inside generated content */
.stMarkdown h2, .stMarkdown h3 { background:linear-gradient(135deg,rgba(109,40,217,.08),rgba(109,40,217,.02));
    border-left:4px solid var(--accent); padding:.4rem .8rem !important; border-radius:0 8px 8px 0; }
.stMarkdown h2, .stMarkdown h3 { margin:.5rem 0 .6rem !important; line-height:1.3 !important; }
.stMarkdown h4, .stMarkdown h5 { color:var(--accent) !important; margin-top:.85rem !important; }
[data-testid="stCaptionContainer"] { margin-top:.6rem; }

/* pill tabs */
[role="tablist"] { gap:8px; border-bottom:1px solid var(--border); padding-bottom:8px; }
[role="tab"] { height:40px; padding:0 18px; border-radius:999px; background:var(--card-bg); border:1px solid var(--border); }
[role="tab"] p { font-size:0.95rem; font-weight:600; color:var(--text); margin:0; }
[role="tab"]:hover { background:var(--soft); border-color:var(--accent); }
[role="tab"][aria-selected="true"] { background:var(--accent); border-color:var(--accent); }
[role="tab"][aria-selected="true"] p { color:#fff !important; }
[role="tablist"] .react-aria-SelectionIndicator, [role="tablist"] [data-baseweb="tab-highlight"],
[role="tablist"] [data-baseweb="tab-border"] { display:none !important; }
[role="tab"], [role="tab"]:focus, [role="tab"]:focus-visible, [role="tab"]:active { outline:none !important; box-shadow:none !important; }

/* metrics */
[data-testid="stMetric"] { background:var(--card-bg); border:1px solid var(--border); border-radius:10px; padding:1rem; display:flex; flex-direction:column; align-items:center; text-align:center; gap:.9rem; }
[data-testid="stMetric"] > div, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { justify-content:center; text-align:center; width:100%; }
[data-testid="stMetricLabel"] { background:var(--metric-bg); color:var(--muted) !important; font-size:.72rem !important; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; padding:.2rem .55rem; border-radius:4px; display:inline-block; }
[data-testid="stMetricValue"] { font-size:1.4rem !important; font-weight:600 !important; font-variant-numeric:tabular-nums; line-height:1.3 !important; color:var(--text) !important; }

/* sidebar */
[data-testid="stSidebar"][aria-expanded="true"] { min-width:340px; max-width:340px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:10px; }
</style>
""", unsafe_allow_html=True)

for k, v in {"result": None, "chat": [], "source": ""}.items():
    st.session_state.setdefault(k, v)


# ─── Helpers ────────────────────────────────────────────────────────────────────
def banner(title, sub="", **kwargs):
    s = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(f'<div class="section-label">{title}{s}</div>', unsafe_allow_html=True)


def card(title, body, **kwargs):
    with st.container(border=True):
        banner(title)
        st.markdown(body)


def reset():
    st.session_state.update(result=None, chat=[], source="")
    st.rerun()


def fmt_duration(sec):
    if not sec or sec <= 0:
        return "N/A"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def youtube_name(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return f"YouTube · {m.group(1)}" if m else "YouTube"


NON_LATIN = re.compile(r"[ऀ-ॿ؀-ۿЀ-ӿঀ-৿]")


def looks_non_english(text, threshold=0.12):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 80 and sum(bool(NON_LATIN.match(c)) for c in letters) / len(letters) > threshold


def actions_to_df(text):
    """Parse 'task | owner | deadline' lines into a DataFrame."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "none found" in line.lower() or "no action items" in line.lower():
            continue
        p = [x.strip() for x in re.sub(r"^\s*(\d+[.)]|-|\*)\s*", "", line).split("|")]
        if p[0]:
            rows.append({"Task": p[0],
                         "Owner": p[1] if len(p) > 1 else "Viewer",
                         "Deadline": p[2] if len(p) > 2 else "Not specified"})
    return pd.DataFrame(rows)


def run_audio_pipeline(source_path: str, lang: str):
    """Normalize, chunk, and transcribe an audio file with progress updates."""
    st.markdown(":material/graphic_eq: **Step 1/5 · Audio preparation** — normalizing and chunking")
    chunks, duration_sec = process_input(source_path)
    prog = st.empty()

    def on_progress(cur, total):
        prog.markdown(f":material/transcribe: **Step 2/5 · Transcription** — chunk {cur} of {total} via {lang.capitalize()}")

    on_progress(1, len(chunks))
    text = transcribe_all(chunks, lang, on_progress=on_progress)
    return text, len(chunks), duration_sec


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists(LOGO):
        c_logo, c_name = st.columns([1, 3], vertical_alignment="center")
        c_logo.image(LOGO, width=68)
        c_name.markdown("### AI Video Assistant\nVideo Intelligence & RAG Chat")
    else:
        st.markdown("### AI Video Assistant\nVideo Intelligence & RAG Chat")

    if st.session_state.result:
        if st.button("New analysis", icon=":material/refresh:", width="stretch"):
            reset()
        st.divider()

    banner("1. Select input source")
    source_type = st.radio("Source", ["YouTube URL", "Upload File"], horizontal=True, label_visibility="collapsed", index=1)
    youtube_url, uploaded = "", None
    if source_type == "YouTube URL":
        youtube_url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...",
                                    help="Paste any public YouTube link.")
    else:
        uploaded = st.file_uploader("Audio / video file", help="Choose a file from your computer.",
                                    type=["mp3", "wav", "mp4", "m4a", "mkv", "webm", "aac", "ogg", "flac"])

    banner("2. Transcription engine")
    language = st.radio(
        "Engine", ["english", "hinglish"], label_visibility="collapsed",
        format_func=lambda x: "English / Global (Groq Whisper)" if x == "english" else "Hinglish / Hindi (Sarvam AI)",
        help="Groq Whisper: fast English & multilingual. Sarvam AI: mixed Hindi/Hinglish audio.",
    )
    run = st.button("Run analysis", icon=":material/play_arrow:", width="stretch", type="primary")

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Automated Transcription · Multi-Section Summaries · Interactive RAG Chat</div>',
            unsafe_allow_html=True)

# ─── Pipeline ───────────────────────────────────────────────────────────────────
if run:
    target, name, tmp = None, "", None
    if source_type == "YouTube URL":
        if youtube_url.strip():
            target, name = youtube_url.strip(), youtube_name(youtube_url)
        else:
            st.error("Please provide a valid YouTube URL.")
    elif uploaded:
        tmp = tempfile.mkdtemp(prefix="vid_upload_")
        safe_name = os.path.basename(uploaded.name)
        target = os.path.join(tmp, safe_name)
        with open(target, "wb") as f:
            f.write(uploaded.getbuffer())
        name = safe_name
    else:
        st.error("Please upload an audio or video file.")

    if target:
        st.session_state.update(result=None, chat=[], source=name)
        try:
            with st.status("Video processing pipeline...", expanded=True) as status:
                transcript = ""
                chunks_count = 0
                duration = 0.0
                mode = "audio"

                # 1. Fast track: Attempt direct YouTube transcript first (bypasses 403 Forbidden & quotas)
                if source_type == "YouTube URL":
                    st.markdown(":material/subtitles: Checking for native YouTube transcript...")
                    direct_transcript = get_youtube_transcript(target)
                    if direct_transcript:
                        transcript = direct_transcript
                        chunks_count = 1
                        mode = "captions"
                        st.markdown(":material/check_circle: **Native transcript loaded** — bypassed audio download & quotas!")

                # 2. Fallback: Full audio extraction & AI transcription pipeline
                if not transcript:
                    if source_type == "YouTube URL":
                        st.markdown(":material/graphic_eq: Falling back to audio extraction & speech model...")
                        try:
                            transcript, chunks_count, duration = run_audio_pipeline(target, language)
                        except Exception as e:
                            status.update(label="YouTube captions unavailable", state="error", expanded=False)
                            st.warning(
                                "**No captions available. Upload the file instead.**\n\n"
                                "This YouTube video has captions disabled or is restricted by YouTube (e.g., private, age-restricted, or cloud IP limits).\n\n"
                                "**Recommended path:** Switch to **Upload File** in the sidebar and upload the file directly. "
                                "File upload does not depend on YouTube and processes media directly.",
                                icon=":material/subtitles_off:"
                            )
                            with st.expander("Technical details"):
                                st.code(str(e))
                            st.stop()
                    else:
                        transcript, chunks_count, duration = run_audio_pipeline(target, language)

                st.markdown(":material/title: **Step 3/5 · Title** — generating video title")
                title = generate_title(transcript)

                st.markdown(":material/summarize: **Step 4/5 · Content extraction** — summary & key takeaways")
                summary = summarize(transcript)
                ex = extract_all(transcript)

                st.markdown(":material/hub: **Step 5/5 · Knowledge indexing** — building ChromaDB embeddings")
                rag_chain = build_rag_chain(transcript)
                status.update(label="Analysis complete · Video intelligence ready", state="complete", expanded=False)

            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "action_items": ex["action_items"], "key_decisions": ex["key_decisions"],
                "open_questions": ex["open_questions"], "rag_chain": rag_chain,
                "chunks": chunks_count, "duration": duration, "language": language,
                "mode": mode,
            }
            st.toast("Analysis complete", icon=":material/check_circle:")
            st.rerun()
        except Exception as e:
            err_text = str(e).lower()
            if "rate limit" in err_text or "429" in err_text:
                st.error(
                    "**LLM Rate Limit Reached.** Groq/Mistral rate limit was reached. "
                    "Please wait 30 seconds and try again.",
                    icon=":material/hourglass_empty:"
                )
            elif "groq_api_key" in err_text or "api key" in err_text:
                st.error(f"**API Key Configuration Error:** {e}", icon=":material/key_off:")
            else:
                st.error(f"Analysis could not be completed: {e}")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

# ─── Empty state ────────────────────────────────────────────────────────────────
r = st.session_state.result
if not r:
    with st.container(border=True):
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem 2rem">
          <div style="font:700 1.55rem 'Syne',sans-serif;margin-bottom:.4rem;color:var(--text)">
            Ready to transcribe &amp; analyze
          </div>
          <div style="color:var(--muted);max-width:500px;margin:0 auto 1.2rem;line-height:1.6;font-size:0.95rem">
            Upload an audio/video file or enter a YouTube link in the sidebar to extract structured intelligence, action items, and interactive Q&amp;A.
          </div>
          <div style="color:var(--accent);font-weight:600;font-size:0.84rem;letter-spacing:0.04em;text-transform:uppercase">
            Summary &nbsp;·&nbsp; Action Items &nbsp;·&nbsp; Interactive Q&amp;A
          </div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Results ────────────────────────────────────────────────────────────────────
if r.get("mode") == "captions" and r["language"] == "hinglish":
    st.info(
        "Direct YouTube captions were loaded. Sarvam AI's speech translation runs on uploaded audio tracks; "
        "to translate spoken Hindi/Hinglish directly from audio with Sarvam, download the audio and use **Upload File** in the sidebar.",
        icon=":material/info:"
    )
elif r["language"] == "english" and looks_non_english(r["transcript"]):
    if r.get("mode") == "captions":
        st.info(
            "**Hindi / Devanagari text detected in YouTube transcript.** "
            "YouTube provided native Hindi subtitles. If you want spoken Hindi translated to English audio text, "
            "download the audio file and upload it with the **'Hinglish / Hindi (Sarvam AI)'** engine.",
            icon=":material/info:"
        )
    else:
        st.warning(
            "High proportion of non-Latin characters detected. If this is Hindi/Hinglish audio, "
            "re-run with 'Hinglish / Hindi (Sarvam AI)' for better accuracy.",
            icon=":material/warning:"
        )

with st.container(border=True):
    banner("Session title")
    st.markdown(f"### {r['title']}")
    st.caption(f"Source: {st.session_state.source}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Words", f"{len(r['transcript'].split()):,}")
m2.metric("Characters", f"{len(r['transcript']):,}")
if r.get("mode") == "captions":
    m3.metric("Source Mode", "Direct Captions")
    est_read_min = max(1, round(len(r['transcript'].split()) / 180))
    m4.metric("Est. Read Time", f"~{est_read_min} min")
else:
    m3.metric("Audio chunks", r["chunks"])
    m4.metric("Duration", fmt_duration(r["duration"]))
st.write("")

t_sum, t_act, t_key, t_q, t_tr = st.tabs(
    ["Summary", "Action items", "Key takeaways", "Questions & follow-ups", "Transcript & export"])

with t_sum:
    card("Executive summary", r["summary"])

with t_act:
    df = actions_to_df(r["action_items"])
    if df.empty:
        card("Action items & task owners", r["action_items"])
    else:
        with st.container(border=True):
            banner("Action items & task owners")
            st.dataframe(df, hide_index=True, width="stretch")

with t_key:
    card("Key takeaways", r["key_decisions"])

with t_q:
    card("Questions & follow-ups", r["open_questions"])

with t_tr:
    with st.container(border=True):
        banner("Full transcript")
        st.container(height=400).text(r["transcript"])
    report = (
        f"# {r['title']}\n\nGenerated by AI Video Assistant\nSource: {st.session_state.source}\n\n---\n\n"
        f"## Executive summary\n{r['summary']}\n\n---\n\n## Action items & task owners\n{r['action_items']}\n\n---\n\n"
        f"## Key takeaways\n{r['key_decisions']}\n\n---\n\n## Questions & follow-ups\n{r['open_questions']}\n\n---\n\n"
        f"## Full transcript\n{r['transcript']}\n"
    )
    d1, d2, _ = st.columns([2, 2, 4])
    d1.download_button("Download transcript (.txt)", r["transcript"], "transcript.txt", "text/plain",
                       icon=":material/download:", width="stretch")
    d2.download_button("Download report (.md)", report, "video_report.md", "text/markdown",
                       icon=":material/download:", width="stretch")

# ─── Chat ───────────────────────────────────────────────────────────────────────
st.divider()
with st.container(border=True):
    banner("Interactive chat with video",
           "Ask questions, explore specific points, or request further synthesis from the transcript")

    suggestions = {
        "Key takeaways": "What were the key takeaways from this video?",
        "3-bullet summary": "Provide a concise 3-bullet executive overview of the video.",
        "Unanswered questions": "What questions or issues remained unresolved or need follow-up?",
    }
    cols = st.columns(len(suggestions))
    suggested_prompt = next((q for c, (label, q) in zip(cols, suggestions.items())
                            if c.button(label, width="stretch")), None)

    if st.session_state.chat and st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.chat = []
        st.rerun()

    chat_box = st.container()

    prompt = st.chat_input("Ask anything about this video...") or suggested_prompt

    with chat_box:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(r["rag_chain"].stream(prompt))
                except Exception:
                    answer = ask_question(r["rag_chain"], prompt)
                    st.markdown(answer)
            st.session_state.chat.append({"role": "user", "content": prompt})
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.rerun()