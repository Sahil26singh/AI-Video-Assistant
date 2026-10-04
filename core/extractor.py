import re
import time
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_SINGLE_CALL_CHARS = 15_000   # ~4k tokens, safer for free models and rate limits

EXTRACT_PROMPT = """You are an expert video content analyst. Read the video transcript and produce EXACTLY these three sections, in this order, using these exact headers:

### ACTION ITEMS
### KEY TAKEAWAYS
### OPEN QUESTIONS

Rules for each section:
- ACTION ITEMS: tasks, to-dos, steps the speaker tells the viewer or team to do, or recommended next steps. One per line, numbered, in the format: Task | Owner | Deadline. Owner is the person named, or "Viewer" for lectures and tutorials. Deadline is "Not specified" if not mentioned.
- KEY TAKEAWAYS: decisions made, conclusions reached, and the main points to remember. One per line, numbered, specific and concise.
- OPEN QUESTIONS: unresolved questions, open problems, or topics that need follow-up or further study. One per line, numbered.

Global rules:
- Use ONLY information from the transcript. Never invent details.
- Do not repeat the same point across sections or within a section.
- If a section has nothing, write exactly: None found.
- Output the three sections and nothing else: no introduction, no closing remarks, no extra headers.
"""

SECTIONS = [
    ("action_items",   "ACTION ITEMS",   "No action items found."),
    ("key_decisions",  "KEY TAKEAWAYS",  "No key takeaways found."),
    ("open_questions", "OPEN QUESTIONS", "No open questions found."),
]


def get_llm(temperature=0.0):
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(
            model_name="openai/gpt-oss-120b",
            groq_api_key=groq_key,
            temperature=temperature,
        )
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        max_retries=10,
        timeout=120,
    )


def safe_invoke(chain, input_data, retries=5, delay=3):
    """Invoke chain with explicit exponential backoff for rate limits."""
    for attempt in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                wait = delay * (2 ** attempt)
                print(f"Rate limit reached. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise e
    return chain.invoke(input_data)


def _parse_sections(text: str) -> dict:
    # Accept #, ##, ###, **bold**, optional colon: any common header style
    header = r"^[ \t]*(?:#{{1,6}}[ \t]*)?\**[ \t]*{name}[ \t]*\**[ \t]*:?[ \t]*\**[ \t]*$"
    positions = []
    for key, name, _ in SECTIONS:
        m = re.search(header.format(name=re.escape(name)), text, re.IGNORECASE | re.MULTILINE)
        if m:
            positions.append((m.start(), m.end(), key))
    positions.sort()

    out = {}
    for i, (_, end, key) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[key] = text[end:stop].strip()

    result = {}
    for key, _, default in SECTIONS:
        content = out.get(key, "")
        # Only treat as empty if the WHOLE section is the "none found" line
        if not content or re.fullmatch(r"[\W_]*none found\.?[\W_]*", content, re.IGNORECASE):
            content = default
        result[key] = content

    # Model ignored the headers entirely: keep its answer instead of losing it
    if not positions and text.strip():
        result["key_decisions"] = text.strip()
    return result


def _extract_once(text: str) -> dict:
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", EXTRACT_PROMPT),
            ("human", "Transcript:\n{text}\n\nNow output the three sections exactly as instructed."),
        ])
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )
    return _parse_sections(safe_invoke(chain, {"text": text}))


def _merge_lines(blocks: list, default: str) -> str:
    seen, lines = set(), []
    for block in blocks:
        if block == default:
            continue
        for line in block.splitlines():
            clean = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()   # strip old numbering
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                lines.append(clean)
    return "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1)) or default


def extract_all(transcript: str) -> dict:
    """One LLM call for short/medium videos, chunked + merged for long ones."""
    if len(transcript) <= MAX_SINGLE_CALL_CHARS:
        return _extract_once(transcript)

    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_SINGLE_CALL_CHARS, chunk_overlap=500)
    parts = []
    for i, chunk in enumerate(splitter.split_text(transcript)):
        if i:
            time.sleep(2)
        parts.append(_extract_once(chunk))

    return {
        key: _merge_lines([p[key] for p in parts], default)
        for key, _, default in SECTIONS
    }