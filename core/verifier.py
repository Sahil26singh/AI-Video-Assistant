import re
import string
from typing import Dict, List, Tuple

PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’«»—–…")


def normalize_text(text: str) -> str:
    """Lowercase and strip ASCII & Unicode punctuation and excess whitespace for robust text matching."""
    if not text:
        return ""
    clean = text.lower().translate(PUNCT_TABLE)
    return " ".join(clean.split())


def is_empty_section(text: str) -> bool:
    """Check if section has no items or contains an empty indicator."""
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    return (
        t in ("none", "none.", "none found", "none found.", "no action items found.", "no highlights found.", "no open questions found.")
        or re.match(r"^no\s+(?:highlights|questions|action\s+items|items)?\s*found\.?$", t) is not None
        or re.match(r"^none\s*found\.?$", t) is not None
    )


def is_quote_grounded(quote: str, transcript: str, threshold: float = 0.70) -> bool:
    """
    Check if an extracted quote or claim is genuinely grounded in the transcript:
    1. Direct exact or normalized substring match.
    2. Contiguous word sequence n-gram matching.
    3. Token overlap ratio for slightly paraphrased or punctuation-variant quotes.
    """
    if not quote or not transcript:
        return False

    norm_quote = normalize_text(quote)
    norm_transcript = normalize_text(transcript)

    # 1. Exact normalized substring match
    if norm_quote in norm_transcript:
        return True

    words = norm_quote.split()
    if len(words) < 3:
        return norm_quote in norm_transcript

    # 2. Check 3-word or 4-word contiguous n-grams from the quote inside transcript
    seq_len = min(4, len(words))
    total_sequences = len(words) - seq_len + 1

    if total_sequences > 0:
        matching_sequences = sum(
            1 for i in range(total_sequences)
            if " ".join(words[i : i + seq_len]) in norm_transcript
        )
        if matching_sequences / total_sequences >= threshold:
            return True

    # 3. Word set overlap for longer sentences
    quote_set = set(words)
    transcript_words = set(norm_transcript.split())
    overlap = len(quote_set & transcript_words) / len(quote_set)
    return overlap >= 0.80


def verify_extraction(extracted_data: dict, transcript: str) -> dict:
    """
    Evaluate groundedness for all extracted sections (Action Items, Highlights, Questions).
    Returns overall faithfulness metrics, grounding percentage, and per-item status.
    """
    sections = [
        ("action_items", "Action Items"),
        ("key_decisions", "Highlights"),
        ("open_questions", "Open Questions"),
    ]

    total = 0
    verified = 0
    section_results = {}

    for key, label in sections:
        raw_text = extracted_data.get(key, "")
        items = []

        if not is_empty_section(raw_text):
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Strip leading numbering
                clean = re.sub(r"^\s*(\*{0,2}\d+[.)]\*{0,2}|-|\*)\s*", "", line)
                clean_no_bold = clean.strip().strip("*").strip()
                parts = [p.strip() for p in clean.split("|")]

                if key == "action_items" and len(parts) >= 4:
                    claim = parts[0].strip().strip("*").strip()
                    quote = " | ".join(parts[3:]).strip().strip('"\'“”‘’')
                elif key != "action_items" and len(parts) >= 2:
                    claim = parts[0].strip().strip("*").strip()
                    quote = " | ".join(parts[1:]).strip().strip('"\'“”‘’')
                elif len(parts) >= 2:
                    claim = parts[0].strip().strip("*").strip()
                    quote = parts[-1].strip().strip('"\'“”‘’')
                else:
                    quote_match = re.search(r'^(.*?)\s*[-:]?\s*["“](.+?)["”]\s*$', clean)
                    if quote_match and len(quote_match.group(1).strip()) > 3:
                        claim = quote_match.group(1).strip().strip("*").strip()
                        quote = quote_match.group(2).strip().strip('"\'“”‘’')
                    else:
                        claim = clean_no_bold
                        quote = ""

                total += 1
                is_valid = is_quote_grounded(quote, transcript) if quote else False

                # Fallback: check if the claim itself has high overlap in transcript
                if not is_valid and is_quote_grounded(claim, transcript, threshold=0.75):
                    is_valid = True

                if is_valid:
                    verified += 1

                items.append({
                    "claim": claim,
                    "quote": quote,
                    "is_grounded": is_valid,
                    "raw": line,
                })

        section_results[key] = items

    grounding_score = round((verified / total * 100), 1) if total > 0 else 100.0

    if grounding_score >= 80:
        badge = "Verified Grounded"
        color = "green"
    elif grounding_score >= 50:
        badge = "Partially Grounded"
        color = "orange"
    else:
        badge = "Low Grounding"
        color = "red"

    return {
        "total_claims": total,
        "verified_claims": verified,
        "grounding_score": grounding_score,
        "badge": badge,
        "color": color,
        "sections": section_results,
    }
