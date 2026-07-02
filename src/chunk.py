"""
Chunks cleaned paper text (from data/extracted/) into retrieval-ready pieces.

Implements two strategies for later comparison:
  1. chunk_fixed()    -- naive sliding window, ignores document structure
  2. chunk_by_section() -- detects section headers (Abstract, Introduction,
                           Related Work, Method, Results, Conclusion, References, etc.)
                           and chunks within each section separately

Usage:
    python src/chunk.py

Outputs:
    store/chunks_fixed.json
    store/chunks_section.json
Each is a list of dicts: {id, paper_id, section, text}
"""

import os
import re
import json

INPUT_DIR = os.path.join("data", "extracted")
STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

# ---- Config ----
FIXED_CHUNK_WORDS = 350
FIXED_OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 15  # discard near-empty chunks (headings, captions, index terms)
                       # that can distort retrieval by scoring deceptively high
                       # on short/generic queries despite carrying little content

# Common section header names in CS/AI papers, in the order they usually appear.
# Matched case-insensitively, with or without a leading number (e.g. "2 Related Works").
SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work", "related works",
    "preliminaries",
    "motivation",
    "method", "methods", "methodology", "approach", "framework",
    "model", "architecture", "design",
    "experiment", "experiments", "experimental setup", "experimental results",
    "evaluation",
    "results", "analysis",
    "discussion",
    "applications",
    "limitations",
    "future work",
    "conclusion", "conclusions", "conclusion and perspectives",
    "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    "references",
    "appendix",
]

# Build one regex that matches a line consisting of (optional numbering) + a section keyword,
# optionally followed by more words (to catch things like "5.2.3 Disaggregation and Memorization Functions").
_keyword_pattern = "|".join(sorted((re.escape(k) for k in SECTION_KEYWORDS), key=len, reverse=True))
HEADER_REGEX = re.compile(
    rf"^\s*(?:\d+(?:\.\d+)*\.?\s+)?({_keyword_pattern})\b.*$",
    re.IGNORECASE,
)

# General fallback for numbered subsection headings that don't contain a known
# keyword (e.g. "3.2 Agent Framework", "2.1 Fine-tuning LLMs as Agents").
# These are detected by formatting instead: starts with a number, short line,
# and reads as a Title Case phrase rather than a full sentence.
NUMBERED_HEADER_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.{2,80})$")
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "to",
    "via", "using", "as", "from", "at", "by", "is", "are", "vs", "vs.",
}


def _is_title_case_heading(text: str) -> bool:
    """
    A short phrase counts as a Title Case heading if most of its meaningful
    words (excluding small connector words) start with a capital letter --
    e.g. "Agent Framework" or "Fine-tuning LLMs as Agents" -- as opposed to
    a normal sentence like "We propose a new method for this".
    """
    words = text.split()
    if not (1 <= len(words) <= 10):
        return False
    significant = [w for w in words if w.lower().strip(",:;") not in _STOPWORDS]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[0].isupper())
    return capitalized / len(significant) >= 0.8


def is_probable_header(line: str) -> bool:
    """
    A line is a probable section header if either:
    1. It matches a known section keyword (Introduction, Results, etc.), or
    2. It's a numbered heading that reads as a short Title Case phrase rather
       than a full sentence (catches subsection titles like "3.2 Agent Framework"
       that don't contain a generic keyword).
    In both cases the line must be short (headers are rarely full sentences)
    and must not end like a sentence (period, comma, semicolon), which helps
    avoid misfiring on numbered reference-list entries or in-text lists.
    """
    line = line.strip()
    if not line or len(line) > 80:
        return False

    if HEADER_REGEX.match(line):
        return True

    m = NUMBERED_HEADER_REGEX.match(line)
    if m:
        title = m.group(2).strip()
        if title.endswith((".", ",", ";", ":")):
            return False
        return _is_title_case_heading(title)

    return False


def split_into_sections(text: str):
    """
    Split full paper text into (section_name, section_text) tuples based on
    detected headers. Text before the first detected header is labeled 'front_matter'
    (title, authors, abstract-lead-in, etc.).
    """
    lines = text.split("\n")
    sections = []
    current_name = "front_matter"
    current_lines = []

    for line in lines:
        if is_probable_header(line):
            # flush current section
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return [(name, body) for name, body in sections if body.strip()]


def chunk_text_by_words(text: str, chunk_words: int, overlap_words: int):
    """
    Split text into overlapping chunks of ~chunk_words words each.
    Chunks shorter than MIN_CHUNK_WORDS are dropped -- these are usually
    headings, captions, or index terms with little real content, and can
    score deceptively high on short/generic queries, crowding out more
    informative chunks in the retrieved results.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk_words_slice = words[start:end]
        if len(chunk_words_slice) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words_slice))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_fixed(paper_id: str, text: str):
    """Strategy 1: ignore structure, just slide a fixed-size window over the whole text."""
    pieces = chunk_text_by_words(text, FIXED_CHUNK_WORDS, FIXED_OVERLAP_WORDS)
    return [
        {
            "id": f"{paper_id}_fixed_{i:03d}",
            "paper_id": paper_id,
            "section": "N/A",
            "text": piece,
        }
        for i, piece in enumerate(pieces)
    ]


def chunk_by_section(paper_id: str, text: str):
    """Strategy 2: split by detected section headers first, then chunk within each section."""
    sections = split_into_sections(text)
    all_chunks = []
    for section_name, section_text in sections:
        pieces = chunk_text_by_words(section_text, FIXED_CHUNK_WORDS, FIXED_OVERLAP_WORDS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "id": f"{paper_id}_{re.sub(r'[^a-zA-Z0-9]+', '_', section_name)[:30]}_{i:03d}",
                "paper_id": paper_id,
                "section": section_name,
                "text": piece,
            })
    return all_chunks


def process_all_papers():
    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} extracted papers to chunk.\n")

    fixed_chunks_all = []
    section_chunks_all = []

    for filename in txt_files:
        paper_id = filename.rsplit(".txt", 1)[0]
        path = os.path.join(INPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        fixed = chunk_fixed(paper_id, text)
        sectioned = chunk_by_section(paper_id, text)

        fixed_chunks_all.extend(fixed)
        section_chunks_all.extend(sectioned)

        print(f"  {paper_id[:50]:50s} fixed={len(fixed):3d}  section={len(sectioned):3d}")

    with open(os.path.join(STORE_DIR, "chunks_fixed.json"), "w", encoding="utf-8") as f:
        json.dump(fixed_chunks_all, f, ensure_ascii=False, indent=2)

    with open(os.path.join(STORE_DIR, "chunks_section.json"), "w", encoding="utf-8") as f:
        json.dump(section_chunks_all, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Fixed-size chunks:    {len(fixed_chunks_all)} -> store/chunks_fixed.json")
    print(f"  Section-aware chunks: {len(section_chunks_all)} -> store/chunks_section.json")


if __name__ == "__main__":
    process_all_papers()