"""
Extracts and cleans text from all PDFs in data/papers/.
Outputs one cleaned .txt file per paper into data/extracted/.

Handles:
- Repeated headers/footers (lines that appear on almost every page get stripped)
- Broken line joins (PDF text extraction often splits sentences across lines)

Usage:
    python src/extract.py
"""

import os
import re
import pdfplumber
from collections import Counter

INPUT_DIR = os.path.join("data", "papers")
OUTPUT_DIR = os.path.join("data", "extracted")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _words_to_lines_text(words):
    """
    Given a list of word dicts (already restricted to one column or one page),
    group them into lines by vertical position and join with gap-based spacing.
    """
    if not words:
        return ""

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current_line = [words[0]]
    for w in words[1:]:
        if abs(w["top"] - current_line[-1]["top"]) < 3:  # same line
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    page_lines = []
    for line in lines:
        line = sorted(line, key=lambda w: w["x0"])
        line_text = line[0]["text"]
        for prev, curr in zip(line, line[1:]):
            gap = curr["x0"] - prev["x1"]
            line_text += (" " if gap > 1.0 else "") + curr["text"]
        page_lines.append(line_text)

    return "\n".join(page_lines)


def _detect_column_gutter(words, page_width):
    """
    Look for a vertical strip of blank space roughly in the middle third of
    the page where no word sits -- that's the gap between two columns.
    Returns the x-position of the gutter's center, or None if the page
    looks single-column (no consistent central gap).
    """
    resolution = 2  # points per bucket
    n_buckets = int(page_width / resolution) + 1
    covered = [False] * n_buckets

    for w in words:
        start = max(0, int(w["x0"] / resolution))
        end = min(n_buckets - 1, int(w["x1"] / resolution))
        for i in range(start, end + 1):
            covered[i] = True

    # only search for a gutter within the central band of the page
    search_lo = int((page_width * 0.3) / resolution)
    search_hi = int((page_width * 0.7) / resolution)

    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for i in range(search_lo, search_hi + 1):
        if not covered[i]:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start

    min_gutter_points = 8  # minimum blank width to count as a real column gap
    if best_start is not None and best_len * resolution >= min_gutter_points:
        return (best_start + best_len / 2) * resolution
    return None


def extract_raw_pages(pdf_path):
    """
    Return a list of raw text strings, one per page.

    Handles two problems:
    1. Missing spaces between words (some PDFs don't carry explicit space
       characters) -- fixed by rebuilding text from word bounding boxes with
       gap-based spacing instead of using extract_text() directly.
    2. Two-column layouts reading in the wrong order (words from both columns
       at the same page height getting merged into one scrambled line) --
       fixed by detecting the blank vertical gutter between columns and
       reading the left column fully, then the right column, instead of
       reading straight across the page width.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
            if not words:
                pages.append("")
                continue

            gutter_x = _detect_column_gutter(words, page.width)

            if gutter_x is not None:
                left_words = [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter_x]
                right_words = [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter_x]
                page_text = _words_to_lines_text(left_words) + "\n" + _words_to_lines_text(right_words)
            else:
                page_text = _words_to_lines_text(words)

            pages.append(page_text)
    return pages


def find_repeated_lines(pages, min_page_fraction=0.4):
    """
    Find lines that repeat across many pages -- these are almost always
    headers/footers (e.g. paper title on every page, page numbers, conference name).
    """
    line_counts = Counter()
    for page_text in pages:
        # only look at the first 2 and last 2 lines of each page (where headers/footers live)
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        candidates = lines[:2] + lines[-2:]
        for line in set(candidates):
            line_counts[line] += 1

    threshold = max(2, int(len(pages) * min_page_fraction))
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    return repeated


def clean_pages(pages, repeated_lines):
    """Remove repeated header/footer lines and join broken sentences."""
    cleaned_pages = []
    for page_text in pages:
        lines = page_text.split("\n")
        kept_lines = [l for l in lines if l.strip() not in repeated_lines]
        cleaned_pages.append("\n".join(kept_lines))

    full_text = "\n".join(cleaned_pages)

    # Join lines that were broken mid-sentence: a line that doesn't end in
    # punctuation, followed by a lowercase letter, is probably a wrapped line.
    full_text = re.sub(r"-\n(?=[a-z])", "", full_text)  # rejoin hyphenated words split across lines
    full_text = re.sub(r"(?<![.!?:\n])\n(?=[a-z])", " ", full_text)  # rejoin wrapped sentences

    # Collapse excessive blank lines/whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)

    return full_text.strip()


def process_all_papers():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to process.\n")

    success, failed = 0, 0

    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        out_filename = filename.rsplit(".pdf", 1)[0] + ".txt"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        if os.path.exists(out_path):
            print(f"  Skipping (already extracted): {filename}")
            success += 1
            continue

        try:
            pages = extract_raw_pages(pdf_path)
            if not any(p.strip() for p in pages):
                print(f"  WARNING: no extractable text in {filename} (likely scanned/image PDF)")
                failed += 1
                continue

            repeated_lines = find_repeated_lines(pages)
            cleaned_text = clean_pages(pages, repeated_lines)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"  Extracted: {filename} ({len(pages)} pages, {len(cleaned_text)} chars)")
            success += 1

        except Exception as e:
            print(f"  FAILED: {filename} -> {e}")
            failed += 1

    print(f"\nDone. {success} succeeded, {failed} failed.")
    print(f"Cleaned text saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    process_all_papers()
