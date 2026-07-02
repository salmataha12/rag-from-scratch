"""
Generates a draft ground-truth eval set by extracting each paper's abstract
(from the front_matter section) and asking Groq to produce factual
question/answer pairs grounded ONLY in that abstract.

This is LLM-assisted eval generation, not fully manual authorship -- the
output should be spot-checked (a fast skim of the generated Q&A pairs, not
the full papers) before being trusted as ground truth. Document this
methodology note in your README.

Usage:
    python eval/generate_eval_questions.py

Output:
    eval/questions.json  (draft -- review before using with run_eval.py)
"""

import os
import sys
import json
import re
import glob

# allow importing from src/ when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from groq import Groq
from chunk import split_into_sections  # reuse the section-splitting logic from chunk.py

load_dotenv()

EXTRACTED_DIR = os.path.join("data", "extracted")
OUTPUT_PATH = os.path.join("eval", "questions.json")
GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Pick a spread of papers across topics -- these are arXiv ID PREFIXES, not
# full filenames (the real filenames also include a truncated title, which
# varies). find_paper_file() below matches on the prefix.
SELECTED_PAPER_PREFIXES = [
    "2309.15217v2",  # Ragas
    "2304.12244v3",  # WizardLM
    "2410.20777v1",  # KD-LoRA
    "2508.09534v1",  # Improving Dense Passage Retrieval with Multiple Positive Passages
    "2402.11651v2",  # Learning From Failure
    "2210.15133v1",  # Retrieval Oriented Masking
    "2512.06196v1",  # ARCANE
    "2410.16954v2",  # LoRA-C
]


def find_paper_file(prefix: str) -> str:
    """Find the extracted .txt file whose name starts with the given arXiv ID prefix."""
    matches = glob.glob(os.path.join(EXTRACTED_DIR, f"{prefix}*.txt"))
    if not matches:
        return None
    return matches[0]

QUESTION_GEN_PROMPT = """Based ONLY on the following paper abstract, generate 3 factual
question-answer pairs that could be answered using just this abstract. Do not use any
outside knowledge about the paper.

Requirements:
- Questions should be specific and factual (e.g. "What dataset was used?", "What method
  does this paper propose?", "What was the key result reported?")
- Answers must be short and precise (a phrase or short sentence)
- Include 1-3 "expected_keywords" per question: short exact words/phrases that MUST
  appear in a correct answer (e.g. a dataset name, a metric value, a method name)

Return ONLY valid JSON, no preamble, no markdown formatting, in this exact structure:
{{
  "questions": [
    {{"question": "...", "expected_answer": "...", "expected_keywords": ["...", "..."]}},
    {{"question": "...", "expected_answer": "...", "expected_keywords": ["...", "..."]}},
    {{"question": "...", "expected_answer": "...", "expected_keywords": ["...", "..."]}}
  ]
}}

Abstract:
{abstract}
"""


def get_abstract_text(filepath: str, max_chars: int = 2500) -> str:
    """
    Pull the front_matter section (title, authors, abstract) from the
    extracted text file, truncated to a reasonable length so we're mostly
    capturing the abstract and not spilling into the introduction.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_into_sections(text)
    front_matter = ""
    for name, body in sections:
        if name == "front_matter":
            front_matter = body
            break

    return front_matter[:max_chars]


def generate_questions_for_paper(prefix: str) -> list:
    filepath = find_paper_file(prefix)
    if filepath is None:
        print(f"  WARNING: no extracted file found matching prefix '{prefix}', skipping.")
        return []

    paper_id = os.path.basename(filepath).rsplit(".txt", 1)[0]  # full real paper_id
    abstract = get_abstract_text(filepath)
    if not abstract.strip():
        print(f"  WARNING: no front_matter/abstract text found for {paper_id}, skipping.")
        return []

    prompt = QUESTION_GEN_PROMPT.format(abstract=abstract)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)  # strip markdown fences if present

    try:
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
    except json.JSONDecodeError:
        print(f"  WARNING: could not parse JSON for {paper_id}. Raw output:\n{raw}\n")
        return []

    for q in questions:
        q["expected_paper"] = paper_id

    return questions


def main():
    all_questions = []

    for prefix in SELECTED_PAPER_PREFIXES:
        print(f"Generating questions for: {prefix}")
        questions = generate_questions_for_paper(prefix)
        print(f"  -> {len(questions)} questions generated")
        all_questions.extend(questions)

    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_questions)} draft questions saved to {OUTPUT_PATH}")
    print("IMPORTANT: skim through this file before using it -- LLM-generated")
    print("questions should be spot-checked, not trusted blindly as ground truth.")


if __name__ == "__main__":
    main()