"""
Runs the full pipeline against eval/questions.json and scores:
  1. Retrieval accuracy -- did the expected paper appear among retrieved sources?
  2. Answer accuracy   -- did the generated answer contain the expected keywords?

Usage:
    python eval/run_eval.py                              (defaults: faiss, section, k=5)
    python eval/run_eval.py --retriever numpy             (change retriever backend)
    python eval/run_eval.py --chunking fixed               (change chunking strategy)
    python eval/run_eval.py --k 3                            (change top-k)

Output:
    Prints a per-question pass/fail table and overall accuracy scores.
    Saves detailed results to eval/results_<chunking>_<retriever>.json
"""

import os
import sys
import json
import argparse
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import ask

QUESTIONS_PATH = os.path.join("eval", "questions.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_retrieval_hit(expected_paper: str, retrieved_chunks: list) -> bool:
    """True if the expected paper appears anywhere among the retrieved chunks."""
    retrieved_papers = {c["paper_id"] for c in retrieved_chunks}
    return expected_paper in retrieved_papers


def check_answer_hit(expected_keywords: list, answer_text: str) -> bool:
    """
    True if AT LEAST ONE expected keyword appears in the generated answer
    (case-insensitive). Using "at least one" rather than "all" because
    keywords are often near-synonyms (e.g. "LLMs" / "Large language models")
    and the model may phrase things differently while still being correct.

    Numeric/percentage keywords get special handling: a keyword like "99%"
    should also match a more precise answer like "99.35%" -- the model isn't
    wrong for being more specific than the expected keyword, so we match on
    the leading digits rather than requiring an exact substring.
    """
    answer_lower = answer_text.lower()

    for kw in expected_keywords:
        kw_lower = kw.lower().strip()

        percent_match = re.match(r"^(\d+)\s*%$", kw_lower)
        if percent_match:
            digits = percent_match.group(1)
            # match "99%", "99.35%", "99.0 %", etc. -- same leading integer part
            pattern = rf"\b{digits}(\.\d+)?\s*%"
            if re.search(pattern, answer_lower):
                return True
        else:
            if kw_lower in answer_lower:
                return True

    return False


def run_eval(retriever: str, chunking: str, k: int):
    questions = load_questions()
    print(f"Running eval: {len(questions)} questions | retriever={retriever} | "
          f"chunking={chunking} | k={k}\n")

    results = []
    retrieval_hits, answer_hits = 0, 0

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        expected_paper = q["expected_paper"]
        expected_keywords = q["expected_keywords"]

        result = ask(question_text, retriever=retriever, chunking=chunking, k=k)

        retrieval_ok = check_retrieval_hit(expected_paper, result["retrieved_chunks"])
        answer_ok = check_answer_hit(expected_keywords, result["answer"])

        retrieval_hits += int(retrieval_ok)
        answer_hits += int(answer_ok)

        status_r = "PASS" if retrieval_ok else "FAIL"
        status_a = "PASS" if answer_ok else "FAIL"
        print(f"[{i:2d}/{len(questions)}] retrieval={status_r}  answer={status_a}  "
              f"| {question_text[:60]}")

        results.append({
            "question": question_text,
            "expected_paper": expected_paper,
            "expected_keywords": expected_keywords,
            "generated_answer": result["answer"],
            "retrieval_hit": retrieval_ok,
            "answer_hit": answer_ok,
            "retrieved_papers": list({c["paper_id"] for c in result["retrieved_chunks"]}),
        })

    n = len(questions)
    retrieval_acc = retrieval_hits / n * 100
    answer_acc = answer_hits / n * 100

    print(f"\n{'='*60}")
    print(f"Retrieval accuracy: {retrieval_hits}/{n} ({retrieval_acc:.1f}%)")
    print(f"Answer accuracy:    {answer_hits}/{n} ({answer_acc:.1f}%)")
    print(f"{'='*60}")

    out_path = os.path.join("eval", f"results_{chunking}_{retriever}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"retriever": retriever, "chunking": chunking, "k": k},
            "retrieval_accuracy": retrieval_acc,
            "answer_accuracy": answer_acc,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["numpy", "faiss"], default="faiss")
    parser.add_argument("--chunking", choices=["fixed", "section"], default="section")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    run_eval(args.retriever, args.chunking, args.k)