"""
Runs the eval set against both chunking strategies (fixed vs section-aware)
using the same retriever, and prints a side-by-side comparison table.

This reuses eval/run_eval.py's run_eval() function directly rather than
duplicating scoring logic -- it just calls it twice with different chunking
values and compares the results.

Usage:
    python experiments/chunking_comparison.py                  (default: faiss retriever)
    python experiments/chunking_comparison.py --retriever numpy
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_eval import run_eval

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")


def load_results(chunking: str, retriever: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"results_{chunking}_{retriever}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["numpy", "faiss"], default="faiss")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print(f"Retriever: {args.retriever} | k: {args.k}")
    print("=" * 70)

    print("\n--- Running eval with FIXED-SIZE chunking ---\n")
    run_eval(retriever=args.retriever, chunking="fixed", k=args.k)

    print("\n--- Running eval with SECTION-AWARE chunking ---\n")
    run_eval(retriever=args.retriever, chunking="section", k=args.k)

    fixed_results = load_results("fixed", args.retriever)
    section_results = load_results("section", args.retriever)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20}{'Retrieval Accuracy':<22}{'Answer Accuracy':<20}")
    print("-" * 70)
    print(f"{'Fixed-size':<20}{fixed_results['retrieval_accuracy']:<22.1f}"
          f"{fixed_results['answer_accuracy']:<20.1f}")
    print(f"{'Section-aware':<20}{section_results['retrieval_accuracy']:<22.1f}"
          f"{section_results['answer_accuracy']:<20.1f}")
    print("=" * 70)

    retrieval_delta = section_results['retrieval_accuracy'] - fixed_results['retrieval_accuracy']
    answer_delta = section_results['answer_accuracy'] - fixed_results['answer_accuracy']
    print(f"\nSection-aware vs fixed-size: "
          f"{retrieval_delta:+.1f}% retrieval accuracy, {answer_delta:+.1f}% answer accuracy")

    # Save the comparison summary for easy inclusion in the README
    summary_path = os.path.join(RESULTS_DIR, "chunking_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "retriever": args.retriever,
            "k": args.k,
            "fixed": {
                "retrieval_accuracy": fixed_results["retrieval_accuracy"],
                "answer_accuracy": fixed_results["answer_accuracy"],
            },
            "section": {
                "retrieval_accuracy": section_results["retrieval_accuracy"],
                "answer_accuracy": section_results["answer_accuracy"],
            },
            "retrieval_accuracy_delta": retrieval_delta,
            "answer_accuracy_delta": answer_delta,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
