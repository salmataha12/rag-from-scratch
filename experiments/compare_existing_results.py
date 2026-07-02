"""
Compares two already-saved eval result files (from eval/run_eval.py) without
re-running the eval -- no API calls, just reads the existing JSON results.

Use this when you've already run eval/run_eval.py separately for both
chunking strategies (e.g. results_fixed_faiss.json and results_section_faiss.json
already exist) and just want the comparison table/summary.

Usage:
    python experiments/compare_existing_results.py
    python experiments/compare_existing_results.py --retriever numpy
"""

import os
import json
import argparse

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")


def load_results(chunking: str, retriever: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"results_{chunking}_{retriever}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 'python eval/run_eval.py --chunking {chunking} "
            f"--retriever {retriever}' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["numpy", "faiss"], default="faiss")
    args = parser.parse_args()

    fixed_results = load_results("fixed", args.retriever)
    section_results = load_results("section", args.retriever)

    print("=" * 70)
    print("CHUNKING STRATEGY COMPARISON (from existing results, no re-run)")
    print(f"Retriever: {args.retriever}")
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

    summary_path = os.path.join(RESULTS_DIR, "chunking_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "retriever": args.retriever,
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
