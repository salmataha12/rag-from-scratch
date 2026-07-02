"""
Benchmarks NumPy (brute-force) vs FAISS retrieval speed over the eval
question set. This only exercises the retrieval step (embed query + search),
NOT generation -- so it makes zero Groq API calls and has no rate limit risk.

Usage:
    python experiments/retrieval_speed.py
    python experiments/retrieval_speed.py --chunking fixed
    python experiments/retrieval_speed.py --runs 3   (average over multiple passes)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieve_numpy import NumpyRetriever
from retrieve_faiss import FaissRetriever

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")
STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "store")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_retriever(retriever, questions, k, runs):
    """
    Run all questions through the retriever, `runs` times, and return
    per-query latency stats in milliseconds.
    """
    all_latencies = []

    for run in range(runs):
        for q in questions:
            start = time.perf_counter()
            retriever.retrieve(q["question"], k=k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_latencies.append(elapsed_ms)

    return {
        "avg_ms": sum(all_latencies) / len(all_latencies),
        "min_ms": min(all_latencies),
        "max_ms": max(all_latencies),
        "total_queries": len(all_latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunking", choices=["fixed", "section"], default="section")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=2,
                         help="Number of passes over the question set, averaged.")
    args = parser.parse_args()

    npz_path = os.path.join(STORE_DIR, f"embeddings_{args.chunking}.npz")
    questions = load_questions()

    print(f"Benchmarking retrieval speed | chunking={args.chunking} | "
          f"k={args.k} | {len(questions)} questions x {args.runs} runs "
          f"= {len(questions) * args.runs} queries per retriever\n")

    print("Loading NumPy retriever...")
    numpy_retriever = NumpyRetriever(npz_path)
    print("Loading FAISS retriever...")
    faiss_retriever = FaissRetriever(npz_path, model=numpy_retriever.model)  # share the loaded model

    print("\nBenchmarking NumPy (brute-force)...")
    numpy_stats = benchmark_retriever(numpy_retriever, questions, args.k, args.runs)

    print("Benchmarking FAISS...")
    faiss_stats = benchmark_retriever(faiss_retriever, questions, args.k, args.runs)

    print("\n" + "=" * 70)
    print("RETRIEVAL SPEED COMPARISON")
    print("=" * 70)
    print(f"{'Backend':<15}{'Avg (ms)':<15}{'Min (ms)':<15}{'Max (ms)':<15}")
    print("-" * 70)
    print(f"{'NumPy':<15}{numpy_stats['avg_ms']:<15.2f}"
          f"{numpy_stats['min_ms']:<15.2f}{numpy_stats['max_ms']:<15.2f}")
    print(f"{'FAISS':<15}{faiss_stats['avg_ms']:<15.2f}"
          f"{faiss_stats['min_ms']:<15.2f}{faiss_stats['max_ms']:<15.2f}")
    print("=" * 70)

    speedup = numpy_stats["avg_ms"] / faiss_stats["avg_ms"]
    print(f"\nFAISS is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} "
          f"than NumPy on average, at {len(numpy_retriever.embeddings)} chunks.")
    print("(Note: at this corpus size the difference is expected to be small --")
    print(" FAISS's advantage grows substantially at much larger scales, e.g. 100k+ vectors.)")

    summary_path = os.path.join(RESULTS_DIR, "retrieval_speed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunking": args.chunking,
            "k": args.k,
            "n_chunks": len(numpy_retriever.embeddings),
            "numpy": numpy_stats,
            "faiss": faiss_stats,
            "faiss_speedup_factor": speedup,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
