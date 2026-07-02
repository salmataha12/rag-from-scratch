"""
Single entry point for the RAG pipeline: ties retrieval + generation
together, with toggleable chunking strategy and retriever backend.

This is what eval/run_eval.py, experiments/*, and app.py all call into --
one function, one place to change behavior.

Usage (as a module):
    from src.pipeline import ask
    result = ask("What dataset was used for evaluation?", retriever="faiss", chunking="section")

Usage (standalone test):
    python src/pipeline.py
"""

import os

from retrieve_numpy import NumpyRetriever
from retrieve_faiss import FaissRetriever
from generate import generate_answer

STORE_DIR = "store"

# Cache loaded retrievers so repeated calls (e.g. during eval, which asks
# 25-30 questions in a row) don't reload the embedding model / reopen the
# .npz file every single time.
_retriever_cache = {}


def _get_retriever(retriever: str, chunking: str):
    """
    Return a cached retriever instance for the given (retriever, chunking)
    combination, creating it if it doesn't exist yet.
    """
    key = (retriever, chunking)
    if key in _retriever_cache:
        return _retriever_cache[key]

    npz_path = os.path.join(STORE_DIR, f"embeddings_{chunking}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No embeddings found at {npz_path}. "
            f"Run 'python src/embed.py' first, or check the 'chunking' argument "
            f"(expected 'fixed' or 'section')."
        )

    if retriever == "numpy":
        instance = NumpyRetriever(npz_path)
    elif retriever == "faiss":
        instance = FaissRetriever(npz_path)
    else:
        raise ValueError(f"Unknown retriever: {retriever!r}. Expected 'numpy' or 'faiss'.")

    _retriever_cache[key] = instance
    return instance


def ask(question: str, retriever: str = "faiss", chunking: str = "section", k: int = 5) -> dict:
    """
    Run the full pipeline: retrieve top-k chunks, then generate a grounded answer.

    Args:
        question:  the user's question
        retriever: "numpy" or "faiss"
        chunking:  "fixed" or "section" -- which chunk set / embedding file to use
        k:         number of chunks to retrieve

    Returns:
        {
            "answer": str,
            "sources": [{"paper_id": ..., "section": ...}, ...],
            "retrieved_chunks": [...]  -- full retrieval results, useful for eval/debugging
        }
    """
    retriever_instance = _get_retriever(retriever, chunking)
    retrieved_chunks = retriever_instance.retrieve(question, k=k)
    result = generate_answer(question, retrieved_chunks)
    result["retrieved_chunks"] = retrieved_chunks
    return result


if __name__ == "__main__":
    test_question = "What is retrieval augmented generation?"
    print(f"Question: {test_question}\n")

    result = ask(test_question, retriever="faiss", chunking="section", k=5)

    print("Answer:")
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['paper_id']} | {s['section']}")
