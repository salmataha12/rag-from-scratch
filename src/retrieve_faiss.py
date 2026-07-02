"""
FAISS-backed retrieval over the embedded chunks -- same interface as
retrieve_numpy.py, so the two can be swapped and benchmarked against
each other in experiments/retrieval_speed.py.

Uses IndexFlatIP (exact inner-product search). Since embeddings are
unit-normalized (done in embed.py), inner product == cosine similarity,
so this returns mathematically identical results to the NumPy version --
the difference is purely in retrieval speed/implementation, not accuracy.
(FAISS also offers approximate indexes like IndexHNSWFlat for much larger
datasets where exact search becomes too slow; not needed at this corpus
size, but worth knowing for the design doc.)

Usage (as a module):
    from src.retrieve_faiss import FaissRetriever
    retriever = FaissRetriever("store/embeddings_section.npz")
    results = retriever.retrieve("What dataset was used for evaluation?", k=5)

Usage (standalone test):
    python src/retrieve_faiss.py
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FaissRetriever:
    def __init__(self, npz_path: str, model=None):
        data = np.load(npz_path, allow_pickle=True)
        embeddings = data["embeddings"].astype("float32")  # FAISS requires float32
        self.ids = data["ids"]
        self.paper_ids = data["paper_ids"]
        self.sections = data["sections"]
        self.texts = data["texts"]

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # exact inner-product search
        self.index.add(embeddings)

        self.model = model if model is not None else SentenceTransformer(MODEL_NAME)

        print(f"Loaded {self.index.ntotal} chunks from {npz_path} into FAISS index (dim={dim})")

    def _embed_query(self, query: str) -> np.ndarray:
        prefixed = BGE_QUERY_INSTRUCTION + query
        vec = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32").reshape(1, -1)  # FAISS expects a 2D batch, even for one query

    def retrieve(self, query: str, k: int = 5):
        query_vec = self._embed_query(query)

        scores, indices = self.index.search(query_vec, k)  # both shape: (1, k)
        scores, indices = scores[0], indices[0]

        results = []
        for score, idx in zip(scores, indices):
            results.append({
                "score": float(score),
                "id": str(self.ids[idx]),
                "paper_id": str(self.paper_ids[idx]),
                "section": str(self.sections[idx]),
                "text": str(self.texts[idx]),
            })
        return results


if __name__ == "__main__":
    retriever = FaissRetriever("store/embeddings_section.npz")

    test_query = "What is retrieval augmented generation?"
    print(f"\nQuery: {test_query}\n")

    results = retriever.retrieve(test_query, k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  paper={r['paper_id'][:40]}  section={r['section'][:40]}")
        print(f"    {r['text'][:150]}...")
        print()
