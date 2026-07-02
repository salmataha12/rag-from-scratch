"""
Manual (NumPy, brute-force) retrieval over the embedded chunks.

No FAISS, no vector DB -- just embedding the query and computing cosine
similarity against every stored chunk vector directly with NumPy. Since
embeddings were saved normalized (unit length) in embed.py, cosine
similarity reduces to a plain dot product.

Usage (as a module):
    from src.retrieve_numpy import NumpyRetriever
    retriever = NumpyRetriever("store/embeddings_section.npz")
    results = retriever.retrieve("What dataset was used for evaluation?", k=5)

Usage (standalone test):
    python src/retrieve_numpy.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained so that QUERIES (not documents) should be prefixed
# with this instruction for best retrieval performance. The chunks themselves
# were embedded WITHOUT this prefix in embed.py.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class NumpyRetriever:
    def __init__(self, npz_path: str, model=None):
        """
        Load embeddings + metadata from a .npz file produced by embed.py,
        and prepare a SentenceTransformer for encoding queries.
        """
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings = data["embeddings"]  # shape: (n_chunks, dim), already normalized
        self.ids = data["ids"]
        self.paper_ids = data["paper_ids"]
        self.sections = data["sections"]
        self.texts = data["texts"]

        self.model = model if model is not None else SentenceTransformer(MODEL_NAME)

        print(f"Loaded {len(self.embeddings)} chunks from {npz_path} "
              f"(dim={self.embeddings.shape[1]})")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the BGE instruction prefix, normalized to unit length."""
        prefixed = BGE_QUERY_INSTRUCTION + query
        vec = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
        return vec

    def retrieve(self, query: str, k: int = 5):
        """
        Return the top-k most similar chunks to the query.

        Since both query and chunk vectors are unit-normalized, cosine
        similarity is just the dot product -- no separate normalization
        step needed at search time.
        """
        query_vec = self._embed_query(query)

        # dot product of the query vector against every stored chunk vector at once
        similarities = self.embeddings @ query_vec  # shape: (n_chunks,)

        top_k_idx = np.argsort(-similarities)[:k]  # sort descending, take top k

        results = []
        for idx in top_k_idx:
            results.append({
                "score": float(similarities[idx]),
                "id": str(self.ids[idx]),
                "paper_id": str(self.paper_ids[idx]),
                "section": str(self.sections[idx]),
                "text": str(self.texts[idx]),
            })
        return results


if __name__ == "__main__":
    # Quick manual test
    retriever = NumpyRetriever("store/embeddings_section.npz")

    test_query = "What is retrieval augmented generation?"
    print(f"\nQuery: {test_query}\n")

    results = retriever.retrieve(test_query, k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  paper={r['paper_id'][:40]}  section={r['section'][:40]}")
        print(f"    {r['text'][:150]}...")
        print()
