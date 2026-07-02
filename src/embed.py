"""
Encodes chunks from both chunking strategies into embeddings using
BAAI/bge-small-en-v1.5, and saves vectors + metadata for retrieval.

Usage:
    python src/embed.py

Outputs:
    store/embeddings_fixed.npz    (vectors + metadata for fixed-size chunks)
    store/embeddings_section.npz  (vectors + metadata for section-aware chunks)

Note on bge models: BGE was trained so that QUERIES should be prefixed with
an instruction ("Represent this sentence for searching relevant passages: ")
but DOCUMENTS/chunks should NOT be prefixed. That instruction gets added
later in retrieve.py when embedding the user's question -- not here, since
here we're only embedding the chunks themselves (the documents).
"""

import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

STORE_DIR = "store"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_FILES = {
    "fixed": os.path.join(STORE_DIR, "chunks_fixed.json"),
    "section": os.path.join(STORE_DIR, "chunks_section.json"),
}


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunk_set(model, chunks, strategy_name):
    texts = [c["text"] for c in chunks]
    print(f"  Encoding {len(texts)} chunks ({strategy_name})...")

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product later
    )
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")

    return embeddings


def save_embeddings(path, embeddings, chunks):
    """
    Save vectors alongside their metadata (paper_id, section, text, id) so
    retrieval can return not just a similarity score but which paper/section
    a chunk came from.
    """
    ids = [c["id"] for c in chunks]
    paper_ids = [c["paper_id"] for c in chunks]
    sections = [c["section"] for c in chunks]
    texts = [c["text"] for c in chunks]

    np.savez_compressed(
        path,
        embeddings=embeddings,
        ids=np.array(ids, dtype=object),
        paper_ids=np.array(paper_ids, dtype=object),
        sections=np.array(sections, dtype=object),
        texts=np.array(texts, dtype=object),
    )
    print(f"  Saved -> {path}")


def main():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {dim}\n")

    for strategy_name, chunk_path in CHUNK_FILES.items():
        if not os.path.exists(chunk_path):
            print(f"WARNING: {chunk_path} not found, skipping ({strategy_name}).")
            continue

        print(f"--- {strategy_name} chunks ---")
        chunks = load_chunks(chunk_path)
        embeddings = embed_chunk_set(model, chunks, strategy_name)

        out_path = os.path.join(STORE_DIR, f"embeddings_{strategy_name}.npz")
        save_embeddings(out_path, embeddings, chunks)
        print()

    print("All done.")


if __name__ == "__main__":
    main()