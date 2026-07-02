"""
Streamlit UI for the RAG-from-scratch system. Lets you ask questions against
the paper corpus and toggle chunking strategy / retriever backend live to
see how they affect results.

Also supports uploading your own PDF: it's processed live through the same
extract -> chunk -> embed pipeline used to build the corpus, and merged into
retrieval for that session only (nothing is saved to disk or added to the
persistent corpus/eval set) -- this proves the pipeline is genuinely general
purpose, not hardcoded to the 59 demo papers.

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import re

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import _get_retriever
from generate import generate_answer
from extract import extract_raw_pages, find_repeated_lines, clean_pages
from chunk import chunk_fixed, chunk_by_section

st.set_page_config(page_title="RAG From Scratch", page_icon="📚", layout="wide")

st.title("📚 RAG From Scratch")
st.caption(
    "A retrieval-augmented generation system built from scratch (manual chunking, "
    "embeddings, cosine similarity, and FAISS) over a corpus of 59 arXiv papers on "
    "RAG, retrieval, LLMs, and multi-agent systems."
)

# ---- Sidebar: strategy controls ----
with st.sidebar:
    st.header("Settings")

    chunking = st.selectbox(
        "Chunking strategy",
        options=["section", "fixed"],
        format_func=lambda x: "Section-aware" if x == "section" else "Fixed-size",
        help="Section-aware splits text by detected paper sections (Introduction, "
             "Results, etc.) before chunking. Fixed-size ignores structure.",
    )

    retriever_name = st.selectbox(
        "Retriever backend",
        options=["faiss", "numpy"],
        format_func=lambda x: "FAISS" if x == "faiss" else "NumPy (brute-force)",
        help="Both return mathematically identical results at this corpus size -- "
             "this toggle exists to demonstrate they're interchangeable.",
    )

    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("📄 Add your own PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF to search alongside the corpus (this session only)",
        type=["pdf"],
    )

    if uploaded_file is not None:
        # Only reprocess if this is a new file (avoid re-running on every rerun)
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            with st.spinner(f"Processing '{uploaded_file.name}' through the pipeline..."):
                # --- Extract (reusing extract.py's column-aware logic) ---
                pages = extract_raw_pages(uploaded_file)
                repeated_lines = find_repeated_lines(pages)
                cleaned_text = clean_pages(pages, repeated_lines)

                # --- Chunk (reusing chunk.py, same strategy currently selected) ---
                paper_id = "UPLOADED_" + re.sub(r"[^a-zA-Z0-9]+", "_", uploaded_file.name.rsplit(".pdf", 1)[0])[:50]
                if chunking == "section":
                    new_chunks = chunk_by_section(paper_id, cleaned_text)
                else:
                    new_chunks = chunk_fixed(paper_id, cleaned_text)

                if not new_chunks:
                    st.error("Could not extract usable text from this PDF (it may be scanned/image-based).")
                else:
                    # --- Embed using the same model already loaded by the active retriever ---
                    retriever_instance = _get_retriever(retriever_name, chunking)
                    texts = [c["text"] for c in new_chunks]
                    embeddings = retriever_instance.model.encode(
                        texts, convert_to_numpy=True, normalize_embeddings=True
                    )

                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["uploaded_chunks"] = new_chunks
                    st.session_state["uploaded_embeddings"] = embeddings

                    st.success(f"Processed: {len(new_chunks)} chunks added from '{uploaded_file.name}'")

    if st.session_state.get("uploaded_filename"):
        st.info(f"Active upload: **{st.session_state['uploaded_filename']}** "
                f"({len(st.session_state['uploaded_chunks'])} chunks)")

        search_scope = st.radio(
            "Search scope",
            options=["uploaded_only", "both", "corpus_only"],
            format_func=lambda x: {
                "uploaded_only": "📄 Uploaded document only",
                "both": "🔀 Both (blended by score)",
                "corpus_only": "📚 Corpus only (ignore upload)",
            }[x],
            index=0,  # default: uploaded document only -- safest for "explain this paper"-style questions
            help="'Uploaded document only' is safest when asking about 'this paper' -- "
                 "blending can pull in unrelated corpus chunks that outscore the upload "
                 "on generic questions, causing answers about the wrong document.",
        )

        if st.button("Clear uploaded document"):
            for key in ["uploaded_filename", "uploaded_chunks", "uploaded_embeddings"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        search_scope = "corpus_only"

    st.divider()
    st.caption(
        "**Pipeline:** PDF extraction (column-aware) → chunking → "
        "bge-small-en-v1.5 embeddings → cosine similarity retrieval → "
        "Groq (Llama 3.3 70B) generation"
    )

    st.divider()
    st.caption("Eval results (19 ground-truth questions, corpus only):")
    st.metric("Retrieval accuracy", "100.0%")
    st.metric("Answer accuracy", "89.5%")


def retrieve_combined(question, retriever_instance, k, search_scope):
    """
    Retrieve top-k chunks according to the chosen search scope:
      - "uploaded_only": search ONLY the uploaded document's chunks
      - "corpus_only":   search ONLY the persistent corpus (default when no upload)
      - "both":          search both and merge by score

    Scope matters: for questions like "explain this paper", blending is risky --
    generic questions can cause unrelated corpus chunks to outscore the actual
    uploaded document, producing an answer about the wrong paper entirely.
    """
    uploaded_chunks = st.session_state.get("uploaded_chunks")
    uploaded_embeddings = st.session_state.get("uploaded_embeddings")
    has_upload = uploaded_chunks is not None and uploaded_embeddings is not None

    def search_uploaded():
        query_vec = np.asarray(retriever_instance._embed_query(question)).reshape(-1)
        sims = uploaded_embeddings @ query_vec
        top_idx = np.argsort(-sims)[:k]
        return [
            {
                "score": float(sims[i]),
                "id": uploaded_chunks[i]["id"],
                "paper_id": uploaded_chunks[i]["paper_id"],
                "section": uploaded_chunks[i]["section"],
                "text": uploaded_chunks[i]["text"],
            }
            for i in top_idx
        ]

    if search_scope == "uploaded_only" and has_upload:
        return search_uploaded()

    if search_scope == "corpus_only" or not has_upload:
        return retriever_instance.retrieve(question, k=k)

    # "both"
    base_results = retriever_instance.retrieve(question, k=k)
    uploaded_results = search_uploaded()
    combined = base_results + uploaded_results
    combined.sort(key=lambda r: -r["score"])
    return combined[:k]


# ---- Main: question input ----
question = st.text_input(
    "Ask a question about the paper corpus (or your uploaded PDF)",
    placeholder="e.g. What is retrieval augmented generation?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        start = time.perf_counter()
        try:
            retriever_instance = _get_retriever(retriever_name, chunking)
            retrieved_chunks = retrieve_combined(question, retriever_instance, k, search_scope)
            result = generate_answer(question, retrieved_chunks)
            result["retrieved_chunks"] = retrieved_chunks
            elapsed = time.perf_counter() - start
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success(f"Answered in {elapsed:.2f}s")

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader(f"Retrieved sources (top {k})")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        is_uploaded = chunk["paper_id"].startswith("UPLOADED_")
        badge = "🆕 UPLOADED — " if is_uploaded else ""
        with st.expander(
            f"[{i}] {badge}{chunk['paper_id'][:60]}  —  {chunk['section']}  "
            f"(score: {chunk['score']:.3f})"
        ):
            st.write(chunk["text"])

elif submitted:
    st.warning("Please enter a question first.")