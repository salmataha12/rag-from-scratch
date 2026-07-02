"""
Streamlit UI for the RAG-from-scratch system. Lets you ask questions against
the paper corpus and toggle chunking strategy / retriever backend live to
see how they affect results -- a nice way to demo the comparison work.

Usage:
    streamlit run app.py
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import ask

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

    retriever = st.selectbox(
        "Retriever backend",
        options=["faiss", "numpy"],
        format_func=lambda x: "FAISS" if x == "faiss" else "NumPy (brute-force)",
        help="Both return mathematically identical results at this corpus size -- "
             "this toggle exists to demonstrate they're interchangeable.",
    )

    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.caption(
        "**Pipeline:** PDF extraction (column-aware) → chunking → "
        "bge-small-en-v1.5 embeddings → cosine similarity retrieval → "
        "Groq (Llama 3.3 70B) generation"
    )

    st.divider()
    st.caption("Eval results (19 ground-truth questions):")
    st.metric("Retrieval accuracy", "100.0%")
    st.metric("Answer accuracy", "89.5%")


# ---- Main: question input ----
question = st.text_input(
    "Ask a question about the paper corpus",
    placeholder="e.g. What is retrieval augmented generation?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        start = time.perf_counter()
        try:
            result = ask(question, retriever=retriever, chunking=chunking, k=k)
            elapsed = time.perf_counter() - start
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success(f"Answered in {elapsed:.2f}s")

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader(f"Retrieved sources (top {k})")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        with st.expander(
            f"[{i}] {chunk['paper_id'][:60]}  —  {chunk['section']}  "
            f"(score: {chunk['score']:.3f})"
        ):
            st.write(chunk["text"])

elif submitted:
    st.warning("Please enter a question first.")