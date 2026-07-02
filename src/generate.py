"""
Takes a question + retrieved chunks, builds a grounded prompt, and calls
Groq to generate an answer that cites which paper/section it came from.

Usage (as a module):
    from src.generate import generate_answer
    answer = generate_answer(question, retrieved_chunks)

Usage (standalone test):
    python src/generate.py
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from .env

GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions strictly using the "
    "provided context from academic papers. Follow these rules:\n"
    "1. Only use information found in the context below. Do not use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say so clearly "
    "instead of guessing.\n"
    "3. For every claim, cite the source in the format [paper_id, section].\n"
    "4. Be concise and precise -- prefer short, well-cited answers over long ones."
)


def _format_context(chunks: list) -> str:
    """
    Turn retrieved chunks into a numbered context block the model can cite from,
    e.g. "[1] (paper_id, section): <chunk text>"
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] ({c['paper_id']}, {c['section']}):\n{c['text']}\n")
    return "\n".join(lines)


def generate_answer(question: str, chunks: list, temperature: float = 0.2) -> dict:
    """
    Generate an answer grounded in the given retrieved chunks.

    Returns a dict: {"answer": str, "sources": list of {paper_id, section}}
    """
    context = _format_context(chunks)

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above, citing sources as "
        f"[paper_id, section]."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    answer_text = response.choices[0].message.content

    sources = [{"paper_id": c["paper_id"], "section": c["section"]} for c in chunks]

    return {
        "answer": answer_text,
        "sources": sources,
    }


if __name__ == "__main__":
    # Quick manual test: retrieve then generate, end to end
    from retrieve_faiss import FaissRetriever

    retriever = FaissRetriever("store/embeddings_section.npz")

    test_question = "What is retrieval augmented generation?"
    print(f"Question: {test_question}\n")

    retrieved = retriever.retrieve(test_question, k=5)
    result = generate_answer(test_question, retrieved)

    print("Answer:")
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['paper_id']} | {s['section']}")
