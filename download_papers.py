"""
Downloads a batch of arXiv papers on RAG / retrieval / LLM / fine-tuning topics
to use as the test corpus for the RAG-from-scratch project.

"""

import arxiv
import os
import time
import re
import urllib.request

OUTPUT_DIR = os.path.join("data", "papers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Search queries covering the topic mix we agreed on.
# Each query pulls a batch of papers; total should land around 50-70.
QUERIES = [
    ("retrieval augmented generation", 15),
    ("dense passage retrieval embeddings", 10),
    ("sentence transformers semantic search", 8),
    ("large language model survey", 10),
    ("LoRA parameter efficient fine-tuning", 8),
    ("multi-agent large language models", 10),
]


def safe_filename(title: str, arxiv_id: str) -> str:
    """Turn a paper title into a safe filename, prefixed with its arXiv id."""
    clean = re.sub(r"[^a-zA-Z0-9\s-]", "", title)
    clean = re.sub(r"\s+", "_", clean.strip())[:80]
    return f"{arxiv_id}_{clean}.pdf"


def download_papers():
    client = arxiv.Client()
    downloaded = set()
    total = 0

    for query, max_results in QUERIES:
        print(f"\nSearching: '{query}' (up to {max_results} papers)")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            if result.entry_id in downloaded:
                continue  # skip duplicates across overlapping queries

            arxiv_id = result.get_short_id()
            filename = safe_filename(result.title, arxiv_id)
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                print(f"  Already have: {filename}")
                downloaded.add(result.entry_id)
                continue

            try:
                # Download directly from the PDF URL instead of relying on
                # the library's download_pdf() method (API differs across versions).
                urllib.request.urlretrieve(result.pdf_url, filepath)
                print(f"  Downloaded: {filename}")
                downloaded.add(result.entry_id)
                total += 1
                time.sleep(1)  # be polite to arXiv's servers
            except Exception as e:
                print(f"  FAILED: {result.title[:60]} -> {e}")

    print(f"\nDone. Downloaded {total} new papers to '{OUTPUT_DIR}/'.")
    print(f"Total unique papers so far: {len(downloaded)}")


if __name__ == "__main__":
    download_papers()
