# RAG From Scratch

A retrieval-augmented generation (RAG) system built from the ground up — manual PDF extraction, chunking, embeddings, and cosine similarity retrieval — with no LangChain, no vector database abstraction, and no framework doing the retrieval math for you. Built over a corpus of 59 arXiv papers spanning RAG, dense retrieval, LLMs, LoRA fine-tuning, and multi-agent systems.

The goal of this project wasn't to build "a RAG chatbot" — it was to understand and implement every component of a retrieval pipeline by hand first, then layer in production-style tooling (FAISS, evaluation, an interactive UI) on top of that foundation, and measure the result rather than just demo it.

## Live demo

Run `streamlit run app.py` to try it — includes live PDF upload, so you can ask questions against a document that was never part of the original corpus (see [Streamlit App](#streamlit-app) below).

---

## Architecture

```
PDF corpus (59 arXiv papers)
        │
        ▼
┌─────────────────┐
│   extract.py     │  Column-aware, gap-based text extraction
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    chunk.py      │  Two strategies: fixed-size vs section-aware
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    embed.py      │  BAAI/bge-small-en-v1.5, normalized vectors
└─────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  retrieve_numpy.py / retrieve_faiss.py │  Manual cosine similarity vs FAISS IndexFlatIP
└─────────────────────────────┘
        │
        ▼
┌─────────────────┐
│   generate.py     │  Groq (Llama 3.3 70B), grounded + cited answers
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    app.py         │  Streamlit UI, live PDF upload, scope control
└─────────────────┘
```

`pipeline.py` ties retrieval + generation into a single `ask()` function that every other script (`eval/run_eval.py`, `experiments/*`, `app.py`) calls into — one entry point, one place to change behavior.

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| PDF extraction | `pdfplumber` (word-level, custom logic) | Needed manual control over spacing and column ordering — see [Design Decisions](#design-decisions) |
| Embeddings | `sentence-transformers` — `BAAI/bge-small-en-v1.5` | Better retrieval accuracy than `all-MiniLM-L6-v2` at similar size/speed |
| Retrieval (v1) | NumPy — manual cosine similarity | Proves understanding of the underlying math |
| Retrieval (v2) | FAISS — `IndexFlatIP` | Industry-standard, benchmarked against the manual version |
| Generation | Groq API — `llama-3.3-70b-versatile` | Fast, free-tier friendly |
| UI | Streamlit | Fast to build, good for demoing strategy toggles live |
| Eval | Custom scoring script + LLM-assisted ground truth | See [Evaluation](#evaluation) |

No LangChain, no LlamaIndex, no vector database — every component up through retrieval is implemented directly.

---

## Repo structure

```
rag-from-scratch/
├── data/
│   ├── papers/          # raw PDFs (59 arXiv papers)
│   └── extracted/        # cleaned .txt per paper
├── src/
│   ├── extract.py          # PDF -> cleaned text (column-aware)
│   ├── chunk.py              # fixed-size AND section-aware chunking
│   ├── embed.py                # chunks -> vectors (bge-small)
│   ├── retrieve_numpy.py         # manual cosine similarity
│   ├── retrieve_faiss.py          # FAISS IndexFlatIP
│   ├── generate.py                  # prompt + Groq call
│   └── pipeline.py                    # unified ask() entry point
├── eval/
│   ├── questions.json                  # 19 ground-truth Q&A pairs
│   ├── run_eval.py                      # scores retrieval + answer accuracy
│   ├── generate_eval_questions.py        # LLM-assisted question generation
│   └── results_*.json                     # saved eval runs
├── experiments/
│   ├── compare_existing_results.py         # chunking strategy comparison
│   └── retrieval_speed.py                   # NumPy vs FAISS benchmark
├── store/                                     # embeddings, chunk JSON (generated, gitignored)
├── app.py                                       # Streamlit UI
├── download_papers.py                            # arXiv corpus downloader
├── requirements.txt
└── README.md
```

---

## Design decisions

**Chunk size: 350 words, 50-word overlap.** Chosen to keep chunks small enough for precise retrieval while still containing enough context for the LLM to generate a coherent, grounded answer.

**Cosine similarity for retrieval.** Embeddings are normalized to unit length at encode time (`normalize_embeddings=True`), which means cosine similarity reduces to a plain dot product — simpler and faster to compute manually, with no accuracy tradeoff.

**Top-k = 5.** Balances providing enough context for synthesis against diluting the prompt with irrelevant chunks.

**Embedding model: `BAAI/bge-small-en-v1.5` over `all-MiniLM-L6-v2`.** MiniLM is the more commonly used default in tutorials, but BGE models are trained with a contrastive objective specifically tuned for retrieval and score meaningfully better on retrieval-focused benchmarks (MTEB), at a similar size/speed footprint. BGE also requires an instruction prefix on the *query* side only (`"Represent this sentence for searching relevant passages: "`) — documents are embedded without it. This asymmetry is implemented in `retrieve_numpy.py`/`retrieve_faiss.py` and is easy to miss if copying boilerplate embedding code without reading the model card.

**Two chunking strategies, compared empirically rather than assumed.** See [Experiments](#experiments) below — this isn't a hypothetical comparison, it's measured against the eval set.

**Column-aware PDF extraction.** Most CS/AI papers use a two-column layout. A naive top-to-bottom text extraction reads across both columns at once, scrambling sentences (e.g. `"WepresentAceWiki"` merging with unrelated text from the other column at the same page height). The extractor detects the blank vertical "gutter" between columns per page and reads the left column fully before the right, which fixed the vast majority of scrambling — though it isn't perfect on every page layout (see [Limitations](#limitations)).

**Minimum chunk length filter (15 words).** Added after a live demo surfaced a near-empty one-word chunk ("RAG") scoring deceptively high on a generic query, crowding out more informative results. Short chunks carry little content but can still score well on generic queries purely due to embedding-space quirks with short text.

---

## Experiments

### 1. Chunking strategy: fixed-size vs section-aware

| Strategy | Retrieval Accuracy | Answer Accuracy |
|---|---|---|
| Fixed-size | 100.0% | 84.2% |
| Section-aware | 100.0% | 89.5% |

**Section-aware chunking: +5.3% answer accuracy, no change in retrieval accuracy**, evaluated with FAISS at k=5 on the 19-question ground-truth set.

Interpretation: retrieval accuracy tied because the eval questions are specific enough that both strategies surface the correct *paper* regardless of exact chunk boundaries. The answer-accuracy gap is the more informative signal — it suggests section-aware chunks are cleaner and more self-contained (not accidentally spanning two unrelated sections), giving the LLM better-quality context to generate from even when raw retrieval performs identically. With only 19 questions, a single question flip changes the score by ~5%, so this should be read as a consistent, real signal at modest scale — not an exaggerated claim of section-aware chunking being dramatically superior.

### 2. Retrieval speed: NumPy vs FAISS

| Backend | Avg latency | Min | Max |
|---|---|---|---|
| NumPy (brute-force) | 20.60 ms | 14.60 ms | 49.14 ms |
| FAISS (`IndexFlatIP`) | 20.10 ms | 15.22 ms | 31.25 ms |

**FAISS is ~1.02x faster than NumPy** — essentially a tie at this corpus size (2,599 chunks). Both are dominated by the same underlying matrix operations at this scale; FAISS's real advantage is architectural, not raw speed here. It supports approximate indexes (e.g. `IndexHNSWFlat`) that scale sub-linearly with corpus size, which would matter significantly at 100k+ vectors — not the bottleneck for a corpus this size. Verified correctness first: both retrievers return mathematically identical top-5 results and scores for the same query, confirming FAISS is a correct drop-in replacement, not just a different implementation with different behavior.

---

## Evaluation

### Methodology

19 ground-truth question/answer pairs across 8 papers (Ragas, WizardLM, KD-LoRA, Dense Passage Retrieval, Learning From Failure, Retrieval-Oriented Masking, ARCANE, LoRA-C), each tagged with an `expected_paper` and `expected_keywords`.

**Ground truth generation was LLM-assisted, not fully manual.** Given time constraints, questions were generated by prompting Groq with each paper's abstract and asking for factual Q&A pairs grounded only in that text (`eval/generate_eval_questions.py`), then spot-checked rather than verified against full paper reads. Two specific factual claims (the RAGAS acronym expansion, and an author's institutional affiliation) were independently verified via web search during review. Weak or overly vague auto-generated questions (e.g. tautological ones that just restated the paper title) were manually removed.

An earlier version of the question set generated questions in isolation from paper context (e.g. *"What is the name of the proposed method?"* with no anchor), which scored only 47.6% retrieval accuracy — not because retrieval was broken, but because the questions themselves were ambiguous across a 59-paper corpus where many papers "propose a method." Adding paper-anchoring phrases (e.g. *"In the KD-LoRA paper, what..."*) — which also better reflects how people actually phrase real queries — brought retrieval accuracy to 100%. This is documented here rather than hidden because it's a genuinely useful finding about eval design, not just a result to report.

### Scoring

- **Retrieval accuracy**: did the expected paper appear anywhere in the top-k retrieved chunks?
- **Answer accuracy**: did the generated answer contain at least one expected keyword (case-insensitive, with special handling so a keyword like `"99%"` also matches a more precise answer like `"99.35%"`)?

### Results (FAISS, section-aware chunking, k=5)

| Metric | Score |
|---|---|
| Retrieval accuracy | **19/19 (100.0%)** |
| Answer accuracy | **17/19 (89.5%)** |

**Manual review of the 2 remaining answer failures:**
- One was a genuine retrieval-granularity limitation: the correct paper was retrieved, but the specific fact needed lived in a different section than what was in the top-5 chunks.
- The other was a scoring-strictness case, not a wrong answer: the model gave a substantively correct summary that didn't happen to contain the exact expected keyword phrase — a known tradeoff of automated keyword-matching evaluation versus more expensive semantic-similarity scoring.

---

## Streamlit App

Run with `streamlit run app.py`. Features:

- **Live strategy toggles** — switch chunking strategy (fixed/section) and retriever backend (NumPy/FAISS) mid-session and re-ask the same question to see the effect directly, rather than trusting the numbers in this README on faith.
- **PDF upload** — upload any PDF and it's processed live through the *same* `extract → chunk → embed` functions used to build the corpus (not a separate code path), proving the pipeline is genuinely general-purpose and not hardcoded to the 59 demo papers. Processed entirely in-memory for the session; nothing is written to disk or added to the persistent corpus/eval set.
- **Search scope control** — when a document is uploaded, defaults to searching *only* that document rather than blending with the corpus. This was a deliberate fix after testing surfaced a real failure mode: a generic question like *"explain the methodology in this paper"* caused corpus chunks to outscore the actual uploaded document on pure embedding similarity, producing a fluent but wrong answer about an unrelated paper. Blending remains available as an explicit opt-in for cases where comparing across corpus and upload is actually desired.

---

## Limitations

- **Two-column extraction isn't perfect on every page.** The gutter-detection heuristic fixed the large majority of column-scrambling issues, but pages with unusual layouts (e.g. dense tables, prompt-injection example text with irregular spacing) can still produce minor spacing artifacts. Not corrected further, on the judgment that pursuing a fully general PDF layout parser was a disproportionate time investment relative to its marginal benefit for this project's goals.
- **Section header detection is heuristic, not exhaustive.** Combines a keyword list (Introduction, Results, Conclusion, etc.) with a general numbered Title-Case pattern (catching things like "3.2 Agent Framework" without a known keyword). A small fraction of papers with unconventional heading styles fall back to single-section chunking, equivalent to the fixed-size baseline for that paper specifically.
- **Eval set is small (19 questions, 8 papers) and partially LLM-assisted.** A larger, fully manually-verified benchmark would be more rigorous; this was a deliberate scope tradeoff given time constraints, documented rather than hidden.
- **Keyword-matching evaluation has known false negatives.** Semantically correct answers phrased differently than the expected keyword register as failures. Estimated true answer accuracy after manual review is closer to ~95%, not 89.5% — both numbers are reported for transparency.
- **FAISS shows negligible speedup at this corpus scale (2,599 chunks).** Its value here is architectural and forward-looking (supports approximate indexing at much larger scale), not a demonstrated performance win at the current size.

## Future work

- Cross-encoder reranking on top of initial retrieval (retrieve top-20, rerank to top-5)
- Embedding model comparison (bge-small vs e5-base vs MiniLM) with the same eval methodology
- Rebuild the retrieval/generation layer with LangChain and LangGraph, to compare framework abstraction against the from-scratch implementation directly
- Expand the eval set with fully manually-verified questions across more of the 59-paper corpus

---

## Setup

```powershell
git clone <repo-url>
cd rag-from-scratch
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

```powershell
python download_papers.py       # downloads the 59-paper corpus from arXiv
python src/extract.py           # PDF -> cleaned text
python src/chunk.py             # cleaned text -> chunks (both strategies)
python src/embed.py             # chunks -> embeddings (both strategies)
python eval/run_eval.py         # run the eval suite
streamlit run app.py            # launch the interactive UI
```

## Data source

Papers sourced from [arXiv](https://arxiv.org/) via the `arxiv` Python package, spanning categories cs.CL and cs.AI, across search queries for retrieval-augmented generation, dense retrieval, sentence embeddings, LLM surveys, LoRA fine-tuning, and multi-agent LLM systems.
