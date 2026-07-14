# Hybrid RAG

Production-oriented **Retrieval-Augmented Generation** chatbot that answers questions from your own documents using hybrid search, cross-encoder reranking, grounded generation, and citation verification.

## Architecture

```
User Question
    → Dense Vector Search (FAISS + BGE)
    → BM25 Keyword Search
    → Reciprocal Rank Fusion (RRF)
    → Cross-Encoder Reranker (bge-reranker-large)
    → LLM Generator (Groq)
    → Citation Verification (NLI / similarity / LLM judge)
    → Verified Answer
```

## Features

| Step | Component | Implementation |
|------|-----------|----------------|
| Load | PDF / DOCX / TXT / HTML | `pypdf`, `python-docx`, BeautifulSoup |
| Chunk | Overlapping text splits | LangChain `RecursiveCharacterTextSplitter` (500 / 100) |
| Embed | Dense vectors | `BAAI/bge-large-en-v1.5` |
| Store | Local vector DB | FAISS |
| Keyword | Sparse retrieval | `rank-bm25` |
| Merge | Rank fusion | Reciprocal Rank Fusion |
| Rerank | Cross-encoder | `BAAI/bge-reranker-large` |
| Generate | Grounded LLM | Groq (`llama-3.3-70b-versatile`) |
| Verify | Citation checks | DeBERTa NLI (default) |

## Project layout

```text
Hybrid-RAG/
├── app/
│   ├── api.py              # FastAPI routes
│   ├── rag_pipeline.py     # End-to-end orchestration
│   ├── retriever.py        # FAISS + BM25 + RRF
│   ├── reranker.py         # Cross-encoder reranker
│   ├── verifier.py         # Citation verification
│   ├── generator.py        # Groq answer generation
│   ├── embeddings.py       # Sentence-transformer embeddings
│   ├── chunker.py          # Text splitting
│   ├── loaders.py          # Document loaders
│   └── config.py           # Settings from .env
├── data/
│   ├── raw_docs/           # Source documents
│   └── vector_store/       # Persisted FAISS + BM25 indexes
├── evaluation/
│   ├── ragas_eval.py
│   └── test_queries.json
├── frontend/
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` and set your [Groq](https://console.groq.com/) API key:

```env
GROQ_API_KEY=gsk_...
```

### 3. Ingest sample documents

A sample employee handbook is already in `data/raw_docs/`.

```bash
python -c "from app.rag_pipeline import RAGPipeline; print(RAGPipeline(auto_load=False).ingest_directory())"
```

First run downloads embedding / reranker / NLI models from Hugging Face (several GB). Prefer a machine with enough disk and, optionally, a GPU (`EMBEDDING_DEVICE=cuda` in `.env`).

### 4. Start the API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the UI (separate terminal)

```bash
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501, ask:

> How many annual leave days can employees carry forward?

Expected grounded answer: **10** days, with a verified citation to the handbook.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Index readiness + chunk count |
| `POST` | `/ingest` | Upload files (`multipart/form-data`, field `files`) |
| `POST` | `/ingest/directory` | Index everything under `data/raw_docs/` |
| `POST` | `/query` | Ask a question |

### Query example

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"How many annual leave days can employees carry forward?\"}"
```

Response includes the answer, verified citations, retrieved/reranked passages, and pipeline metadata.

## Citation verification modes

Set `VERIFIER_MODE` in `.env`:

| Mode | Behavior |
|------|----------|
| `nli` (default) | DeBERTa MNLI entailment check |
| `similarity` | Embedding cosine similarity threshold |
| `llm` | Groq LLM-as-judge (`YES` / `NO` / `UNSURE`) |

Unsupported citations are dropped. If none survive, the system returns:

`I don't know based on the available documents.`

## Evaluation

```bash
python evaluation/ragas_eval.py --ingest
```

Writes `evaluation/eval_results.json` with a simple accuracy score over curated queries. Optional Ragas metrics can be enabled by installing `ragas` / `datasets` (see comments in `requirements.txt`).

## Docker

```bash
docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:8501  

Models are cached in a Docker volume (`hf_cache`). Provide `.env` with `GROQ_API_KEY` before starting.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROQ_API_KEY` | _(required for answers)_ | Groq API key |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Dense encoder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `VERIFIER_MODE` | `nli` | `nli` / `similarity` / `llm` |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `CHUNK_SIZE` | `500` | Chunk character size |
| `CHUNK_OVERLAP` | `100` | Overlap characters |
| `RERANK_TOP_K` | `5` | Passages sent to the LLM |

## Notes

- **Hybrid search** combines semantic recall (dense) with exact keyword matching (BM25).
- **Reranking** is the quality jump most demos skip — the cross-encoder scores query+passage jointly.
- **Verification** is the production layer: answers without supported evidence are refused.
- First-time model downloads are large; subsequent runs load from cache.
- For lighter local experiments you can switch to smaller models (e.g. `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-base`) via `.env`.

## License

MIT
