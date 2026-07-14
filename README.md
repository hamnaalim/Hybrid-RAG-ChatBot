# Hybrid RAG

Production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from **your own documents** — not from the model’s general knowledge alone.

It uses hybrid search (dense vectors + BM25), cross-encoder reranking, Groq for grounded generation, and citation verification before returning an answer.

---

## About

Upload PDFs, DOCX, TXT, or HTML, then ask questions like:

> How many annual leave days can employees carry forward?

The system:

1. Searches your documents (semantic + keyword)
2. Merges results with Reciprocal Rank Fusion (RRF)
3. Reranks the best passages
4. Sends only those passages to the LLM
5. Verifies citations against the evidence
6. Returns a grounded answer (or “I don’t know” if unsupported)

---

## Architecture

```text
User Question
      │
      ▼
Query Processing
      │
      ├──────────────────┐
      ▼                  ▼
Dense Vector Search   BM25 Search
(FAISS + BGE)         (keywords)
      │                  │
      └────────┬─────────┘
               ▼
        Merge (RRF)
               ▼
   Cross-Encoder Reranker
               ▼
      Top Relevant Chunks
               ▼
         LLM (Groq)
               ▼
    Citation Verification
               ▼
     Final Verified Answer
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Chunking | LangChain text splitters |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| Vector DB | FAISS (local) |
| Keyword search | `rank-bm25` |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `BAAI/bge-reranker-large` |
| LLM | Groq |
| Citation verification | DeBERTa NLI (default) |
| Deployment | Docker |

---

## Project structure

```text
RAG/
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
│   └── config.py           # Settings from environment
├── data/
│   ├── raw_docs/           # Source documents
│   └── vector_store/       # Persisted FAISS + BM25 indexes
├── evaluation/
│   ├── ragas_eval.py
│   └── test_queries.json
├── frontend/
│   └── streamlit_app.py
├── .env.example            # Safe template (no secrets)
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Security: keep your Groq API key private

**Never commit your real API key to GitHub.**

| File | Commit? | Purpose |
|------|---------|---------|
| `.env` | **No** (gitignored) | Your real `GROQ_API_KEY` — stays on your machine |
| `.env.example` | **Yes** | Template with placeholder only |

Rules:

1. Put the key only in a local `.env` file (not in code, README, or screenshots).
2. `.gitignore` already ignores `.env`.
3. Share `.env.example` only — never a copy of `.env`.
4. If you ever pasted a key into a commit or chat, **rotate/revoke it** on [Groq Console](https://console.groq.com/) and create a new one.

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd RAG

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets locally

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and set your key (get one from [Groq Console](https://console.groq.com/)):

```env
GROQ_API_KEY=gsk_your_key_here
```

Do **not** commit `.env`.

### 3. Ingest sample documents (optional)

A sample handbook may already be under `data/raw_docs/`.

```bash
python -c "from app.rag_pipeline import RAGPipeline; print(RAGPipeline(auto_load=False).ingest_directory())"
```

First run downloads embedding / reranker / NLI models from Hugging Face (large download).

### 4. Run the API

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the UI (second terminal)

```bash
streamlit run frontend/streamlit_app.py
```

- UI: http://localhost:8501  
- API health: http://127.0.0.1:8000/health  
- API docs: http://127.0.0.1:8000/docs  

Upload your own documents in the Streamlit sidebar → **Upload & index**, then ask questions.

You need **both** processes running (API on `8000` and Streamlit on `8501`).

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Index readiness + chunk count |
| `POST` | `/ingest` | Upload files (`multipart`, field `files`) |
| `POST` | `/ingest/directory` | Index everything under `data/raw_docs/` |
| `POST` | `/query` | Ask a question |

Example:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"How many annual leave days can employees carry forward?\"}"
```

---

## Configuration

Copy from `.env.example`. Common variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROQ_API_KEY` | _(required)_ | Groq API key — **local `.env` only** |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Dense encoder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `VERIFIER_MODE` | `nli` | `nli` / `similarity` / `llm` |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `CHUNK_SIZE` | `500` | Chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap |
| `RERANK_TOP_K` | `5` | Chunks sent to the LLM |

Lighter CPU options (in `.env`):

```env
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

---

## Citation verification

| Mode | Behavior |
|------|----------|
| `nli` (default) | DeBERTa MNLI entailment check |
| `similarity` | Embedding similarity threshold |
| `llm` | Groq LLM-as-judge |

Unsupported citations are dropped. If none survive, the system replies that it doesn’t know based on the available documents.

---

## Evaluation

```bash
python evaluation/ragas_eval.py --ingest
```

Writes `evaluation/eval_results.json`.

---

## Docker

```bash
# Create local .env first (never commit it)
copy .env.example .env

docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:8501  

---

## Topics

`rag` · `fastapi` · `streamlit` · `faiss` · `bm25` · `hybrid-search` · `reranker` · `groq` · `python`

---

## License

MIT
