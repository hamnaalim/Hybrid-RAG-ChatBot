## About

Hybrid RAG is a document Q&A system that answers questions **only from your uploaded files**.

Instead of stuffing full documents into an LLM, it:

1. Loads PDF / DOCX / TXT / HTML
2. Splits text into overlapping chunks
3. Retrieves with **hybrid search** (dense vectors + BM25 keywords)
4. Merges results with **Reciprocal Rank Fusion (RRF)**
5. Reranks with a **cross-encoder**
6. Generates answers with **Groq**
7. **Verifies citations** so weak or unsupported claims are filtered

### Stack
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Embeddings:** BAAI/bge-large-en-v1.5  
- **Vector store:** FAISS  
- **Keyword search:** rank-bm25  
- **Reranker:** BAAI/bge-reranker-large  
- **LLM:** Groq  
- **Verification:** DeBERTa NLI (optional similarity / LLM judge)

### Why it exists
Most demos stop at “embed + chat.” This project shows a clearer production path: hybrid retrieval, reranking, grounded generation, and citation checks.
