"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw_docs"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    index_name: str = "hybrid_rag"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = 16

    # Retrieval
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    hybrid_top_k: int = 20

    # Reranking
    reranker_model: str = "BAAI/bge-reranker-large"
    rerank_top_k: int = 5

    # Generation (Groq)
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Citation verification
    verifier_mode: Literal["nli", "similarity", "llm"] = "nli"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    similarity_threshold: float = 0.55
    nli_entailment_threshold: float = 0.5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


settings = Settings()
settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
