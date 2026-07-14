"""Hybrid retrieval: dense (FAISS) + BM25 + Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunker import Chunk
from app.config import settings
from app.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "scores": self.scores,
        }


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked ID lists with Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """FAISS dense index + BM25 keyword index with RRF merge."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store_dir: Path | None = None,
        index_name: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()
        self.store_dir = Path(store_dir or settings.vector_store_dir)
        self.index_name = index_name or settings.index_name
        self.chunks: list[Chunk] = []
        self._id_to_chunk: dict[str, Chunk] = {}
        self._faiss_index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self._faiss_index is not None and self._bm25 is not None

    def build(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self.chunks = list(chunks)
        self._id_to_chunk = {c.chunk_id: c for c in self.chunks}
        texts = [c.content for c in self.chunks]

        logger.info("Embedding %d chunks for FAISS", len(texts))
        vectors = self.embedding_model.embed_documents(texts)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._faiss_index = index

        logger.info("Building BM25 index")
        self._tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("Hybrid index ready (%d chunks, dim=%d)", len(self.chunks), dim)

    def dense_search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not built")
        top_k = top_k or settings.dense_top_k
        top_k = min(top_k, len(self.chunks))
        q = self.embedding_model.embed_query(query).reshape(1, -1)
        scores, indices = self._faiss_index.search(q, top_k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                ScoredChunk(chunk=chunk, score=float(score), scores={"dense": float(score)})
            )
        return results

    def bm25_search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built")
        top_k = top_k or settings.bm25_top_k
        top_k = min(top_k, len(self.chunks))
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[ScoredChunk] = []
        for idx in top_indices:
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                ScoredChunk(chunk=chunk, score=score, scores={"bm25": score})
            )
        return results

    def hybrid_search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[ScoredChunk]:
        dense = self.dense_search(query, top_k=dense_top_k)
        sparse = self.bm25_search(query, top_k=bm25_top_k)

        dense_ids = [s.chunk.chunk_id for s in dense]
        sparse_ids = [s.chunk.chunk_id for s in sparse]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=rrf_k or settings.rrf_k)

        dense_map = {s.chunk.chunk_id: s for s in dense}
        sparse_map = {s.chunk.chunk_id: s for s in sparse}

        limit = top_k or settings.hybrid_top_k
        results: list[ScoredChunk] = []
        for chunk_id, rrf_score in fused[:limit]:
            chunk = self._id_to_chunk[chunk_id]
            scores: dict[str, float] = {"rrf": float(rrf_score)}
            if chunk_id in dense_map:
                scores["dense"] = dense_map[chunk_id].scores.get("dense", dense_map[chunk_id].score)
            if chunk_id in sparse_map:
                scores["bm25"] = sparse_map[chunk_id].scores.get("bm25", sparse_map[chunk_id].score)
            results.append(ScoredChunk(chunk=chunk, score=float(rrf_score), scores=scores))
        return results

    def save(self) -> Path:
        if not self.is_ready:
            raise RuntimeError("Nothing to save — build the index first")

        self.store_dir.mkdir(parents=True, exist_ok=True)
        base = self.store_dir / self.index_name

        faiss.write_index(self._faiss_index, str(base.with_suffix(".faiss")))
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "tokenized_corpus": self._tokenized_corpus,
            "embedding_model": self.embedding_model.model_name,
        }
        with open(base.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        with open(base.with_suffix(".bm25.pkl"), "wb") as f:
            pickle.dump(self._bm25, f)

        logger.info("Saved index to %s", base)
        return base

    def load(self) -> None:
        base = self.store_dir / self.index_name
        faiss_path = base.with_suffix(".faiss")
        meta_path = base.with_suffix(".meta.json")
        bm25_path = base.with_suffix(".bm25.pkl")

        if not faiss_path.exists() or not meta_path.exists() or not bm25_path.exists():
            raise FileNotFoundError(f"Index files not found under {self.store_dir / self.index_name}")

        self._faiss_index = faiss.read_index(str(faiss_path))
        with open(meta_path, encoding="utf-8") as f:
            payload = json.load(f)
        self.chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        self._id_to_chunk = {c.chunk_id: c for c in self.chunks}
        self._tokenized_corpus = payload.get("tokenized_corpus", [_tokenize(c.content) for c in self.chunks])
        with open(bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)  # noqa: S301 — local trusted artifact
        logger.info("Loaded index with %d chunks", len(self.chunks))
