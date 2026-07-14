"""End-to-end Hybrid RAG pipeline orchestration."""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from app.chunker import chunk_documents
from app.config import settings
from app.generator import GenerationResult, GroqGenerator
from app.loaders import Document, load_directory
from app.reranker import CrossEncoderReranker, get_reranker
from app.retriever import HybridRetriever, ScoredChunk
from app.verifier import CitationVerifier, get_verifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    reranked: list[dict] = field(default_factory=list)
    verified: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """Ingest → hybrid retrieve → rerank → generate → verify."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        generator: GroqGenerator | None = None,
        verifier: CitationVerifier | None = None,
        auto_load: bool = True,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or get_reranker()
        self.generator = generator or GroqGenerator()
        self.verifier = verifier or get_verifier()
        if auto_load:
            try:
                self.retriever.load()
                logger.info("Loaded existing vector store")
            except FileNotFoundError:
                logger.info("No existing index found; ingest documents before querying")

    @property
    def is_ready(self) -> bool:
        return self.retriever.is_ready

    def ingest_directory(self, directory: Path | str | None = None, persist: bool = True) -> dict:
        directory = Path(directory or settings.raw_docs_dir)
        documents = load_directory(directory)
        return self.ingest_documents(documents, persist=persist)

    def ingest_files(self, paths: Sequence[Path | str], persist: bool = True) -> dict:
        """Save uploads into data/raw_docs, then rebuild the index from that folder.

        Rebuilding from the whole folder avoids answering from a stale sample doc
        when the user expects answers from newly uploaded files (and keeps older
        docs available alongside the new ones).
        """
        settings.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            src = Path(path)
            dest = settings.raw_docs_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        return self.ingest_directory(settings.raw_docs_dir, persist=persist)

    def ingest_documents(self, documents: Sequence[Document], persist: bool = True) -> dict:
        if not documents:
            raise ValueError("No documents to ingest")
        chunks = chunk_documents(documents)
        self.retriever.build(chunks)
        if persist:
            self.retriever.save()
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "sources": sorted({d.source for d in documents}),
        }

    def retrieve(self, question: str) -> list[ScoredChunk]:
        if not self.retriever.is_ready:
            raise RuntimeError("Index is empty. Ingest documents first.")
        return self.retriever.hybrid_search(question)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        skip_verification: bool = False,
    ) -> PipelineResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        retrieved = self.retrieve(question)
        reranked = self.reranker.rerank(question, retrieved, top_k=top_k or settings.rerank_top_k)
        generation: GenerationResult = self.generator.generate(question, reranked)

        if skip_verification:
            verified_result = generation
            verified_flag = False
        else:
            verified_result = self.verifier.verify(generation)
            verified_flag = bool(verified_result.citations) and all(
                c.verified for c in verified_result.citations
            )

        citations = [
            {
                "marker": c.marker,
                "claim": c.claim,
                "source": c.source,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "passage": c.passage,
                "verified": c.verified,
                "verification_score": c.verification_score,
                "verification_label": c.verification_label,
            }
            for c in verified_result.citations
        ]

        return PipelineResult(
            question=question,
            answer=verified_result.answer,
            citations=citations,
            retrieved=[s.to_dict() for s in retrieved],
            reranked=[s.to_dict() for s in reranked],
            verified=verified_flag,
            meta={
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "llm_model": settings.llm_model,
                "verifier_mode": settings.verifier_mode,
                "num_retrieved": len(retrieved),
                "num_reranked": len(reranked),
            },
        )


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
