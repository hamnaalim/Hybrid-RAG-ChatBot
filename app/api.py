"""FastAPI backend for the Hybrid RAG system."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.rag_pipeline import get_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid RAG API",
    description="Production-oriented RAG with hybrid retrieval, reranking, and citation verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    skip_verification: bool = False


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    sources: list[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    verified: bool
    meta: dict[str, Any]


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "index_ready": pipeline.is_ready,
        "chunk_count": len(pipeline.retriever.chunks) if pipeline.is_ready else 0,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    pipeline = get_pipeline()
    saved: list[Path] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for upload in files:
                if not upload.filename:
                    continue
                dest = tmp_dir / Path(upload.filename).name
                content = await upload.read()
                dest.write_bytes(content)
                saved.append(dest)
            if not saved:
                raise HTTPException(status_code=400, detail="No valid files provided")
            stats = pipeline.ingest_files(saved, persist=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(status="ok", **stats)


@app.post("/ingest/directory", response_model=IngestResponse)
def ingest_directory() -> IngestResponse:
    pipeline = get_pipeline()
    try:
        stats = pipeline.ingest_directory(settings.raw_docs_dir, persist=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Directory ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(status="ok", **stats)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No index available. Upload documents via /ingest first.",
        )
    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            skip_verification=request.skip_verification,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result.to_dict())


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
