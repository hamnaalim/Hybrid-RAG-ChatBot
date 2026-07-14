"""Text chunking utilities."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.loaders import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with stable ID and source metadata."""

    chunk_id: str
    content: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)
    index: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            source=data["source"],
            page=int(data.get("page", 1)),
            metadata=data.get("metadata", {}),
            index=int(data.get("index", 0)),
        )


def _make_chunk_id(source: str, page: int, index: int, content: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{index}|{content[:64]}".encode("utf-8")).hexdigest()
    return digest[:16]


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split documents into overlapping chunks with provenance metadata."""
    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    global_index = 0

    for doc in documents:
        pieces = splitter.split_text(doc.content)
        page = int(doc.metadata.get("page", 1))
        for local_index, text in enumerate(pieces):
            text = text.strip()
            if not text:
                continue
            chunk_id = _make_chunk_id(doc.source, page, local_index, text)
            meta = {
                **doc.metadata,
                "chunk_index": local_index,
                "global_index": global_index,
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=text,
                    source=doc.source,
                    page=page,
                    metadata=meta,
                    index=global_index,
                )
            )
            global_index += 1

    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks
