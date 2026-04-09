"""Chunking helpers for the investment analysis RAG pipeline."""

from __future__ import annotations

import logging
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split input documents into recursive text chunks with metadata preserved."""
    logger.info(
        "Chunking documents with chunk_size=%s and chunk_overlap=%s",
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    logger.info("Created %s chunks", len(chunks))
    return chunks


def print_chunks(chunks: List[Document], limit: int = 5) -> None:
    """Print a readable preview of chunk contents for debugging and demonstrations."""
    logger.info("Printing up to %s chunks for inspection", limit)
    for index, chunk in enumerate(chunks[:limit], start=1):
        metadata = chunk.metadata
        print(f"\n--- Chunk {index} ---")
        print(
            f"Source: {metadata.get('source')} | "
            f"Page: {metadata.get('page')} | "
            f"Chunk Length: {len(chunk.page_content)}"
        )
        print(chunk.page_content)
