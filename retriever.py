"""Semantic retrieval helpers for the investment analysis RAG pipeline."""

from __future__ import annotations

import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

TOP_K = 10


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = TOP_K,
) -> List[Document]:
    """Run semantic similarity search and return the top matching chunks."""
    logger.info("Running semantic retrieval with top_k=%s", top_k)
    results = vector_store.similarity_search(query, k=top_k)
    logger.info("Retrieved %s chunks for the query", len(results))
    return results


def print_retrieved_chunks(chunks: List[Document]) -> None:
    """Print retrieved chunks before answer generation for transparency."""
    logger.info("Printing %s retrieved chunks", len(chunks))
    print("\n================ Retrieved Context ================\n")
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        print(f"[Chunk {index}] Page {metadata.get('page')} | Source: {metadata.get('source')}")
        print(chunk.page_content)
        print("\n--------------------------------------------------\n")
