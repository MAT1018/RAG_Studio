"""Embedding and vector store utilities for the investment analysis RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedding_model(model: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """Create a local sentence-transformers embedding model for FAISS indexing."""
    logger.info("Initializing local embedding model: %s", model)
    try:
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as error:
        raise RuntimeError(
            "The local embedding model could not be loaded. On the first run, "
            "sentence-transformers needs to download the model from Hugging Face once. "
            "Please make sure the machine has internet access for that initial download, then rerun setup."
        ) from error


def preview_embeddings(
    chunks: List[Document],
    embedding_model: Embeddings,
    limit: int = 3,
    dimensions: int = 8,
) -> List[Tuple[str, List[float]]]:
    """Generate sample embeddings for a small chunk subset and return printable previews."""
    logger.info(
        "Generating embedding preview for %s chunks with %s dimensions displayed",
        min(limit, len(chunks)),
        dimensions,
    )
    sample_chunks = chunks[:limit]
    vectors = embedding_model.embed_documents([chunk.page_content for chunk in sample_chunks])
    preview_data: List[Tuple[str, List[float]]] = []

    for chunk, vector in zip(sample_chunks, vectors):
        preview_data.append((chunk.page_content[:120].replace("\n", " "), vector[:dimensions]))

    return preview_data


def print_embedding_preview(preview_data: List[Tuple[str, List[float]]]) -> None:
    """Print sample chunk text and the first few embedding dimensions for inspection."""
    logger.info("Printing embedding preview for %s chunks", len(preview_data))
    for index, (snippet, vector_head) in enumerate(preview_data, start=1):
        print(f"\n--- Embedding Preview {index} ---")
        print(f"Chunk Snippet: {snippet}")
        print(f"Vector Head: {vector_head}")


def build_and_save_vectorstore(
    chunks: List[Document],
    persist_dir: str,
    embedding_model: Embeddings,
) -> FAISS:
    """Embed chunked documents, build a FAISS index, and persist it locally."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building FAISS vector store with %s chunks", len(chunks))
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(str(persist_path))
    logger.info("Saved FAISS vector store to %s", persist_path.resolve())
    return vector_store


def load_vectorstore(
    persist_dir: str,
    embedding_model: Embeddings,
) -> FAISS:
    """Load a previously saved FAISS vector store from local disk."""
    persist_path = Path(persist_dir)
    index_file = persist_path / "index.faiss"
    metadata_file = persist_path / "index.pkl"

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store directory not found: {persist_dir}. Run setup first to build the FAISS index."
        )
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "FAISS index files are missing. Expected both "
            f"{index_file} and {metadata_file}. Run setup first to generate the local vector database."
        )

    logger.info("Loading FAISS vector store from %s", persist_path.resolve())
    return FAISS.load_local(
        str(persist_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
