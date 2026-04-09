"""PDF ingestion utilities for the investment analysis RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF with PyMuPDF and convert each page into a LangChain document."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Starting PDF ingestion from %s", path)
    documents: List[Document] = []

    with fitz.open(path) as pdf:
        logger.info("Opened PDF with %s pages", pdf.page_count)
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text", sort=True).strip()
            if not page_text:
                logger.warning("Skipping empty page %s", page_index + 1)
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": str(path.resolve()),
                        "page": page_index + 1,
                        "total_pages": pdf.page_count,
                    },
                )
            )

    logger.info("Completed ingestion with %s non-empty page documents", len(documents))
    return documents
