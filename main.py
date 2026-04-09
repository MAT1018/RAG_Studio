"""CLI entry point for the investment analysis RAG system."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from chunking import chunk_documents, print_chunks
from embed import (
    build_and_save_vectorstore,
    build_embedding_model,
    load_vectorstore,
    preview_embeddings,
    print_embedding_preview,
)
from generator import build_llm, generate_answer, print_formatted_answer
from ingest import ingest_pdf
from retriever import print_retrieved_chunks, retrieve_relevant_chunks


DEFAULT_INDEX_DIR = "storage/faiss_index"
MANIFEST_FILENAME = "manifest.json"


def configure_logging() -> None:
    """Configure application-wide logging output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def compute_file_sha256(file_path: str) -> str:
    """Compute a stable SHA-256 fingerprint for a file on disk."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(persist_dir: str) -> Path:
    """Return the metadata manifest path stored alongside the FAISS index."""
    return Path(persist_dir) / MANIFEST_FILENAME


def has_ready_index(persist_dir: str) -> bool:
    """Return True when the persisted FAISS files are already present."""
    persist_path = Path(persist_dir)
    return (persist_path / "index.faiss").exists() and (persist_path / "index.pkl").exists()


def load_index_manifest(persist_dir: str) -> dict:
    """Load persisted index metadata when available."""
    path = manifest_path(persist_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_index_manifest(
    persist_dir: str,
    pdf_path: str,
    document_count: int,
    chunk_count: int,
) -> None:
    """Persist basic index metadata so the CLI can reuse an existing index safely."""
    path = manifest_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf_name": Path(pdf_path).name,
        "pdf_path": str(Path(pdf_path).resolve()),
        "pdf_sha256": compute_file_sha256(pdf_path),
        "document_count": document_count,
        "chunk_count": chunk_count,
    }
    path.write_text(json.dumps(payload, indent=2))


def index_matches_pdf(pdf_path: str, persist_dir: str) -> bool:
    """Return True when the current PDF matches the PDF used to build the saved index."""
    manifest = load_index_manifest(persist_dir)
    if not manifest or not has_ready_index(persist_dir):
        return False
    return manifest.get("pdf_sha256") == compute_file_sha256(pdf_path)


def run_setup(
    pdf_path: str,
    persist_dir: str,
    print_chunk_preview: bool,
    print_embedding_sample: bool,
) -> None:
    """Execute the ingestion, chunking, embedding, and vector persistence pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting setup pipeline")

    if index_matches_pdf(pdf_path, persist_dir):
        manifest = load_index_manifest(persist_dir)
        logger.info("Reusing existing FAISS index for the same PDF")
        print(
            "\nExisting FAISS index already matches this PDF. "
            f"Reusing saved index with {manifest.get('chunk_count', 0)} chunks.\n"
        )
        return

    documents = ingest_pdf(pdf_path)
    chunks = chunk_documents(documents)

    if print_chunk_preview:
        print_chunks(chunks)

    embedding_model = build_embedding_model()

    if print_embedding_sample:
        preview_data = preview_embeddings(chunks, embedding_model)
        print_embedding_preview(preview_data)

    build_and_save_vectorstore(chunks, persist_dir, embedding_model)
    save_index_manifest(
        persist_dir=persist_dir,
        pdf_path=pdf_path,
        document_count=len(documents),
        chunk_count=len(chunks),
    )
    logger.info("Setup pipeline completed successfully")
    print(f"\nVector store saved to: {Path(persist_dir).resolve()}\n")


def run_query(query: str, persist_dir: str) -> None:
    """Execute retrieval and grounded answer generation for a user query."""
    logger = logging.getLogger(__name__)
    logger.info("Starting query pipeline")

    try:
        embedding_model = build_embedding_model()
        vector_store = load_vectorstore(persist_dir, embedding_model)
        retrieved_chunks = retrieve_relevant_chunks(vector_store, query)
        print_retrieved_chunks(retrieved_chunks)

        llm = build_llm()
        answer = generate_answer(llm, query, retrieved_chunks)
        print_formatted_answer(answer)
        logger.info("Query pipeline completed successfully")
    except FileNotFoundError as error:
        logger.error("Query aborted because the vector store is unavailable: %s", error)
        print(f"\nError: {error}\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for setup and query workflows."""
    parser = argparse.ArgumentParser(
        description="Production-ready RAG system for investment analysis using a PDF textbook."
    )
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="Ingest the PDF and build the FAISS index.")
    setup_parser.add_argument("--pdf", required=True, help="Path to the textbook PDF.")
    setup_parser.add_argument(
        "--persist-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory used to save the FAISS index locally.",
    )
    setup_parser.add_argument(
        "--print-chunks",
        action="store_true",
        help="Print a preview of generated chunks.",
    )
    setup_parser.add_argument(
        "--print-embeddings",
        action="store_true",
        help="Print a sample of embedding vectors.",
    )

    query_parser = subparsers.add_parser("query", help="Query the persisted FAISS index.")
    query_parser.add_argument("--question", required=True, help="Question to ask the RAG system.")
    query_parser.add_argument(
        "--persist-dir",
        default=DEFAULT_INDEX_DIR,
        help="Directory containing the saved FAISS index.",
    )

    return parser


def interactive_menu() -> str:
    """Show a simple CLI menu when no subcommand is provided."""
    print("Choose an option:")
    print("1. Setup (ingest + embed)")
    print("2. Query")
    return input("Enter 1 or 2: ").strip()


def main() -> None:
    """Run the CLI application with either subcommands or the interactive menu."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup":
        run_setup(
            pdf_path=args.pdf,
            persist_dir=args.persist_dir,
            print_chunk_preview=args.print_chunks,
            print_embedding_sample=args.print_embeddings,
        )
        return

    if args.command == "query":
        run_query(query=args.question, persist_dir=args.persist_dir)
        return

    selection = interactive_menu()
    if selection == "1":
        pdf_path = input("Enter the path to the textbook PDF: ").strip()
        print_chunks_choice = input("Print chunk previews? (y/n): ").strip().lower() == "y"
        print_embeddings_choice = input("Print embedding previews? (y/n): ").strip().lower() == "y"
        run_setup(
            pdf_path=pdf_path,
            persist_dir=DEFAULT_INDEX_DIR,
            print_chunk_preview=print_chunks_choice,
            print_embedding_sample=print_embeddings_choice,
        )
        return

    if selection == "2":
        question = input("Enter your investment analysis question: ").strip()
        run_query(query=question, persist_dir=DEFAULT_INDEX_DIR)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
