"""Streamlit frontend for the investment analysis RAG system."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from langchain_core.documents import Document

from chunking import chunk_documents
from embed import (
    build_and_save_vectorstore,
    build_embedding_model,
    load_vectorstore,
    preview_embeddings,
)
from generator import build_llm, generate_answer
from ingest import ingest_pdf
from retriever import retrieve_relevant_chunks


DEFAULT_INDEX_DIR = "storage/faiss_index"
UPLOAD_DIR = Path("storage/uploads")
MANIFEST_FILENAME = "manifest.json"
MANDATORY_QUERIES = [
    "how to deal with brokerage houses?",
    "what is theory of diversification?",
    "how to become intelligent investor?",
    "how to do business valuation?",
    "what is putting all eggs in one basket analogy?",
]


def configure_logging() -> None:
    """Configure logging for the Streamlit app session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def apply_app_styling() -> None:
    """Inject custom styling so the demo interface feels polished and intentional."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(192, 226, 255, 0.75), transparent 32%),
                radial-gradient(circle at top right, rgba(248, 212, 170, 0.85), transparent 28%),
                linear-gradient(180deg, #f4efe6 0%, #fbfaf8 42%, #eef3f7 100%);
            color: #112433;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.25rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.75rem 1.8rem;
            border-radius: 20px;
            background: rgba(255, 252, 246, 0.86);
            border: 1px solid rgba(17, 36, 51, 0.12);
            box-shadow: 0 22px 60px rgba(17, 36, 51, 0.09);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.4rem;
            color: #0d2231;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.6;
            color: #365062;
            margin: 0;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 0.5rem;
        }
        .metric-card {
            background: rgba(13, 34, 49, 0.92);
            color: #f8f5ef;
            padding: 1rem 1.1rem;
            border-radius: 16px;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.75;
        }
        .metric-value {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 750;
            color: #0d2231;
            margin-bottom: 0.4rem;
        }
        .context-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.82);
            border-left: 4px solid #b9652b;
            margin-bottom: 0.85rem;
        }
        .answer-card {
            padding: 1.25rem 1.3rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255, 249, 238, 0.94), rgba(255, 255, 255, 0.9));
            border: 1px solid rgba(185, 101, 43, 0.22);
            box-shadow: 0 18px 50px rgba(185, 101, 43, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize Streamlit session state used across setup and query actions."""
    st.session_state.setdefault("pdf_name", None)
    st.session_state.setdefault("vectorstore_ready", False)
    st.session_state.setdefault("chunk_preview", [])
    st.session_state.setdefault("embedding_preview", [])
    st.session_state.setdefault("retrieved_chunks", [])
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("persist_dir", DEFAULT_INDEX_DIR)
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("document_count", 0)
    st.session_state.setdefault("chunk_count", 0)


def list_persisted_files(persist_dir: str) -> List[str]:
    """List files inside the persisted vector store directory for demo verification."""
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return []

    return sorted(
        str(path.relative_to(persist_path.parent))
        for path in persist_path.rglob("*")
        if path.is_file()
    )


def save_uploaded_pdf(uploaded_file) -> Path:
    """Persist an uploaded PDF to the local storage directory for ingestion."""
    logger = logging.getLogger(__name__)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / uploaded_file.name
    logger.info("Saving uploaded PDF to %s", destination.resolve())
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


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


def load_index_manifest(persist_dir: str) -> dict:
    """Load persisted index metadata when available."""
    path = manifest_path(persist_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_index_manifest(
    persist_dir: str,
    pdf_path: str,
    pdf_name: str,
    document_count: int,
    chunk_count: int,
) -> None:
    """Persist basic index metadata so the app can reuse an existing index safely."""
    path = manifest_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf_name": pdf_name,
        "pdf_path": str(Path(pdf_path).resolve()),
        "pdf_sha256": compute_file_sha256(pdf_path),
        "document_count": document_count,
        "chunk_count": chunk_count,
    }
    path.write_text(json.dumps(payload, indent=2))


def has_ready_index(persist_dir: str) -> bool:
    """Return True when the persisted FAISS files are already present."""
    persist_path = Path(persist_dir)
    return (persist_path / "index.faiss").exists() and (persist_path / "index.pkl").exists()


def index_matches_pdf(pdf_path: str, persist_dir: str) -> bool:
    """Return True when the current PDF matches the PDF used to build the saved index."""
    manifest = load_index_manifest(persist_dir)
    if not manifest or not has_ready_index(persist_dir):
        return False
    return manifest.get("pdf_sha256") == compute_file_sha256(pdf_path)


def build_index(
    pdf_path: str,
    pdf_name: str,
    persist_dir: str,
    show_chunk_preview: bool,
    show_embedding_preview: bool,
) -> Tuple[int, int, List[Document], List[Tuple[str, List[float]]]]:
    """Run the ingestion and embedding pipeline for the uploaded textbook."""
    logger = logging.getLogger(__name__)
    logger.info("Starting frontend setup pipeline")

    documents = ingest_pdf(pdf_path)
    chunks = chunk_documents(documents)

    chunk_preview = chunks[:5] if show_chunk_preview else []

    embedding_model = build_embedding_model()
    embedding_preview: List[Tuple[str, List[float]]] = []
    if show_embedding_preview:
        try:
            embedding_preview = preview_embeddings(chunks, embedding_model)
        except RuntimeError as error:
            logger.warning("Embedding preview unavailable: %s", error)

    build_and_save_vectorstore(chunks, persist_dir, embedding_model)
    save_index_manifest(
        persist_dir=persist_dir,
        pdf_path=pdf_path,
        pdf_name=pdf_name,
        document_count=len(documents),
        chunk_count=len(chunks),
    )
    logger.info("Frontend setup pipeline completed")
    return len(documents), len(chunks), chunk_preview, embedding_preview


def answer_question(question: str, persist_dir: str) -> Tuple[List[Document], str]:
    """Retrieve textbook context and generate a grounded answer for the query."""
    logger = logging.getLogger(__name__)
    logger.info("Starting frontend query pipeline")

    embedding_model = build_embedding_model()
    vector_store = load_vectorstore(persist_dir, embedding_model)
    retrieved_chunks = retrieve_relevant_chunks(vector_store, question)

    llm = build_llm()
    answer = generate_answer(llm, question, retrieved_chunks)
    logger.info("Frontend query pipeline completed")
    return retrieved_chunks, answer


def render_hero() -> None:
    """Render the top section that introduces the assignment demo."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Investment Analysis RAG Studio</div>
            <p class="hero-subtitle">
                Upload a finance textbook, build the FAISS index, inspect chunking and embedding samples,
                and ask grounded investment-analysis questions with retrieved context shown before the answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    """Render a compact overview of the current demo state."""
    pdf_name = st.session_state.pdf_name or "Not loaded"
    index_state = "Ready" if st.session_state.vectorstore_ready else "Not ready"
    chunk_count = st.session_state.chunk_count

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card">
                <div class="metric-label">Textbook</div>
                <div class="metric-value">{pdf_name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Index Status</div>
                <div class="metric-value">{index_state}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Generated Chunks</div>
                <div class="metric-value">{chunk_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assignment_checklist() -> None:
    """Render the assignment-specific demo guidance and mandatory queries."""
    st.markdown('<div class="section-title">Assignment Checklist</div>', unsafe_allow_html=True)
    st.info(
        "For the video demo: upload the textbook, show chunk and embedding previews, "
        "show the local FAISS index files, and then ask the five mandatory questions exactly as provided."
    )
    for index, query in enumerate(MANDATORY_QUERIES, start=1):
        st.write(f"{index}. {query}")


def render_backend_verification(persist_dir: str) -> None:
    """Render local vector-store details for backend verification during the demo."""
    persisted_files = list_persisted_files(persist_dir)
    if not persisted_files:
        return

    st.markdown('<div class="section-title">Backend Verification</div>', unsafe_allow_html=True)
    st.write(f"Local vector database path: `{Path(persist_dir).resolve()}`")
    st.write("Persisted files:")
    for file_name in persisted_files:
        st.code(file_name, language="text")

    st.write(
        f"Indexed pages: {st.session_state.document_count} | "
        f"Indexed chunks: {st.session_state.chunk_count}"
    )


def render_chunk_preview(chunks: List[Document]) -> None:
    """Render sample generated chunks in expandable cards."""
    if not chunks:
        return

    st.markdown('<div class="section-title">Chunk Preview</div>', unsafe_allow_html=True)
    for index, chunk in enumerate(chunks, start=1):
        with st.expander(f"Chunk {index} | Page {chunk.metadata.get('page')}"):
            st.write(chunk.page_content)
            st.caption(f"Source: {chunk.metadata.get('source')}")


def render_embedding_preview(embedding_preview: List[Tuple[str, List[float]]]) -> None:
    """Render a compact table-like preview of chunk snippets and vector heads."""
    if not embedding_preview:
        return

    st.markdown('<div class="section-title">Embedding Preview</div>', unsafe_allow_html=True)
    for index, (snippet, vector_head) in enumerate(embedding_preview, start=1):
        with st.expander(f"Embedding Sample {index}"):
            st.write(f"Chunk snippet: {snippet}")
            st.code(str(vector_head), language="text")


def render_retrieved_chunks(chunks: List[Document]) -> None:
    """Render retrieved context before the final answer."""
    if not chunks:
        return

    st.markdown('<div class="section-title">Retrieved Context</div>', unsafe_allow_html=True)
    for index, chunk in enumerate(chunks, start=1):
        st.markdown(
            f"""
            <div class="context-card">
                <strong>Chunk {index}</strong> | Page {chunk.metadata.get('page')}<br/>
                <small>{chunk.metadata.get('source')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(chunk.page_content)


def render_answer(answer: str) -> None:
    """Render the final grounded answer in a visually distinct container."""
    if not answer:
        return

    st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit frontend for the investment analysis RAG workflow."""
    st.set_page_config(
        page_title="Investment Analysis RAG",
        page_icon="📘",
        layout="wide",
    )
    configure_logging()
    initialize_state()
    apply_app_styling()
    render_hero()
    render_metrics()

    with st.sidebar:
        st.header("Pipeline Controls")
        persist_dir = st.text_input("Vector Store Directory", value=st.session_state.persist_dir)
        st.session_state.persist_dir = persist_dir

        uploaded_file = st.file_uploader("Upload Textbook PDF", type=["pdf"])
        show_chunk_preview = st.checkbox("Show chunk preview", value=True)
        show_embedding_preview = st.checkbox("Show embedding preview", value=True)

        if st.button("Run Setup", use_container_width=True):
            if uploaded_file is None:
                st.error("Upload a PDF first to build the index.")
            else:
                with st.spinner("Ingesting PDF, chunking text, creating embeddings, and saving FAISS index..."):
                    try:
                        pdf_path = save_uploaded_pdf(uploaded_file)
                        if index_matches_pdf(str(pdf_path), persist_dir):
                            manifest = load_index_manifest(persist_dir)
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.vectorstore_ready = True
                            st.session_state.document_count = manifest.get("document_count", 0)
                            st.session_state.chunk_count = manifest.get("chunk_count", 0)
                            st.success(
                                "This PDF was already indexed earlier. Reusing the existing FAISS index instead of rebuilding it."
                            )
                        else:
                            document_count, chunk_count, chunk_preview, embedding_preview = build_index(
                                pdf_path=str(pdf_path),
                                pdf_name=uploaded_file.name,
                                persist_dir=persist_dir,
                                show_chunk_preview=show_chunk_preview,
                                show_embedding_preview=show_embedding_preview,
                            )
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.vectorstore_ready = True
                            st.session_state.chunk_preview = chunk_preview
                            st.session_state.embedding_preview = embedding_preview
                            st.session_state.document_count = document_count
                            st.session_state.chunk_count = chunk_count
                            st.success(
                                f"Setup complete. Parsed {document_count} pages and generated {chunk_count} chunks."
                            )
                            if show_embedding_preview and not embedding_preview:
                                st.warning(
                                    "Chunking and setup completed, but embedding preview could not be shown. "
                                    "This usually means the Gemini API quota or rate limit was reached during preview."
                                )
                    except RuntimeError as error:
                        st.session_state.vectorstore_ready = False
                        st.error(str(error))

        st.divider()
        st.caption("Mandatory assignment queries")
        for query_option in MANDATORY_QUERIES:
            if st.button(query_option, use_container_width=True):
                st.session_state.question_input = query_option

        st.divider()
        question = st.text_area(
            "Investment Question",
            placeholder="Example: What does the textbook say about portfolio diversification and risk reduction?",
            height=140,
            key="question_input",
        )

        if st.button("Ask Question", use_container_width=True):
            if not question.strip():
                st.error("Enter a question before querying the index.")
            elif not Path(persist_dir).exists():
                st.error("Run setup first so the FAISS index exists before querying.")
            else:
                with st.spinner("Retrieving context and generating a grounded answer..."):
                    try:
                        retrieved_chunks, answer = answer_question(question.strip(), persist_dir)
                        st.session_state.retrieved_chunks = retrieved_chunks
                        st.session_state.answer = answer
                        st.success("Answer generated from retrieved textbook context.")
                    except FileNotFoundError as error:
                        st.error(str(error))
                    except RuntimeError as error:
                        st.error(str(error))

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        render_assignment_checklist()
        render_backend_verification(persist_dir)
        render_chunk_preview(st.session_state.chunk_preview)
        render_embedding_preview(st.session_state.embedding_preview)

    with right_col:
        render_retrieved_chunks(st.session_state.retrieved_chunks)
        render_answer(st.session_state.answer)


if __name__ == "__main__":
    main()
