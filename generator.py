"""Answer generation utilities for the investment analysis RAG pipeline."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import List

from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash-lite"

load_dotenv()


def format_provider_error(error: Exception) -> str:
    """Convert Gemini exceptions into user-friendly messages for the app and CLI."""
    if isinstance(error, ResourceExhausted):
        return (
            "Gemini rejected the request because the current API quota or rate limit was exceeded. "
            "Check your Google AI Studio usage, plan, and billing details, then retry."
        )
    if isinstance(error, GoogleAPIError):
        return f"Gemini API request failed: {error}"
    return str(error)


def extract_retry_delay_seconds(error: Exception, default_seconds: int = 30) -> int:
    """Extract retry delay from a Gemini quota error message when available."""
    message = str(error)

    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))) + 1)

    match = re.search(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)\s*\}", message, re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)) + 1)

    return default_seconds


def is_quota_error(error: Exception) -> bool:
    """Return True when Gemini reports rate limiting or quota exhaustion."""
    message = str(error).lower()
    return isinstance(error, ResourceExhausted) or "429" in message or "quota" in message


def build_llm(model: str = DEFAULT_CHAT_MODEL) -> ChatGoogleGenerativeAI:
    """Create a deterministic Gemini chat model for grounded financial answers."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set.")

    logger.info("Initializing Gemini chat model: %s", model)
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def build_grounded_prompt(query: str, retrieved_chunks: List[Document]) -> List[object]:
    """Create a strict context-grounded prompt that forbids unsupported claims."""
    context = "\n\n".join(
        [
            (
                f"[Chunk {index} | Page {doc.metadata.get('page')}]\n"
                f"{doc.page_content}"
            )
            for index, doc in enumerate(retrieved_chunks, start=1)
        ]
    )

    system_prompt = (
        "You are a financial analysis assistant. Answer in a precise, professional, "
        "financial expert tone.\n"
        "You must follow these rules:\n"
        "1. Use only the information contained in the provided context.\n"
        "2. Do not add outside knowledge, assumptions, or invented facts.\n"
        "3. If the context does not contain enough information, say: "
        "'The provided textbook context does not contain enough information to answer this question.'\n"
        "4. Reference the relevant chunk numbers or page numbers when helpful.\n"
        "5. Keep the answer clear, structured, and grounded in the retrieved text."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Produce a grounded answer based only on the context above."
    )

    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def generate_answer(llm: ChatGoogleGenerativeAI, query: str, retrieved_chunks: List[Document]) -> str:
    """Generate a final answer using only the retrieved textbook context."""
    logger.info("Generating grounded answer from retrieved context")
    messages = build_grounded_prompt(query, retrieved_chunks)
    while True:
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as error:
            if not is_quota_error(error):
                logger.exception("Failed to generate grounded answer")
                raise RuntimeError(format_provider_error(error)) from error

            sleep_seconds = extract_retry_delay_seconds(error)
            logger.warning(
                "Gemini generation quota reached. Sleeping for %s seconds before retry.",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def print_formatted_answer(answer: str) -> None:
    """Print the final answer in a clean presentation format."""
    logger.info("Printing final formatted answer")
    print("\n================ Final Answer ================\n")
    print(answer)
    print("\n=============================================\n")
