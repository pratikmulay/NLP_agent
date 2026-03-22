"""
Summarization pipeline — Map-Reduce with LangChain text splitter.

Chunks text via RecursiveCharacterTextSplitter (chunk_size=2000, overlap=200),
then summarises each chunk in parallel and reduces to a final summary.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Text splitter (stateless, reuse freely) ──────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


# ── Document summarisation (map-reduce) ─────────────────────────────────────

async def summarize_document(text: str) -> dict[str, Any]:
    """
    Summarise a long document using map-reduce chunking.

    1. Split text into chunks (max 10).
    2. **Map**: summarise each chunk in parallel via the LLM provider.
    3. **Reduce**: summarise the concatenated chunk summaries.

    Returns::

        {
            "summary": str,
            "chunks_processed": int,
            "method": "direct" | "map_reduce",
        }
    """
    from app.llm.provider import get_llm_provider

    provider = get_llm_provider()
    chunks = _splitter.split_text(text)

    if not chunks:
        return {"summary": "", "chunks_processed": 0, "method": "direct"}

    # Direct summarisation for short texts
    if len(chunks) == 1:
        summary = await provider.summarize(chunks[0])
        return {"summary": summary, "chunks_processed": 1, "method": "direct"}

    # Map phase: parallel summarisation of each chunk (cap at 10)
    tasks = [provider.summarize(chunk) for chunk in chunks[:10]]
    chunk_summaries = await asyncio.gather(*tasks)

    # Reduce phase: combine chunk summaries into final summary
    combined = "\n\n".join(chunk_summaries)
    final_summary = await provider.summarize(combined, mode="final")

    return {
        "summary": final_summary,
        "chunks_processed": len(chunks[:10]),
        "method": "map_reduce",
    }


async def summarize_texts(texts: list[str]) -> dict[str, Any]:
    """
    Summarise a column of text values.

    Concatenates all texts (with separators), then runs map-reduce.
    """
    combined_text = "\n\n---\n\n".join(texts)
    result = await summarize_document(combined_text)
    result["total_texts"] = len(texts)
    return result
