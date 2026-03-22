"""
POST /summarize — Summarise a document or text column via map-reduce.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.pipelines import summarizer
from app.schemas.requests import SummarizeRequest
from app.schemas.responses import SummarizeResponse
from app.utils.text_cleaner import clean_text, clean_texts

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarise a single document or a column of text values.

    Uses map-reduce chunking: split into chunks of 2000 chars,
    summarise each in parallel, then reduce to a final summary.
    """
    mode = request.get_mode()

    if mode == "document" and request.text:
        logger.info(
            "Document summarization requested (%d chars)", len(request.text)
        )
        cleaned = clean_text(request.text)
        result = await summarizer.summarize_document(cleaned)
        return SummarizeResponse(**result)

    elif mode == "column" and request.texts:
        logger.info(
            "Column summarization requested for %d texts", len(request.texts)
        )
        cleaned = clean_texts(request.texts)
        result = await summarizer.summarize_texts(cleaned)
        return SummarizeResponse(**result)

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' (single document) or 'texts' (column of strings).",
        )
