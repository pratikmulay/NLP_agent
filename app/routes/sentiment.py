"""
POST /sentiment — Batch sentiment analysis on a text column.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.pipelines import sentiment as sentiment_pipeline
from app.schemas.requests import TextListRequest
from app.schemas.responses import SentimentResponse
from app.utils.batch_processor import process_text_column
from app.utils.text_cleaner import clean_texts

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextListRequest):
    """
    Run batch sentiment analysis on a list of texts.

    Returns per-row labels plus an aggregated column-level distribution
    (positive / negative / neutral counts and percentages).
    """
    logger.info("Sentiment analysis requested for %d texts", len(request.texts))

    cleaned = clean_texts(request.texts)
    per_row = await process_text_column(cleaned, sentiment_pipeline.analyze_batch)
    aggregated = sentiment_pipeline.aggregate(per_row)

    return SentimentResponse(
        total=aggregated["total"],
        distribution=aggregated["distribution"],
        average_confidence=aggregated["average_confidence"],
        per_row=per_row,
    )
