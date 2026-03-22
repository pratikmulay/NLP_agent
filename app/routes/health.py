"""
GET /health — Service health check.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.schemas.responses import HealthResponse

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Return service health status including loaded models and uptime.
    """
    from app.pipelines import sentiment, ner, embeddings, topics

    models_loaded = {
        "sentiment": sentiment._pipeline_instance is not None,
        "spacy_ner": ner._nlp_instance is not None,
        "embeddings": embeddings._model_instance is not None,
        "bertopic": topics._model_instance is not None,
    }

    return HealthResponse(
        status="healthy",
        service="nlp-text-agent",
        version="1.0.0",
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - _start_time, 2),
    )
