"""
Sentiment analysis pipeline — HuggingFace Transformers.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest by default.
Loaded once at startup as a singleton via ``get_pipeline()``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_pipeline_instance = None


def get_pipeline():
    """Return the cached HuggingFace sentiment pipeline (lazy-loaded)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        from transformers import pipeline as hf_pipeline

        settings = get_settings()
        logger.info("Loading sentiment model: %s", settings.SENTIMENT_MODEL)
        _pipeline_instance = hf_pipeline(
            "sentiment-analysis",
            model=settings.SENTIMENT_MODEL,
            tokenizer=settings.SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment model loaded successfully.")
    return _pipeline_instance


# ── Batch analysis ───────────────────────────────────────────────────────────

# Label mapping for the cardiffnlp model
_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def analyze_batch(texts: list[str]) -> list[dict[str, Any]]:
    """
    Run sentiment analysis on a batch of texts.

    Returns a list of dicts: ``{"label": "positive"|"negative"|"neutral", "score": float}``
    """
    pipe = get_pipeline()
    raw_results = pipe(texts, batch_size=len(texts))

    results = []
    for item in raw_results:
        label = _LABEL_MAP.get(item["label"], item["label"])
        results.append({"label": label, "score": round(item["score"], 4)})
    return results


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-row sentiment labels into a column-level distribution.

    Returns::

        {
            "total": int,
            "distribution": {
                "positive": {"count": int, "percentage": float},
                "negative": {"count": int, "percentage": float},
                "neutral":  {"count": int, "percentage": float},
            },
            "average_confidence": float,
        }
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "distribution": {}, "average_confidence": 0.0}

    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    total_score = 0.0

    for r in results:
        label = r.get("label", "neutral")
        counts[label] = counts.get(label, 0) + 1
        total_score += r.get("score", 0.0)

    distribution = {}
    for label, count in counts.items():
        distribution[label] = {
            "count": count,
            "percentage": round((count / total) * 100, 2),
        }

    return {
        "total": total,
        "distribution": distribution,
        "average_confidence": round(total_score / total, 4),
    }
