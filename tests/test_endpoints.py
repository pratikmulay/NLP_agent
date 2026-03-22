"""
Integration tests — hit all 7 API endpoints via httpx.AsyncClient.
All NLP models are mocked.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from httpx import AsyncClient, ASGITransport

from tests.conftest import SAMPLE_TEXTS, SAMPLE_LABELS, SAMPLE_DOCUMENT


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_sentiment_pipeline():
    mock_pipe = MagicMock()
    mock_pipe.side_effect = lambda texts, **kwargs: [
        {"label": "LABEL_2", "score": 0.9} for _ in texts
    ]
    return mock_pipe


def _mock_spacy_nlp():
    from tests.conftest import MockDoc, MockEntity

    mock_nlp = MagicMock()

    def mock_pipe(texts, **kwargs):
        for text in texts:
            ents = []
            if "Apple" in text:
                ents.append(MockEntity("Apple", "ORG", 0, 5))
            yield MockDoc(ents)

    mock_nlp.pipe = mock_pipe
    return mock_nlp


def _mock_sentence_transformer():
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(
        len(texts), 384
    ).astype("float32")
    return mock_model


def _mock_llm_provider():
    provider = MagicMock()
    provider.summarize = AsyncMock(return_value="Mock summary.")
    provider.classify = AsyncMock(return_value={
        "text": "sample",
        "label": "technology",
        "confidence": 0.85,
    })
    return provider


# ── Patches ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_all_models():
    """Patch all heavy NLP models so no downloads are needed."""
    with (
        patch("app.pipelines.sentiment._pipeline_instance", _mock_sentiment_pipeline()),
        patch("app.pipelines.sentiment.get_pipeline", return_value=_mock_sentiment_pipeline()),
        patch("app.pipelines.ner._nlp_instance", _mock_spacy_nlp()),
        patch("app.pipelines.ner.get_nlp", return_value=_mock_spacy_nlp()),
        patch("app.pipelines.embeddings._model_instance", _mock_sentence_transformer()),
        patch("app.pipelines.embeddings.get_model", return_value=_mock_sentence_transformer()),
        patch("app.llm.provider._provider_instance", _mock_llm_provider()),
        patch("app.llm.provider.get_llm_provider", return_value=_mock_llm_provider()),
    ):
        yield


@pytest.fixture
async def client():
    """Create an async test client with all models mocked."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "nlp-text-agent"


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "NLP/Text Agent"


@pytest.mark.asyncio
async def test_sentiment(client):
    resp = await client.post("/sentiment", json={"texts": SAMPLE_TEXTS[:3]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "distribution" in data
    assert "per_row" in data


@pytest.mark.asyncio
async def test_entities(client):
    resp = await client.post("/entities", json={"texts": SAMPLE_TEXTS[:3]})
    assert resp.status_code == 200
    data = resp.json()
    assert "total_entities" in data
    assert "entity_types" in data


@pytest.mark.asyncio
async def test_embed(client):
    resp = await client.post("/embed", json={"texts": SAMPLE_TEXTS[:3]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert data["dimension"] == 384
    assert len(data["embeddings"]) == 3


@pytest.mark.asyncio
async def test_classify(client):
    resp = await client.post(
        "/classify",
        json={"texts": SAMPLE_TEXTS[:2], "labels": SAMPLE_LABELS},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "label_distribution" in data
    assert "results" in data


@pytest.mark.asyncio
async def test_summarize_document(client):
    resp = await client.post("/summarize", json={"text": SAMPLE_DOCUMENT})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["method"] in ("direct", "map_reduce")


@pytest.mark.asyncio
async def test_summarize_column(client):
    resp = await client.post("/summarize", json={"texts": SAMPLE_TEXTS[:3]})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data


@pytest.mark.asyncio
async def test_summarize_no_input(client):
    resp = await client.post("/summarize", json={})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_sentiment_empty_texts(client):
    resp = await client.post("/sentiment", json={"texts": []})
    assert resp.status_code == 422  # validation error


@pytest.mark.asyncio
async def test_classify_needs_two_labels(client):
    resp = await client.post(
        "/classify",
        json={"texts": ["hello"], "labels": ["only_one"]},
    )
    assert resp.status_code == 422
