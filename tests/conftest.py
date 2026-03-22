"""
Shared test fixtures for the NLP/Text Agent test suite.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Force free-mode config for tests
os.environ.setdefault("ENV_FILE", ".env.free")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")
os.environ.setdefault("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("HF_CACHE_DIR", "/tmp/hf_cache_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("NLP_BATCH_SIZE", "32")
os.environ.setdefault("MAX_TEXT_LENGTH", "10000")
os.environ.setdefault("PORT", "8005")


# ── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Apple Inc. reported record revenue of $120 billion in Q4 2024.",
    "The weather in London is absolutely terrible today, raining all day.",
    "I love this product! Best purchase I've ever made. Highly recommend!",
    "The United Nations held a meeting in Geneva about climate change.",
    "This movie was a complete waste of time. I want my money back.",
    "Tesla's new electric vehicle has amazing autopilot features.",
    "Barack Obama visited the White House to meet with officials.",
    "Python programming is great for data science and machine learning.",
    "The restaurant served cold food and the service was very slow.",
    "Google announced their new AI model at the annual developer conference.",
]

SAMPLE_LABELS = ["technology", "politics", "entertainment", "sports", "business"]

SAMPLE_DOCUMENT = """
Artificial intelligence (AI) has rapidly evolved from a niche research topic
to one of the most transformative technologies of the 21st century. Modern AI
systems can process natural language, recognize images, and generate creative
content with remarkable accuracy.

The development of large language models (LLMs) such as GPT-4, Claude, and
Llama has revolutionized how we interact with computers. These models are
trained on vast amounts of text data and can perform tasks ranging from
translation to code generation.

However, the rise of AI also brings significant challenges. Concerns about bias,
privacy, job displacement, and the environmental impact of training large models
are increasingly important topics in public discourse.

Governments around the world are beginning to regulate AI development. The
European Union's AI Act, the United States' executive orders on AI safety, and
China's AI governance framework represent different approaches to ensuring
responsible AI development.

Despite these challenges, the potential benefits of AI are enormous. From
healthcare diagnostics to climate modeling, AI applications continue to expand
into new domains, promising to solve some of humanity's most pressing problems.
"""


@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS.copy()


@pytest.fixture
def sample_labels():
    return SAMPLE_LABELS.copy()


@pytest.fixture
def sample_document():
    return SAMPLE_DOCUMENT


# ── Mock HuggingFace pipeline ────────────────────────────────────────────────

@pytest.fixture
def mock_sentiment_pipeline():
    """Mock the HuggingFace sentiment pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.side_effect = lambda texts, **kwargs: [
        {"label": ["LABEL_2", "LABEL_0", "LABEL_2", "LABEL_1", "LABEL_0"][i % 5], "score": 0.95}
        for i, _ in enumerate(texts)
    ]
    return mock_pipe


# ── Mock spaCy ───────────────────────────────────────────────────────────────

class MockEntity:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class MockDoc:
    def __init__(self, ents):
        self.ents = ents


@pytest.fixture
def mock_spacy_nlp():
    """Mock the spaCy NLP model."""
    mock_nlp = MagicMock()

    def mock_pipe(texts, **kwargs):
        for text in texts:
            entities = []
            if "Apple" in text:
                entities.append(MockEntity("Apple Inc.", "ORG", 0, 10))
            if "London" in text:
                entities.append(MockEntity("London", "GPE", 15, 21))
            if "Obama" in text:
                entities.append(MockEntity("Barack Obama", "PERSON", 0, 12))
            if "Google" in text:
                entities.append(MockEntity("Google", "ORG", 0, 6))
            if "Tesla" in text:
                entities.append(MockEntity("Tesla", "ORG", 0, 5))
            if "United Nations" in text:
                entities.append(MockEntity("United Nations", "ORG", 4, 18))
            if "Geneva" in text:
                entities.append(MockEntity("Geneva", "GPE", 40, 46))
            if "White House" in text:
                entities.append(MockEntity("White House", "FAC", 25, 36))
            yield MockDoc(entities)

    mock_nlp.pipe = mock_pipe
    return mock_nlp


# ── Mock SentenceTransformer ─────────────────────────────────────────────────

@pytest.fixture
def mock_sentence_transformer():
    """Mock the SentenceTransformer model."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(
        len(texts), 384
    ).astype("float32")
    return mock_model


# ── Mock LLM Provider ───────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_provider():
    """Mock the LLM provider for summarize/classify."""
    provider = MagicMock()
    provider.summarize = AsyncMock(return_value="This is a mock summary of the text.")
    provider.classify = AsyncMock(return_value={
        "text": "sample",
        "label": "technology",
        "confidence": 0.85,
    })
    return provider
