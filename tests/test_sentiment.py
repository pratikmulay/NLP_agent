"""
Tests for the sentiment pipeline — mocks HuggingFace model calls.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from app.pipelines.sentiment import analyze_batch, aggregate, _LABEL_MAP


class TestAnalyzeBatch:
    def test_batch_returns_labels(self, mock_sentiment_pipeline):
        """Verify batch analysis returns mapped labels with scores."""
        with patch("app.pipelines.sentiment.get_pipeline", return_value=mock_sentiment_pipeline):
            texts = ["Great product!", "Terrible service.", "It's okay I guess."]
            results = analyze_batch(texts)

            assert len(results) == 3
            for r in results:
                assert "label" in r
                assert "score" in r
                assert r["label"] in {"positive", "negative", "neutral"}
                assert 0 <= r["score"] <= 1

    def test_label_mapping(self):
        """Verify the label map covers all cardiffnlp labels."""
        assert _LABEL_MAP["LABEL_0"] == "negative"
        assert _LABEL_MAP["LABEL_1"] == "neutral"
        assert _LABEL_MAP["LABEL_2"] == "positive"


class TestAggregate:
    def test_aggregation_counts(self):
        results = [
            {"label": "positive", "score": 0.9},
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.6},
        ]
        agg = aggregate(results)

        assert agg["total"] == 4
        assert agg["distribution"]["positive"]["count"] == 2
        assert agg["distribution"]["positive"]["percentage"] == 50.0
        assert agg["distribution"]["negative"]["count"] == 1
        assert agg["distribution"]["negative"]["percentage"] == 25.0
        assert agg["distribution"]["neutral"]["count"] == 1

    def test_aggregation_empty(self):
        agg = aggregate([])
        assert agg["total"] == 0
        assert agg["distribution"] == {}

    def test_average_confidence(self):
        results = [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.6},
        ]
        agg = aggregate(results)
        assert agg["average_confidence"] == 0.7
