"""
Tests for the NER pipeline — mocks spaCy model calls.
"""

from __future__ import annotations

from unittest.mock import patch

from app.pipelines.ner import extract_batch, aggregate


class TestExtractBatch:
    def test_extracts_entities(self, mock_spacy_nlp):
        """Verify entity extraction returns correct structure."""
        with patch("app.pipelines.ner.get_nlp", return_value=mock_spacy_nlp):
            texts = [
                "Apple Inc. reported record revenue.",
                "Barack Obama visited the White House.",
            ]
            results = extract_batch(texts)

            assert len(results) == 2

            # First text should have Apple Inc. (ORG)
            assert len(results[0]) >= 1
            assert results[0][0]["label"] == "ORG"
            assert results[0][0]["text"] == "Apple Inc."

            # Second text should have Obama (PERSON) and White House (FAC)
            assert len(results[1]) >= 2

    def test_empty_texts(self, mock_spacy_nlp):
        """Verify empty text produces no entities."""
        with patch("app.pipelines.ner.get_nlp", return_value=mock_spacy_nlp):
            results = extract_batch(["Nothing here"])
            assert len(results) == 1
            assert results[0] == []


class TestAggregate:
    def test_aggregation_structure(self):
        all_entities = [
            [
                {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                {"text": "Google", "label": "ORG", "start": 10, "end": 16},
            ],
            [
                {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
                {"text": "London", "label": "GPE", "start": 20, "end": 26},
            ],
        ]
        agg = aggregate(all_entities)

        assert agg["total_entities"] == 4
        assert "ORG" in agg["entity_types"]
        assert "GPE" in agg["entity_types"]
        assert agg["entity_types"]["ORG"]["count"] == 3

    def test_top_values_limited_to_10(self):
        """Ensure top_values is capped at 10 per entity type."""
        entities = [
            [{"text": f"Entity_{i}", "label": "ORG", "start": 0, "end": 5}]
            for i in range(20)
        ]
        agg = aggregate(entities)
        assert len(agg["entity_types"]["ORG"]["top_values"]) <= 10

    def test_empty(self):
        agg = aggregate([])
        assert agg["total_entities"] == 0
        assert agg["entity_types"] == {}
