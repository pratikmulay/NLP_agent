"""
Pydantic request models for all API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextListRequest(BaseModel):
    """Request body containing a list of texts (used by most endpoints)."""
    texts: list[str] = Field(..., min_length=1, description="List of text strings to process.")


class ClassifyRequest(BaseModel):
    """Request body for zero-shot classification."""
    texts: list[str] = Field(..., min_length=1, description="List of text strings to classify.")
    labels: list[str] = Field(..., min_length=2, description="Candidate labels for classification.")


class SummarizeRequest(BaseModel):
    """Request body for summarization — single document or list of texts."""
    text: str | None = Field(None, description="Single document text to summarize.")
    texts: list[str] | None = Field(None, description="List of text strings to summarize as a column.")

    def get_mode(self) -> str:
        if self.text:
            return "document"
        return "column"


class EmbedRequest(BaseModel):
    """Request body for embedding generation."""
    texts: list[str] = Field(..., min_length=1, description="List of text strings to embed.")
    store: bool = Field(False, description="Whether to store embeddings in pgvector (paid mode).")
    metadata: list[dict] | None = Field(None, description="Optional per-text metadata for storage.")
