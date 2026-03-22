"""
Text cleaning utilities — strip HTML, normalize whitespace, detect language,
truncate at MAX_TEXT_LENGTH. All functions are pure and stateless.
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup

from app.config import get_settings


# ── HTML stripping ───────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove all HTML tags from text, preserving inner content."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


# ── Whitespace normalisation ─────────────────────────────────────────────────

_MULTI_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of any whitespace to a single space and strip edges."""
    if not text:
        return ""
    return _MULTI_WHITESPACE.sub(" ", text).strip()


# ── Language detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> Optional[str]:
    """
    Return ISO-639-1 language code for *text*, or None on failure.
    Uses langdetect (Google's language-detection port).
    """
    if not text or len(text.strip()) < 20:
        return None
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return None


# ── Truncation ───────────────────────────────────────────────────────────────

def truncate(text: str, max_length: Optional[int] = None) -> str:
    """Truncate *text* to at most *max_length* characters (default from settings)."""
    if not text:
        return ""
    if max_length is None:
        max_length = get_settings().MAX_TEXT_LENGTH
    if len(text) <= max_length:
        return text
    return text[:max_length]


# ── Composite cleaner ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Full cleaning pipeline: strip HTML → normalise whitespace → truncate."""
    text = strip_html(text)
    text = normalize_whitespace(text)
    text = truncate(text)
    return text


def clean_texts(texts: list[str]) -> list[str]:
    """Clean a list of texts through the full pipeline."""
    return [clean_text(t) for t in texts]
