"""
Tests for app.utils.text_cleaner — HTML stripping, whitespace normalisation,
language detection, and truncation.
"""

from app.utils.text_cleaner import (
    strip_html,
    normalize_whitespace,
    detect_language,
    truncate,
    clean_text,
    clean_texts,
)


class TestStripHTML:
    def test_removes_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_preserves_plain_text(self):
        assert strip_html("No tags here") == "No tags here"

    def test_handles_empty(self):
        assert strip_html("") == ""

    def test_nested_tags(self):
        html = "<div><p>Text <a href='#'>link</a></p></div>"
        result = strip_html(html)
        assert "Text" in result
        assert "link" in result
        assert "<" not in result

    def test_script_tags(self):
        result = strip_html("<script>alert('xss')</script>Safe text")
        assert "Safe text" in result


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert normalize_whitespace("hello\n\n\nworld") == "hello world"

    def test_collapses_tabs(self):
        assert normalize_whitespace("hello\t\tworld") == "hello world"

    def test_strips_edges(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_handles_empty(self):
        assert normalize_whitespace("") == ""


class TestDetectLanguage:
    def test_english(self):
        result = detect_language(
            "This is a perfectly normal English sentence that should be detectable."
        )
        assert result == "en"

    def test_short_text_returns_none(self):
        assert detect_language("hi") is None

    def test_empty_returns_none(self):
        assert detect_language("") is None


class TestTruncate:
    def test_truncates_long_text(self):
        text = "a" * 20000
        result = truncate(text, max_length=100)
        assert len(result) == 100

    def test_preserves_short_text(self):
        text = "short"
        assert truncate(text, max_length=100) == "short"

    def test_handles_empty(self):
        assert truncate("", max_length=100) == ""

    def test_custom_max_length(self):
        text = "a" * 50
        result = truncate(text, max_length=25)
        assert len(result) == 25


class TestCleanText:
    def test_full_pipeline(self):
        html_text = "<p>Hello   <b>world</b>   </p>"
        result = clean_text(html_text)
        assert result == "Hello world"

    def test_preserves_clean_text(self):
        assert clean_text("Already clean") == "Already clean"


class TestCleanTexts:
    def test_batch_cleaning(self):
        texts = ["<p>One</p>", "  Two  ", "<b>Three</b>"]
        results = clean_texts(texts)
        assert len(results) == 3
        assert results[0] == "One"
        assert results[1] == "Two"
        assert results[2] == "Three"
