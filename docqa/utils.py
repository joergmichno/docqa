"""Utility functions for text processing and formatting.

Provides helper functions used across the docqa pipeline for
text cleaning, sentence extraction, truncation, and display formatting.
"""

import re

from docqa.models import SearchResult


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces.

    Collapses multiple whitespace characters (spaces, tabs, newlines)
    into single spaces and strips the result.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text with normalized whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """Split text into individual sentences.

    Strips markdown headings (lines starting with ``#``) before
    splitting so that headings do not pollute extracted answers.
    Uses a regex-based approach that handles common sentence-ending
    punctuation (.!?) followed by whitespace or end of string.

    Args:
        text: Input text to split.

    Returns:
        List of non-empty sentences.
    """
    # Remove markdown heading lines (e.g. "# Title" or "## Subtitle")
    cleaned = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length, adding ellipsis if needed.

    Args:
        text: Text to truncate.
        max_length: Maximum number of characters. Defaults to 200.

    Returns:
        Truncated text with '...' appended if it was shortened.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def format_source(result: SearchResult) -> str:
    """Format a search result for terminal display.

    Shows the rank, score, document ID, and a preview of the chunk content.

    Args:
        result: A SearchResult to format.

    Returns:
        Formatted string for display.
    """
    preview = truncate(result.chunk.content, max_length=120)
    doc_id = result.chunk.document_id
    score_pct = result.score * 100
    return (
        f"  [{result.rank}] (score: {score_pct:.1f}%) "
        f"doc: {doc_id}\n"
        f"      {preview}"
    )
