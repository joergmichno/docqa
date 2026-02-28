"""Data models for the docqa document Q&A system.

Defines the core data structures used throughout the pipeline:
Document, Chunk, SearchResult, and Answer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnswerConfidence(Enum):
    """Confidence level for a generated answer."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NO_ANSWER = "NO_ANSWER"


@dataclass
class Document:
    """Represents a loaded document.

    Attributes:
        id: Unique identifier for the document.
        title: Human-readable title derived from filename.
        content: Full text content of the document.
        source_path: Filesystem path where the document was loaded from.
        metadata: Optional key-value metadata (e.g., file size, date).
    """

    id: str
    title: str
    content: str
    source_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of a document for retrieval.

    Documents are split into overlapping chunks so that each chunk
    fits within the context window and can be independently scored.

    Attributes:
        id: Unique identifier for this chunk.
        document_id: ID of the parent document.
        content: The text content of this chunk.
        start_pos: Character offset where this chunk starts in the original document.
        end_pos: Character offset where this chunk ends in the original document.
        metadata: Optional key-value metadata.
    """

    id: str
    document_id: str
    content: str
    start_pos: int
    end_pos: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieval result with relevance score.

    Attributes:
        chunk: The matched chunk.
        score: Cosine similarity score (0.0 to 1.0).
        rank: Position in the ranked result list (1-based).
    """

    chunk: Chunk
    score: float
    rank: int


@dataclass
class Answer:
    """Generated answer with source attribution.

    Attributes:
        question: The original user question.
        answer_text: The generated or extracted answer.
        confidence: Confidence level of the answer.
        sources: List of search results used to produce the answer.
        model_used: Name of the model used, or None for extractive mode.
    """

    question: str
    answer_text: str
    confidence: AnswerConfidence
    sources: list[SearchResult] = field(default_factory=list)
    model_used: Optional[str] = None
