"""Tests for the docqa.models module."""

from docqa.models import (
    Answer,
    AnswerConfidence,
    Chunk,
    Document,
    SearchResult,
)


class TestDocument:
    """Tests for the Document dataclass."""

    def test_create_document(self) -> None:
        """A Document can be created with required fields."""
        doc = Document(
            id="doc_1",
            title="Test Doc",
            content="Hello world.",
            source_path="/tmp/test.txt",
        )
        assert doc.id == "doc_1"
        assert doc.title == "Test Doc"
        assert doc.content == "Hello world."
        assert doc.source_path == "/tmp/test.txt"

    def test_default_metadata(self) -> None:
        """Document metadata defaults to an empty dict."""
        doc = Document(id="d", title="T", content="C", source_path="/p")
        assert doc.metadata == {}

    def test_custom_metadata(self) -> None:
        """Document accepts custom metadata."""
        doc = Document(
            id="d",
            title="T",
            content="C",
            source_path="/p",
            metadata={"lang": "en"},
        )
        assert doc.metadata["lang"] == "en"


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_create_chunk(self) -> None:
        """A Chunk stores position information."""
        chunk = Chunk(
            id="c_0",
            document_id="doc_1",
            content="Some text.",
            start_pos=0,
            end_pos=10,
        )
        assert chunk.start_pos == 0
        assert chunk.end_pos == 10
        assert chunk.document_id == "doc_1"


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """A SearchResult wraps a Chunk with score and rank."""
        chunk = Chunk(id="c", document_id="d", content="t", start_pos=0, end_pos=1)
        result = SearchResult(chunk=chunk, score=0.85, rank=1)
        assert result.score == 0.85
        assert result.rank == 1
        assert result.chunk is chunk


class TestAnswer:
    """Tests for the Answer dataclass."""

    def test_create_answer_defaults(self) -> None:
        """An Answer defaults to empty sources and no model."""
        answer = Answer(
            question="What?",
            answer_text="Something.",
            confidence=AnswerConfidence.HIGH,
        )
        assert answer.sources == []
        assert answer.model_used is None

    def test_answer_with_model(self) -> None:
        """An Answer can specify the model used."""
        answer = Answer(
            question="Q",
            answer_text="A",
            confidence=AnswerConfidence.MEDIUM,
            model_used="claude-sonnet-4-20250514",
        )
        assert answer.model_used == "claude-sonnet-4-20250514"


class TestAnswerConfidence:
    """Tests for the AnswerConfidence enum."""

    def test_confidence_values(self) -> None:
        """All expected confidence levels exist."""
        assert AnswerConfidence.HIGH.value == "HIGH"
        assert AnswerConfidence.MEDIUM.value == "MEDIUM"
        assert AnswerConfidence.LOW.value == "LOW"
        assert AnswerConfidence.NO_ANSWER.value == "NO_ANSWER"
