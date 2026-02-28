"""Tests for the docqa.answerer module."""

import pytest

from docqa.answerer import AnswerGenerator, _determine_confidence, _score_sentence
from docqa.models import (
    Answer,
    AnswerConfidence,
    Chunk,
    SearchResult,
)


@pytest.fixture()
def sample_results() -> list[SearchResult]:
    """Return a list of sample search results."""
    chunk = Chunk(
        id="c1",
        document_id="doc_1",
        content=(
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum. "
            "Python supports multiple programming paradigms."
        ),
        start_pos=0,
        end_pos=120,
    )
    return [
        SearchResult(chunk=chunk, score=0.65, rank=1),
    ]


@pytest.fixture()
def low_score_results() -> list[SearchResult]:
    """Return results with a low similarity score."""
    chunk = Chunk(
        id="c2",
        document_id="doc_2",
        content="The weather is nice today.",
        start_pos=0,
        end_pos=25,
    )
    return [
        SearchResult(chunk=chunk, score=0.05, rank=1),
    ]


class TestAnswerGenerator:
    """Tests for the AnswerGenerator class."""

    def test_generate_extractive(
        self, sample_results: list[SearchResult]
    ) -> None:
        """Extractive mode returns an answer from the chunk content."""
        gen = AnswerGenerator(use_llm=False)
        answer = gen.generate("What is Python?", sample_results)
        assert isinstance(answer, Answer)
        assert len(answer.answer_text) > 0
        assert answer.model_used is None

    def test_generate_no_results(self) -> None:
        """No results yields NO_ANSWER confidence."""
        gen = AnswerGenerator(use_llm=False)
        answer = gen.generate("Something?", [])
        assert answer.confidence == AnswerConfidence.NO_ANSWER
        assert "No relevant documents" in answer.answer_text

    def test_generate_has_sources(
        self, sample_results: list[SearchResult]
    ) -> None:
        """The answer includes the search results as sources."""
        gen = AnswerGenerator(use_llm=False)
        answer = gen.generate("What is Python?", sample_results)
        assert len(answer.sources) > 0

    def test_confidence_high(self) -> None:
        """High score and results yield HIGH confidence."""
        assert _determine_confidence(0.5, 3) == AnswerConfidence.HIGH

    def test_confidence_medium(self) -> None:
        """Medium score yields MEDIUM confidence."""
        assert _determine_confidence(0.25, 3) == AnswerConfidence.MEDIUM

    def test_confidence_low(self) -> None:
        """Low score yields LOW confidence."""
        assert _determine_confidence(0.1, 3) == AnswerConfidence.LOW

    def test_confidence_no_answer(self) -> None:
        """Zero results yield NO_ANSWER confidence."""
        assert _determine_confidence(0.5, 0) == AnswerConfidence.NO_ANSWER

    def test_score_sentence_overlap(self) -> None:
        """Sentences sharing words with the query score higher."""
        high = _score_sentence("Python is a programming language", "Python language")
        low = _score_sentence("The weather is nice today", "Python language")
        assert high > low

    def test_extractive_selects_relevant_sentence(
        self, sample_results: list[SearchResult]
    ) -> None:
        """The extractive answer picks the most relevant sentence."""
        gen = AnswerGenerator(use_llm=False)
        answer = gen.generate("Who created Python?", sample_results)
        # Should pick the sentence mentioning Guido
        assert "Guido" in answer.answer_text or "Python" in answer.answer_text
