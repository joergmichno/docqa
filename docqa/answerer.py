"""Answer generation module for the docqa RAG pipeline.

Supports two modes:
- **Offline (default):** Extractive answering using sentence-level
  similarity scoring against the query.
- **LLM mode:** When the ANTHROPIC_API_KEY environment variable is
  set, uses the Anthropic Claude API to generate a grounded answer
  from the retrieved context.
"""

import os
from typing import Optional

from docqa.models import Answer, AnswerConfidence, SearchResult
from docqa.utils import extract_sentences


def _score_sentence(sentence: str, query: str) -> float:
    """Score a sentence's relevance to the query using word overlap.

    Uses a simple Jaccard-like word overlap ratio as a fast,
    dependency-free relevance heuristic.

    Args:
        sentence: Candidate sentence.
        query: User query.

    Returns:
        Overlap ratio between 0.0 and 1.0.
    """
    query_words = set(query.lower().split())
    sentence_words = set(sentence.lower().split())
    if not query_words:
        return 0.0
    intersection = query_words & sentence_words
    union = query_words | sentence_words
    return len(intersection) / len(union) if union else 0.0


def _determine_confidence(
    score: float,
    num_results: int,
) -> AnswerConfidence:
    """Determine the answer confidence based on retrieval quality.

    Args:
        score: Top retrieval score.
        num_results: Number of search results available.

    Returns:
        An AnswerConfidence enum value.
    """
    if num_results == 0 or score <= 0.0:
        return AnswerConfidence.NO_ANSWER
    if score >= 0.4:
        return AnswerConfidence.HIGH
    if score >= 0.2:
        return AnswerConfidence.MEDIUM
    return AnswerConfidence.LOW


class AnswerGenerator:
    """Generates answers from retrieved document chunks.

    In offline mode, it selects the sentence from the top-ranked
    chunk that best matches the query.  When an Anthropic API key
    is available, it can optionally call the Claude API to produce
    a more fluent, synthesised answer.

    Attributes:
        use_llm: Whether LLM mode is active.
        api_key: The Anthropic API key (if available).
    """

    def __init__(self, use_llm: Optional[bool] = None) -> None:
        """Initialize the answer generator.

        Args:
            use_llm: Force LLM mode on/off. If ``None`` (default),
                     LLM mode is enabled automatically when the
                     ``ANTHROPIC_API_KEY`` environment variable is set.
        """
        self.api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        if use_llm is None:
            self.use_llm = bool(self.api_key)
        else:
            self.use_llm = use_llm

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> Answer:
        """Generate an answer for the given question.

        Dispatches to the LLM or extractive pipeline depending on
        the current mode configuration.

        Args:
            question: The user's natural-language question.
            results: Ranked search results from the retriever.

        Returns:
            An Answer object with the generated text and metadata.
        """
        if not results:
            return Answer(
                question=question,
                answer_text="No relevant documents found to answer this question.",
                confidence=AnswerConfidence.NO_ANSWER,
                sources=[],
                model_used=None,
            )

        if self.use_llm and self.api_key:
            return self._generate_with_llm(question, results)
        return self._generate_extractive(question, results)

    # ------------------------------------------------------------------
    # Extractive (offline) mode
    # ------------------------------------------------------------------

    def _generate_extractive(
        self,
        question: str,
        results: list[SearchResult],
    ) -> Answer:
        """Produce an extractive answer by selecting the best sentence.

        Iterates over the top result's sentences and returns the one
        with the highest word-overlap score against the query.

        Args:
            question: The user's question.
            results: Ranked search results.

        Returns:
            An Answer with the best-matching sentence.
        """
        top = results[0]
        sentences = extract_sentences(top.chunk.content)

        if not sentences:
            return Answer(
                question=question,
                answer_text=top.chunk.content,
                confidence=_determine_confidence(top.score, len(results)),
                sources=results,
                model_used=None,
            )

        scored = [(s, _score_sentence(s, question)) for s in sentences]
        best_sentence, best_score = max(scored, key=lambda x: x[1])

        combined_score = (top.score + best_score) / 2
        confidence = _determine_confidence(combined_score, len(results))

        return Answer(
            question=question,
            answer_text=best_sentence,
            confidence=confidence,
            sources=results,
            model_used=None,
        )

    # ------------------------------------------------------------------
    # LLM mode (Anthropic Claude)
    # ------------------------------------------------------------------

    def _generate_with_llm(
        self,
        question: str,
        results: list[SearchResult],
    ) -> Answer:
        """Generate an answer using the Anthropic Claude API.

        Builds a prompt with the retrieved context and asks Claude
        to synthesise a grounded answer.  Falls back to extractive
        mode if the API call fails for any reason.

        Args:
            question: The user's question.
            results: Ranked search results providing context.

        Returns:
            An Answer produced by the LLM, or an extractive fallback.
        """
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            return self._generate_extractive(question, results)

        context_parts: list[str] = []
        for r in results[:5]:
            context_parts.append(
                f"[Source: {r.chunk.document_id}, score: {r.score:.2f}]\n"
                f"{r.chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful document Q&A assistant. Answer the "
            "question based ONLY on the provided context. If the context "
            "does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = message.content[0].text
            model_name = "claude-sonnet-4-20250514"

            top_score = results[0].score if results else 0.0
            confidence = _determine_confidence(top_score, len(results))

            return Answer(
                question=question,
                answer_text=answer_text,
                confidence=confidence,
                sources=results,
                model_used=model_name,
            )

        except Exception:
            # Graceful fallback to extractive mode on any API error
            return self._generate_extractive(question, results)
