"""Tests for the docqa.retriever module."""

import pytest

from docqa.indexer import DocumentIndexer, TFIDFIndex
from docqa.models import Document, SearchResult
from docqa.retriever import DocumentRetriever


@pytest.fixture()
def indexed_retriever() -> DocumentRetriever:
    """Build a retriever from two small documents."""
    indexer = DocumentIndexer()
    docs = [
        Document(
            id="doc_py",
            title="Python",
            content=(
                "Python is a high-level programming language. "
                "It supports dynamic typing and automatic memory management. "
                "Python is widely used for web development, data science, "
                "and artificial intelligence applications."
            ),
            source_path="/tmp/python.md",
        ),
        Document(
            id="doc_ai",
            title="AI Agents",
            content=(
                "An AI agent perceives its environment and takes actions. "
                "Agents use large language models as reasoning engines. "
                "Retrieval-augmented generation grounds answers in documents."
            ),
            source_path="/tmp/ai.md",
        ),
    ]
    chunks = []
    for doc in docs:
        chunks.extend(indexer.chunk_document(doc, chunk_size=500))
    index = indexer.build_index(chunks)
    return DocumentRetriever(index)


class TestDocumentRetriever:
    """Tests for the DocumentRetriever.search method."""

    def test_search_returns_results(
        self, indexed_retriever: DocumentRetriever
    ) -> None:
        """A relevant query produces non-empty results."""
        results = indexed_retriever.search("What is Python?")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ranking(self, indexed_retriever: DocumentRetriever) -> None:
        """Results are sorted by descending score."""
        results = indexed_retriever.search("programming language")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_search_ranks_are_sequential(
        self, indexed_retriever: DocumentRetriever
    ) -> None:
        """Result ranks are sequential starting from 1."""
        results = indexed_retriever.search("AI agents")
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_search_respects_top_k(
        self, indexed_retriever: DocumentRetriever
    ) -> None:
        """search returns at most top_k results."""
        results = indexed_retriever.search("language", top_k=1)
        assert len(results) <= 1

    def test_search_empty_query(
        self, indexed_retriever: DocumentRetriever
    ) -> None:
        """An empty query returns no results."""
        results = indexed_retriever.search("")
        assert results == []

    def test_search_relevance(
        self, indexed_retriever: DocumentRetriever
    ) -> None:
        """A Python query should rank the Python doc higher."""
        results = indexed_retriever.search("Python programming language")
        assert len(results) > 0
        assert results[0].chunk.document_id == "doc_py"
