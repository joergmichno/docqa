"""Tests for the docqa.indexer module."""

import os
import tempfile
from pathlib import Path

import pytest

from docqa.indexer import DocumentIndexer, TFIDFIndex
from docqa.models import Chunk, Document


@pytest.fixture()
def sample_doc() -> Document:
    """Return a small sample Document."""
    return Document(
        id="doc_test",
        title="Test",
        content=(
            "Python is a programming language. "
            "It supports object-oriented programming. "
            "Functions are defined with the def keyword."
        ),
        source_path="/tmp/test.md",
    )


@pytest.fixture()
def temp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample documents."""
    (tmp_path / "hello.md").write_text(
        "Hello world. This is a test document about greetings.",
        encoding="utf-8",
    )
    (tmp_path / "python.txt").write_text(
        "Python is a high-level programming language. It is dynamically typed.",
        encoding="utf-8",
    )
    (tmp_path / "image.png").write_bytes(b"\x89PNG")  # should be skipped
    return tmp_path


class TestDocumentIndexer:
    """Tests for the DocumentIndexer class."""

    def test_load_directory(self, temp_docs_dir: Path) -> None:
        """load_directory finds .md and .txt files."""
        indexer = DocumentIndexer()
        docs = indexer.load_directory(str(temp_docs_dir))
        assert len(docs) == 2  # .md and .txt, not .png
        extensions = {d.metadata["extension"] for d in docs}
        assert extensions == {".md", ".txt"}

    def test_load_directory_not_found(self) -> None:
        """load_directory raises FileNotFoundError for missing directory."""
        indexer = DocumentIndexer()
        with pytest.raises(FileNotFoundError):
            indexer.load_directory("/nonexistent/path/xyz")

    def test_load_directory_custom_extensions(self, temp_docs_dir: Path) -> None:
        """load_directory respects custom extension filters."""
        indexer = DocumentIndexer()
        docs = indexer.load_directory(str(temp_docs_dir), extensions=[".md"])
        assert len(docs) == 1
        assert docs[0].metadata["extension"] == ".md"

    def test_chunk_document_basic(self, sample_doc: Document) -> None:
        """chunk_document produces at least one chunk."""
        indexer = DocumentIndexer()
        chunks = indexer.chunk_document(sample_doc, chunk_size=500, overlap=50)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_document_small_size(self, sample_doc: Document) -> None:
        """Small chunk_size produces multiple chunks."""
        indexer = DocumentIndexer()
        chunks = indexer.chunk_document(sample_doc, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_chunk_overlap(self, sample_doc: Document) -> None:
        """Consecutive chunks overlap by the specified amount."""
        indexer = DocumentIndexer()
        chunks = indexer.chunk_document(sample_doc, chunk_size=60, overlap=20)
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            assert chunks[1].start_pos < chunks[0].end_pos

    def test_chunk_empty_document(self) -> None:
        """Empty documents produce no chunks."""
        indexer = DocumentIndexer()
        doc = Document(id="e", title="Empty", content="   ", source_path="/tmp/e.md")
        chunks = indexer.chunk_document(doc)
        assert chunks == []

    def test_build_index(self, sample_doc: Document) -> None:
        """build_index creates a TFIDFIndex with correct counts."""
        indexer = DocumentIndexer()
        chunks = indexer.chunk_document(sample_doc, chunk_size=500)
        index = indexer.build_index(chunks)
        assert isinstance(index, TFIDFIndex)
        assert index.num_chunks == len(chunks)
        assert index.num_documents == 1

    def test_build_index_empty_raises(self) -> None:
        """build_index raises ValueError on empty input."""
        indexer = DocumentIndexer()
        with pytest.raises(ValueError, match="empty"):
            indexer.build_index([])


class TestTFIDFIndex:
    """Tests for TFIDFIndex save/load round-trip."""

    def test_save_and_load(self, sample_doc: Document, tmp_path: Path) -> None:
        """An index can be saved and loaded back identically."""
        indexer = DocumentIndexer()
        chunks = indexer.chunk_document(sample_doc)
        index = indexer.build_index(chunks)

        save_dir = str(tmp_path / "test_index")
        index.save(save_dir)

        loaded = TFIDFIndex.load(save_dir)
        assert loaded.num_chunks == index.num_chunks
        assert loaded.num_documents == index.num_documents
        assert loaded.chunks[0].content == index.chunks[0].content
