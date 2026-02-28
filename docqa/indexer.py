"""Document indexing module for the docqa RAG pipeline.

Handles loading documents from the filesystem, splitting them into
overlapping chunks, and building a TF-IDF index for similarity search.
"""

import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from docqa.models import Chunk, Document
from docqa.utils import clean_text


class TFIDFIndex:
    """Stores a TF-IDF vectorizer, the document-term matrix, and chunk metadata.

    This is the core search index. It wraps scikit-learn's TfidfVectorizer
    together with the sparse matrix and the chunk references needed to
    map matrix rows back to their source chunks.

    Attributes:
        vectorizer: Fitted TfidfVectorizer instance.
        matrix: Sparse TF-IDF matrix (rows = chunks, cols = terms).
        chunks: Ordered list of Chunk objects matching matrix rows.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: "NDArray",
        chunks: list[Chunk],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @property
    def num_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        return len(self.chunks)

    @property
    def num_documents(self) -> int:
        """Return the number of unique documents in the index."""
        return len({c.document_id for c in self.chunks})

    def save(self, directory: str) -> None:
        """Persist the index to disk.

        Saves the vectorizer and matrix as a pickle file and the chunk
        metadata as JSON so it remains inspectable.

        Args:
            directory: Directory path where index files will be written.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tfidf_model.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, f)

        chunks_data = [
            {
                "id": c.id,
                "document_id": c.document_id,
                "content": c.content,
                "start_pos": c.start_pos,
                "end_pos": c.end_pos,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "TFIDFIndex":
        """Load a previously saved index from disk.

        Args:
            directory: Directory containing index files.

        Returns:
            Reconstructed TFIDFIndex instance.

        Raises:
            FileNotFoundError: If index files are missing.
        """
        path = Path(directory)

        with open(path / "tfidf_model.pkl", "rb") as f:
            data = pickle.load(f)

        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = [
            Chunk(
                id=c["id"],
                document_id=c["document_id"],
                content=c["content"],
                start_pos=c["start_pos"],
                end_pos=c["end_pos"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]

        return cls(
            vectorizer=data["vectorizer"],
            matrix=data["matrix"],
            chunks=chunks,
        )


class DocumentIndexer:
    """Loads documents, chunks them, and builds a TF-IDF search index.

    This class orchestrates the indexing side of the RAG pipeline:
    reading files from a directory, splitting their content into
    overlapping text chunks, and fitting a TF-IDF vectorizer.

    Example:
        >>> indexer = DocumentIndexer()
        >>> docs = indexer.load_directory("./my_docs")
        >>> chunks = []
        >>> for doc in docs:
        ...     chunks.extend(indexer.chunk_document(doc))
        >>> index = indexer.build_index(chunks)
    """

    SUPPORTED_EXTENSIONS: list[str] = [".txt", ".md"]

    def load_directory(
        self,
        path: str,
        extensions: Optional[list[str]] = None,
    ) -> list[Document]:
        """Load all supported documents from a directory.

        Recursively walks the directory and loads files whose extensions
        match the provided list (defaults to .txt and .md).

        Args:
            path: Root directory to scan for documents.
            extensions: List of file extensions to include (e.g., ['.md']).
                        Defaults to ['.txt', '.md'].

        Returns:
            List of Document objects with their content loaded.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        allowed = extensions or self.SUPPORTED_EXTENSIONS
        documents: list[Document] = []

        for file_path in sorted(directory.rglob("*")):
            if file_path.suffix.lower() not in allowed:
                continue
            if not file_path.is_file():
                continue

            content = file_path.read_text(encoding="utf-8")
            title = file_path.stem.replace("_", " ").replace("-", " ").title()
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            documents.append(
                Document(
                    id=doc_id,
                    title=title,
                    content=content,
                    source_path=str(file_path),
                    metadata={
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "size_bytes": file_path.stat().st_size,
                    },
                )
            )

        return documents

    def chunk_document(
        self,
        doc: Document,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[Chunk]:
        """Split a document into overlapping text chunks.

        Uses a character-level sliding window. Each chunk overlaps
        with the previous one to avoid losing context at boundaries.

        Args:
            doc: The document to split.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of overlapping characters between consecutive chunks.

        Returns:
            List of Chunk objects covering the full document.
        """
        content = doc.content
        if not content.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = clean_text(content[start:end])

            if chunk_text:
                chunk_id = f"{doc.id}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        content=chunk_text,
                        start_pos=start,
                        end_pos=end,
                        metadata={"chunk_index": chunk_idx},
                    )
                )
                chunk_idx += 1

            if end >= len(content):
                break
            start = end - overlap

        return chunks

    def build_index(self, chunks: list[Chunk]) -> TFIDFIndex:
        """Build a TF-IDF index from a list of chunks.

        Fits a TfidfVectorizer on the chunk contents and produces
        the sparse document-term matrix used for similarity search.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            A TFIDFIndex ready for retrieval queries.

        Raises:
            ValueError: If the chunk list is empty.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        texts = [c.content for c in chunks]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)

        return TFIDFIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)
