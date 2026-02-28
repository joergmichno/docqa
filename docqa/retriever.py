"""Document retrieval module for the docqa RAG pipeline.

Performs cosine similarity search over a TF-IDF index to find
the most relevant chunks for a given query.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from docqa.indexer import TFIDFIndex
from docqa.models import SearchResult


class DocumentRetriever:
    """Retrieves the most relevant document chunks for a query.

    Wraps a TFIDFIndex and provides a search interface that
    transforms the query with the fitted vectorizer, computes
    cosine similarities, and returns ranked results.

    Attributes:
        index: The TF-IDF index to search over.
    """

    def __init__(self, index: TFIDFIndex) -> None:
        """Initialize the retriever with a pre-built index.

        Args:
            index: A TFIDFIndex produced by DocumentIndexer.build_index().
        """
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for chunks most relevant to the query.

        Transforms the query using the fitted TF-IDF vectorizer,
        computes cosine similarity against all indexed chunks,
        and returns the top-k results sorted by descending score.

        Args:
            query: The natural-language question or search query.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of SearchResult objects ranked by relevance score.
            May return fewer than top_k results if the index is small.
        """
        if not query.strip():
            return []

        query_vector = self.index.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.index.matrix).flatten()

        num_results = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:num_results]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            if score <= 0.0:
                continue
            results.append(
                SearchResult(
                    chunk=self.index.chunks[idx],
                    score=score,
                    rank=rank,
                )
            )

        return results
