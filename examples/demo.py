"""Demo script for the docqa document Q&A tool.

Indexes the sample documents in examples/sample_docs/ and asks
a few questions to demonstrate the RAG pipeline in action.
"""

import sys
from pathlib import Path

# Ensure the project root is on the import path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from docqa.answerer import AnswerGenerator
from docqa.indexer import DocumentIndexer
from docqa.retriever import DocumentRetriever
from docqa.utils import format_source


def main() -> None:
    """Run the demo: index sample docs and ask questions."""
    sample_dir = Path(__file__).resolve().parent / "sample_docs"

    print("=" * 60)
    print("  docqa Demo -- RAG-powered Document Q&A")
    print("=" * 60)
    print()

    # Step 1: Index
    print("[1/3] Indexing documents...")
    indexer = DocumentIndexer()
    docs = indexer.load_directory(str(sample_dir))
    print(f"      Loaded {len(docs)} document(s):")
    for doc in docs:
        print(f"        - {doc.title} ({doc.metadata.get('filename', '')})")

    all_chunks = []
    for doc in docs:
        chunks = indexer.chunk_document(doc, chunk_size=500, overlap=50)
        all_chunks.extend(chunks)
    print(f"      Created {len(all_chunks)} chunk(s)")

    index = indexer.build_index(all_chunks)
    print(f"      Vocabulary size: {len(index.vectorizer.vocabulary_)}")
    print()

    # Step 2: Ask questions
    questions = [
        "What is a variable in Python?",
        "How do AI agents work?",
        "What is retrieval-augmented generation?",
        "How does error handling work in Python?",
    ]

    retriever = DocumentRetriever(index)
    answerer = AnswerGenerator(use_llm=False)

    print("[2/3] Asking questions (extractive mode)...")
    print()

    for question in questions:
        results = retriever.search(question, top_k=3)
        answer = answerer.generate(question, results)

        print(f"  Q: {question}")
        print(f"  A: {answer.answer_text}")
        print(f"  Confidence: {answer.confidence.value}")
        if answer.sources:
            print(f"  Top source: {answer.sources[0].chunk.document_id} "
                  f"(score: {answer.sources[0].score:.2%})")
        print()

    # Step 3: Summary
    print("[3/3] Done!")
    print()
    print("This demo showed how docqa:")
    print("  1. Loads and chunks documents from a directory")
    print("  2. Builds a TF-IDF index for similarity search")
    print("  3. Retrieves relevant chunks for a question")
    print("  4. Extracts the best-matching sentence as an answer")
    print()
    print("Set ANTHROPIC_API_KEY to enable LLM-powered answers.")
    print()


if __name__ == "__main__":
    main()
