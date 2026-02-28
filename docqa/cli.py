"""Command-line interface for the docqa document Q&A tool.

Provides three subcommands:
- ``index`` -- Index a directory of documents.
- ``ask``   -- Ask a question about the indexed documents.
- ``info``  -- Display index statistics.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from docqa.answerer import AnswerGenerator
from docqa.indexer import DocumentIndexer, TFIDFIndex
from docqa.models import AnswerConfidence
from docqa.retriever import DocumentRetriever
from docqa.utils import format_source

# ------------------------------------------------------------------
# ANSI colour helpers
# ------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

DEFAULT_INDEX_DIR = ".docqa_index"


def _colour(text: str, code: str) -> str:
    """Wrap *text* in an ANSI colour code."""
    return f"{code}{text}{_RESET}"


def _confidence_colour(confidence: AnswerConfidence) -> str:
    """Return an ANSI colour code for the given confidence level."""
    mapping = {
        AnswerConfidence.HIGH: _GREEN,
        AnswerConfidence.MEDIUM: _YELLOW,
        AnswerConfidence.LOW: _RED,
        AnswerConfidence.NO_ANSWER: _RED,
    }
    return mapping.get(confidence, _RESET)


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------


def _handle_index(args: argparse.Namespace) -> None:
    """Index a directory of documents and save the index to disk."""
    directory = args.directory
    index_dir = args.index_dir or DEFAULT_INDEX_DIR

    print(_colour("Indexing documents...", _CYAN))
    indexer = DocumentIndexer()

    docs = indexer.load_directory(directory)
    if not docs:
        print(_colour("No documents found in the specified directory.", _RED))
        sys.exit(1)

    print(f"  Loaded {_colour(str(len(docs)), _BOLD)} document(s)")

    all_chunks = []
    for doc in docs:
        chunks = indexer.chunk_document(
            doc,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        all_chunks.extend(chunks)
        print(f"  {doc.title}: {len(chunks)} chunk(s)")

    index = indexer.build_index(all_chunks)
    index.save(index_dir)

    print(
        _colour(
            f"\nIndex saved to {index_dir}/ "
            f"({index.num_documents} docs, {index.num_chunks} chunks)",
            _GREEN,
        )
    )


def _handle_ask(args: argparse.Namespace) -> None:
    """Ask a question about the indexed documents."""
    index_dir = args.index_dir or DEFAULT_INDEX_DIR

    if not Path(index_dir).exists():
        print(
            _colour(
                f"No index found at '{index_dir}/'. Run 'docqa index <dir>' first.",
                _RED,
            )
        )
        sys.exit(1)

    index = TFIDFIndex.load(index_dir)
    retriever = DocumentRetriever(index)
    answerer = AnswerGenerator()

    results = retriever.search(args.question, top_k=args.top_k)
    answer = answerer.generate(args.question, results)

    conf_code = _confidence_colour(answer.confidence)

    print()
    print(_colour("Question: ", _BOLD) + args.question)
    print()
    print(_colour("Answer: ", _BOLD) + answer.answer_text)
    print(
        _colour("Confidence: ", _BOLD)
        + _colour(answer.confidence.value, conf_code)
    )
    if answer.model_used:
        print(_colour("Model: ", _BOLD) + answer.model_used)
    else:
        print(_colour("Mode: ", _BOLD) + "extractive (offline)")

    if answer.sources:
        print()
        print(_colour("Sources:", _BOLD))
        for src in answer.sources[:3]:
            print(format_source(src))
    print()


def _handle_info(args: argparse.Namespace) -> None:
    """Display statistics about the current index."""
    index_dir = args.index_dir or DEFAULT_INDEX_DIR

    if not Path(index_dir).exists():
        print(
            _colour(
                f"No index found at '{index_dir}/'. Run 'docqa index <dir>' first.",
                _RED,
            )
        )
        sys.exit(1)

    index = TFIDFIndex.load(index_dir)

    print()
    print(_colour("docqa Index Info", _BOLD))
    print(f"  Index directory : {index_dir}")
    print(f"  Documents       : {index.num_documents}")
    print(f"  Chunks          : {index.num_chunks}")
    vocab_size = len(index.vectorizer.vocabulary_)
    print(f"  Vocabulary size : {vocab_size}")
    print()


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="RAG-powered document Q&A from the command line.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- index -------------------------------------------------------
    idx_parser = subparsers.add_parser(
        "index",
        help="Index a directory of documents",
    )
    idx_parser.add_argument(
        "directory",
        help="Path to the directory containing documents",
    )
    idx_parser.add_argument(
        "--index-dir",
        default=None,
        help=f"Where to save the index (default: {DEFAULT_INDEX_DIR})",
    )
    idx_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Characters per chunk (default: 500)",
    )
    idx_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)",
    )

    # -- ask ---------------------------------------------------------
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about indexed documents",
    )
    ask_parser.add_argument(
        "question",
        help="The question to ask",
    )
    ask_parser.add_argument(
        "--index-dir",
        default=None,
        help=f"Index directory to use (default: {DEFAULT_INDEX_DIR})",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )

    # -- info --------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Show index statistics",
    )
    info_parser.add_argument(
        "--index-dir",
        default=None,
        help=f"Index directory to inspect (default: {DEFAULT_INDEX_DIR})",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the docqa CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "index": _handle_index,
        "ask": _handle_ask,
        "info": _handle_info,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
