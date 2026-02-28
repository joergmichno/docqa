# docqa

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Tests passing](https://img.shields.io/badge/Tests-passing-brightgreen)

**RAG-powered document Q&A from the command line.**

`docqa` indexes local text documents using TF-IDF, retrieves the most relevant passages for a question, and returns an answer with source attribution -- all without requiring an internet connection or API key.

---

## What It Does

Given a directory of `.md` or `.txt` files, `docqa`:

1. **Indexes** the documents by splitting them into overlapping chunks and building a TF-IDF vector index.
2. **Retrieves** the most relevant chunks for a natural-language question using cosine similarity.
3. **Answers** the question by extracting the best-matching sentence from the top result (or by calling the Claude API if an API key is configured).

This is a classic **Retrieval-Augmented Generation (RAG)** pipeline implemented with minimal dependencies.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/joergmichno/docqa.git
cd docqa

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### First Query

```bash
# Index the example documents
docqa index examples/sample_docs/

# Ask a question
docqa ask "What is a variable in Python?"

# View index statistics
docqa info
```

### Example Output

```
Question: What is a variable in Python?

Answer: In Python, variables are created by assigning a value with the equals sign.
Confidence: HIGH
Mode: extractive (offline)

Sources:
  [1] (score: 62.3%) doc: doc_a1b2c3d4
      In Python, variables are created by assigning a value with the equals sign. Python uses dynamic typing, which...
  [2] (score: 31.1%) doc: doc_e5f6g7h8
      Python is a high-level, interpreted programming language known for its clear syntax and readability...
```

## How It Works

```
                  docqa RAG Pipeline

  Documents        Index           Retrieve         Answer
  ---------        -----           --------         ------
  .md / .txt  -->  Chunk into  --> Cosine sim  -->  Extract best
  files            overlapping    search over      sentence (or
                   segments +     TF-IDF vectors   call Claude
                   TF-IDF fit                      API)
```

### Architecture

| Module         | Responsibility                                      |
| -------------- | --------------------------------------------------- |
| `models.py`    | Dataclasses: Document, Chunk, SearchResult, Answer   |
| `indexer.py`   | Load files, chunk text, build TF-IDF index           |
| `retriever.py` | Cosine similarity search over the TF-IDF index       |
| `answerer.py`  | Extractive or LLM-based answer generation            |
| `cli.py`       | argparse CLI with `index`, `ask`, and `info` commands|
| `utils.py`     | Text cleaning, sentence splitting, formatting        |

### Key Design Decisions

- **TF-IDF + cosine similarity** provides a fast, offline retrieval baseline that requires no GPU or external service.
- **Overlapping chunks** ensure that information near chunk boundaries is not lost.
- **Extractive answering** selects the most relevant sentence using word overlap, keeping the tool fully functional without an API key.
- **Optional LLM mode** upgrades to Claude-powered answers when `ANTHROPIC_API_KEY` is set, with automatic fallback on errors.

## CLI Reference

### `docqa index <directory>`

Index all `.md` and `.txt` files in the given directory.

```bash
docqa index ./my_documents/ --chunk-size 500 --overlap 50
```

| Flag           | Default          | Description                 |
| -------------- | ---------------- | --------------------------- |
| `--index-dir`  | `.docqa_index`   | Where to save the index     |
| `--chunk-size`  | `500`           | Characters per chunk        |
| `--overlap`    | `50`             | Overlap between chunks      |

### `docqa ask "<question>"`

Ask a question about the indexed documents.

```bash
docqa ask "How do AI agents work?" --top-k 3
```

| Flag          | Default         | Description                  |
| ------------- | --------------- | ---------------------------- |
| `--index-dir` | `.docqa_index`  | Index directory to query     |
| `--top-k`     | `5`             | Number of results to retrieve|

### `docqa info`

Display index statistics.

```bash
docqa info
```

## Optional LLM Mode

By default, `docqa` works entirely offline using extractive answering. To enable LLM-powered answers:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
docqa ask "What is retrieval-augmented generation?"
```

When the API key is set, `docqa` sends the retrieved context to the Claude API and returns a synthesised answer. If the API call fails for any reason, it falls back to extractive mode automatically.

Install the optional LLM dependency:

```bash
pip install -e ".[llm]"
```

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=docqa --cov-report=term-missing
```

## Project Structure

```
docqa/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── docqa/
│   ├── __init__.py
│   ├── cli.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── answerer.py
│   ├── models.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_answerer.py
│   └── test_models.py
└── examples/
    ├── sample_docs/
    │   ├── python_basics.md
    │   └── ai_agents.md
    └── demo.py
```

## Technologies Used

- **Python 3.10+** -- type hints, dataclasses, modern syntax
- **scikit-learn** -- TF-IDF vectorization and cosine similarity
- **NumPy** -- efficient numerical operations
- **Anthropic SDK** (optional) -- Claude API integration

---

Built by **Joerg Michno** | [michno.jrg@gmail.com](mailto:michno.jrg@gmail.com)
