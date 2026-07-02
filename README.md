# GenAI Document Intelligence — RAG-Based Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload documents (PDF or text) and ask natural-language questions about them, with grounded, source-cited answers — exposed as a REST API built with FastAPI.

## Why this project exists

Large Language Models don't know anything about your private documents, and they confidently make things up when asked about content they've never seen ("hallucination"). RAG solves this by retrieving the actual relevant text from your documents first, then asking the LLM to answer using *only* that retrieved context. This project implements that full pipeline from scratch — chunking, embeddings, vector search, prompt construction, and generation — rather than relying on a single black-box library call.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│  Document   │ --> │   Chunking   │ --> │   Embedding     │ --> │    FAISS    │
│ (.pdf/.txt) │     │ (sliding     │     │ (sentence-      │     │ Vector Index│
│             │     │  window)     │     │  transformers)  │     │             │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────┘
                                                                          │
                                                                          ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│   Answer    │ <-- │     LLM      │ <-- │ Prompt Builder  │ <-- │  Retrieval  │
│ + Sources   │     │ (flan-t5)    │     │  (grounding)    │     │ (top-k sim) │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────┘
```

**Pipeline stages:**
1. **Chunking** (`app/core/chunking.py`) — loads PDF/text documents and splits them into overlapping chunks using a sliding-window strategy, so semantic meaning isn't lost at chunk boundaries.
2. **Embedding** (`app/core/embeddings.py`) — converts each chunk into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`, capturing semantic meaning beyond exact keyword matches.
3. **Vector Store** (`app/core/vector_store.py`) — indexes embeddings in FAISS for fast cosine-similarity nearest-neighbor search, with save/load persistence to disk.
4. **LLM Generation** (`app/core/llm.py`) — builds a grounded prompt from the retrieved chunks and generates an answer using `google/flan-t5-base`, instructed to say "I don't know" rather than hallucinate when the answer isn't in the retrieved context.
5. **API** (`app/main.py`) — exposes `/ingest` and `/query` endpoints via FastAPI, with Pydantic request/response validation and auto-generated interactive docs at `/docs`.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| API framework | FastAPI | Async support, automatic validation via Pydantic, auto-generated OpenAPI docs |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, runs on CPU, strong performance for semantic search |
| Vector search | FAISS | Industry-standard, optimized C++ similarity search, scales to millions of vectors |
| LLM | HuggingFace `flan-t5-base` | Free, open-source, no API key required — fully reproducible by anyone who clones this repo |
| PDF parsing | pypdf | Lightweight, pure-Python PDF text extraction |
| Testing | pytest | Unit tests for chunking, vector store, and prompt-building logic |

> **Note on LLM choice:** This project uses a free local model so anyone can clone and run it without an API key or billing setup. The architecture is provider-agnostic — `app/core/llm.py` is the only file that would need to change to swap in OpenAI, Anthropic, or any other LLM provider.

## Project Structure

```
genai-doc-intelligence/
├── app/
│   ├── main.py              # FastAPI app and route definitions
│   ├── core/
│   │   ├── chunking.py      # Document loading + text chunking
│   │   ├── embeddings.py    # Embedding model wrapper
│   │   ├── vector_store.py  # FAISS index wrapper (add/search/save/load)
│   │   └── llm.py           # Prompt construction + answer generation
│   └── api/
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── test_chunking.py
│   ├── test_vector_store.py
│   └── test_llm.py
├── scripts/
│   └── ingest_cli.py        # CLI tool to index documents without the API
├── data/
│   ├── raw_docs/            # Sample document for testing
│   └── processed/           # Generated FAISS index (gitignored)
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/genai-doc-intelligence.git
cd genai-doc-intelligence

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for interactive Swagger documentation where you can try every endpoint directly in the browser.

### Example usage (curl)

**1. Ingest a document:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/raw_docs/employee_handbook.txt"
```

**2. Ask a question:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of sick leave am I entitled to?", "top_k": 3}'
```

**Response:**
```json
{
  "question": "How many days of sick leave am I entitled to?",
  "answer": "12 days of paid sick leave per year.",
  "sources": ["employee_handbook.txt"],
  "retrieved_chunks": ["...relevant excerpt text..."]
}
```

### Alternative: CLI ingestion (no server needed)

```bash
python scripts/ingest_cli.py data/raw_docs/employee_handbook.txt
```

## Running Tests

```bash
pytest tests/ -v
```

18 unit tests covering chunking logic (overlap correctness, edge cases), vector store operations (indexing, search, save/load round-trips), and prompt construction (grounding instructions, context length limits, source citation).

## Design Decisions Worth Noting

- **Sliding-window chunking with overlap** prevents sentences that fall on a chunk boundary from losing meaning in both halves — a 100-character overlap on 500-character chunks balances context preservation against redundant indexing.
- **Normalized embeddings + Inner Product FAISS index** is mathematically equivalent to cosine similarity but computationally cheaper, since it avoids a separate normalization step at query time.
- **Context length capping in the prompt builder** prevents silently truncating mid-chunk, which could cut off the exact sentence containing the answer.
- **Explicit "I don't know" grounding instruction** in the prompt is the primary defense against hallucination — the model is told to fail visibly rather than invent plausible-sounding but false answers.

## Possible Extensions

- Swap FAISS for a managed vector DB (Pinecone, Weaviate, pgvector) for multi-user persistence
- Add hybrid search (BM25 keyword search + semantic search) for queries with exact terms (IDs, dates)
- Add streaming responses for long answers
- Add a simple frontend (Streamlit) for non-technical users to query documents

## License

MIT
