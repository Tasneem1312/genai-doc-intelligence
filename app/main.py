"""
app/main.py

This is the FastAPI application — it exposes our RAG pipeline as a REST
API with three endpoints:

  POST /ingest   -> upload a document, it gets chunked + embedded + indexed
  POST /query    -> ask a question, get a grounded answer back
  GET  /health   -> simple health check showing how many chunks are indexed

WHY FASTAPI (vs. Flask, which is also in the JD)?
FastAPI is built on Pydantic for automatic request/response validation,
and on Starlette for native async support. It auto-generates interactive
API docs (visit /docs once running) directly from your Pydantic models
and type hints, with effectively zero extra code. For an API-first ML
service like this, that combination of validation + docs + async makes
it the more natural fit than Flask, which needs extra libraries to get
the same things.

HOW TO RUN THIS:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs in your browser for interactive docs.
"""

import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.core.chunking import process_document
from app.core.vector_store import VectorStore
from app.core.llm import generate_answer
from app.api.schemas import QueryRequest, QueryResponse, IngestResponse, HealthResponse

STORE_DIR = "data/processed/vector_store"

# A single in-memory VectorStore instance shared across all requests.
# In a larger production system this might be a managed vector DB
# (e.g. Pinecone, Weaviate, pgvector) instead of an in-process FAISS index,
# but the same chunk -> embed -> index -> retrieve logic applies.
vector_store = VectorStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the app starts up. If a previously saved index exists
    on disk, load it so the API doesn't start empty after every restart.
    """
    global vector_store
    if Path(STORE_DIR, "index.faiss").exists():
        vector_store = VectorStore.load(STORE_DIR)
        print(f"Loaded existing index with {vector_store.index.ntotal} chunks.")
    else:
        print("No existing index found. Starting empty — use /ingest to add documents.")
    yield
    # (no shutdown cleanup needed for this project)


app = FastAPI(
    title="GenAI Document Intelligence API",
    description="A Retrieval-Augmented Generation (RAG) API for natural-language Q&A over uploaded documents.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Basic health check — confirms the API is up and reports index size."""
    return HealthResponse(status="ok", chunks_indexed=vector_store.index.ntotal)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file. It gets saved temporarily, run through the
    chunking pipeline, embedded, and added to the in-memory FAISS index.
    The updated index is persisted to disk so it survives a restart.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    # Save the uploaded file to a temp path so our existing loader functions
    # (which expect a file path) can read it without modification.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = process_document(tmp_path, chunk_size=500, chunk_overlap=100)
        # Restore the original filename on each chunk for clean source citations
        for c in chunks:
            c.source_file = file.filename

        vector_store.add_chunks(chunks)
        vector_store.save(STORE_DIR)
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # always clean up the temp file

    return IngestResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from {file.filename}.",
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Ask a question. Retrieves the top_k most relevant chunks from the
    vector store, then asks the LLM to generate a grounded answer.
    """
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document via /ingest first.",
        )

    results = vector_store.search(request.question, k=request.top_k)
    retrieved_chunks = [chunk for chunk, score in results]

    generation_result = generate_answer(request.question, retrieved_chunks)

    return QueryResponse(
        question=request.question,
        answer=generation_result["answer"],
        sources=generation_result["sources"],
        retrieved_chunks=[c.text for c in retrieved_chunks],
    )
