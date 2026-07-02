"""
app/api/schemas.py

Pydantic models define the "shape" of data flowing in and out of our API.
FastAPI uses these to:
1. Automatically validate incoming requests (reject bad data with a clear
   error before it ever reaches our business logic).
2. Automatically generate interactive API docs (Swagger UI at /docs).
3. Serialize responses consistently.

This is what "clean, modular, testable API design" looks like in practice —
separating the data contracts from the route logic.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """What the client sends when asking a question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The natural language question to ask the document collection.",
        examples=["How many days of sick leave am I entitled to?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve from the vector store before generating an answer.",
    )


class QueryResponse(BaseModel):
    """What the API returns after running the RAG pipeline."""
    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[str] = Field(
        description="The raw chunk text used as context, for transparency/debugging."
    )


class IngestResponse(BaseModel):
    """Returned after a document has been processed and indexed."""
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
