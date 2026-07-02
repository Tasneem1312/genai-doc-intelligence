"""
app/core/chunking.py

This module handles two things:
1. Loading raw documents (PDF or plain text) into clean text.
2. Splitting that text into overlapping "chunks" small enough to feed
   into an embedding model and, later, an LLM's context window.

WHY CHUNKING MATTERS (read this before you skip it):
Embedding models and LLMs can only process a limited amount of text at once.
A 50-page PDF can't be embedded as a single vector and expect that vector
to meaningfully represent every paragraph in it. So we break documents into
smaller, semantically coherent pieces ("chunks"), embed each one separately,
and at query time retrieve only the chunks relevant to the question.

We use a "sliding window with overlap" strategy: each chunk overlaps with
the previous one by a fixed number of characters. This prevents a sentence
or idea that happens to fall exactly on a chunk boundary from being cut in
half and losing meaning in both halves.
"""

from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Chunk:
    """A single chunk of text plus metadata about where it came from."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int


def load_text_file(file_path: str) -> str:
    """Read a plain .txt file and return its full text."""
    return Path(file_path).read_text(encoding="utf-8")


def load_pdf_file(file_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.
    PDFs store text per-page, so we loop through every page and
    concatenate the text, separating pages with a newline.
    """
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n".join(full_text)


def load_document(file_path: str) -> str:
    """Dispatch to the correct loader based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(file_path)
    elif suffix == ".txt":
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")


def clean_text(text: str) -> str:
    """
    Basic text normalization before chunking:
    - Collapse multiple blank lines into one
    - Strip leading/trailing whitespace on each line
    This is a simple NLP preprocessing step. Real production pipelines
    might also handle de-hyphenation, encoding fixes, header/footer removal, etc.
    """
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line != ""]
    return "\n".join(lines)


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using a sliding window over characters.

    Parameters
    ----------
    text : the full cleaned document text
    source_file : name of the file this text came from (for citation/debugging)
    chunk_size : max number of characters per chunk
    chunk_overlap : how many characters consecutive chunks share

    How the sliding window works:
    Imagine text is 1200 characters long, chunk_size=500, overlap=100.
    Chunk 0: characters [0:500]
    Chunk 1: characters [400:900]   <- starts 100 chars before chunk 0 ended
    Chunk 2: characters [800:1200]  <- starts 100 chars before chunk 1 ended
    The "step" each time is (chunk_size - chunk_overlap) = 400.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end].strip()

        if chunk_str:  # skip empty chunks (can happen at the very end)
            chunks.append(
                Chunk(
                    chunk_id=f"{Path(source_file).stem}_{index}",
                    text=chunk_str,
                    source_file=source_file,
                    chunk_index=index,
                )
            )
            index += 1

        start += step

    return chunks


def process_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    """
    Full pipeline for one document: load -> clean -> chunk.
    This is the single function the rest of the app will call.
    """
    raw_text = load_document(file_path)
    cleaned = clean_text(raw_text)
    return chunk_text(cleaned, source_file=Path(file_path).name,
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# Quick manual test when running this file directly:
# python app/core/chunking.py
if __name__ == "__main__":
    sample_path = "data/raw_docs/employee_handbook.txt"
    chunks = process_document(sample_path, chunk_size=400, chunk_overlap=80)
    print(f"Loaded {sample_path}")
    print(f"Produced {len(chunks)} chunks\n")
    for c in chunks[:3]:
        print(f"--- {c.chunk_id} ({len(c.text)} chars) ---")
        print(c.text[:200], "...\n")
