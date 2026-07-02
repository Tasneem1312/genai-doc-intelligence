"""
scripts/ingest_cli.py

A simple command-line tool to index a document without needing to start
the FastAPI server first. Useful for quickly testing the pipeline or
pre-loading documents before deployment.

Usage:
    python scripts/ingest_cli.py data/raw_docs/employee_handbook.txt
"""

import sys
from pathlib import Path

# Allow running this script from the project root: python scripts/ingest_cli.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunking import process_document
from app.core.vector_store import VectorStore

STORE_DIR = "data/processed/vector_store"


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest_cli.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Processing {file_path} ...")
    chunks = process_document(file_path, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")

    if Path(STORE_DIR, "index.faiss").exists():
        store = VectorStore.load(STORE_DIR)
        print(f"Loaded existing index ({store.index.ntotal} chunks already indexed).")
    else:
        store = VectorStore()
        print("Starting a new index.")

    store.add_chunks(chunks)
    store.save(STORE_DIR)
    print(f"Done. Index now contains {store.index.ntotal} chunks total.")


if __name__ == "__main__":
    main()
