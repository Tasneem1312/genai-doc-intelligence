"""
app/core/vector_store.py

This module wraps FAISS (Facebook AI Similarity Search) — a library built
for fast nearest-neighbor search over millions of vectors.

WHY NOT JUST LOOP THROUGH ALL CHUNKS AND COMPARE MANUALLY?
For a 7-chunk document, comparing the query vector to every chunk vector
in a Python loop would be fine. But real document collections can have
tens of thousands of chunks. Comparing a query against all of them one by
one in Python is slow. FAISS uses optimized C++ under the hood and special
indexing structures to make this search extremely fast, even at scale —
that's the whole reason it exists.

We use `IndexFlatIP` (Flat = no approximation, exact search; IP = Inner
Product). Because our embeddings are normalized to unit length, the inner
product between two vectors equals their cosine similarity. So this index
is doing exact cosine-similarity nearest-neighbor search.

This module keeps the FAISS index AND a parallel list of Chunk objects in
sync, so that when FAISS returns "row 4 matched", we can map that back to
the actual chunk text and source file.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path

from app.core.chunking import Chunk
from app.core.embeddings import embed_texts, embed_single_text


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        # IndexFlatIP = exact inner-product search (cosine similarity, since
        # our vectors are normalized). 384 = output size of all-MiniLM-L6-v2.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[Chunk] = []  # parallel list: chunks[i] <-> index row i

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Embed a list of chunks and add them to the FAISS index.
        Order matters: chunks[i] must correspond to row i in the index.
        """
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)  # shape: (n_chunks, 384)
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """
        Embed the query, search the FAISS index, and return the top-k
        (Chunk, similarity_score) pairs, ranked highest similarity first.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = embed_single_text(query).astype("float32").reshape(1, -1)
        k = min(k, self.index.ntotal)  # don't ask for more results than exist

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if there are fewer than k results
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(directory) / "index.faiss"))
        with open(Path(directory) / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str, embedding_dim: int = 384) -> "VectorStore":
        """Load a previously saved FAISS index and chunk metadata from disk."""
        store = cls(embedding_dim=embedding_dim)
        store.index = faiss.read_index(str(Path(directory) / "index.faiss"))
        with open(Path(directory) / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store


# Quick manual test: python -m app.core.vector_store
if __name__ == "__main__":
    from app.core.chunking import process_document

    chunks = process_document(
        "data/raw_docs/employee_handbook.txt", chunk_size=400, chunk_overlap=80
    )

    store = VectorStore()
    store.add_chunks(chunks)
    print(f"Indexed {store.index.ntotal} chunks")

    query = "How many days of sick leave do I get?"
    results = store.search(query, k=3)

    print(f"\nQuery: '{query}'\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"Rank {rank} | score={score:.3f} | {chunk.chunk_id}")
        print(f"  {chunk.text[:150]}...\n")

    # Persist and reload, to prove save/load works correctly
    store.save("data/processed/vector_store")
    reloaded = VectorStore.load("data/processed/vector_store")
    print(f"Reloaded store has {reloaded.index.ntotal} chunks (should match above)")
