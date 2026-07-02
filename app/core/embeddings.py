"""
app/core/embeddings.py

This module turns text into numbers (vectors) that capture semantic meaning.

WHAT IS AN EMBEDDING, IN PLAIN TERMS?
An embedding model reads a sentence and outputs a list of numbers (e.g. 384
numbers) that represent the "meaning" of that sentence in a high-dimensional
space. Sentences with similar meaning end up with vectors that are close
together (measured by cosine similarity), even if they don't share any
exact words. For example "How many sick days do I get?" and "What is the
sick leave entitlement?" will produce very similar vectors, even though
the only shared word is "sick".

This is exactly why RAG works better than plain keyword search: keyword
search would miss the connection between "sick days" and "sick leave
entitlement" unless the exact words matched. Embeddings capture meaning.

We use the `sentence-transformers` library with the `all-MiniLM-L6-v2`
model — it's small (~80MB), fast, runs on CPU, and is free/open-source.
It outputs 384-dimensional vectors.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Loading the model is slow (a few seconds), so we load it ONCE
# at module import time and reuse it everywhere, rather than reloading
# it on every function call.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model as a singleton.
    'Lazy' means we don't load it until it's actually needed —
    this keeps imports of this file fast if embeddings aren't used yet.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D numpy array of embeddings.
    Shape returned: (number_of_texts, 384)

    We normalize the embeddings (unit length) so that we can use a simple
    dot product as a similarity score later, instead of full cosine
    similarity — it's mathematically equivalent but faster.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def embed_single_text(text: str) -> np.ndarray:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([text])[0]


# Quick manual test: python app/core/embeddings.py
if __name__ == "__main__":
    samples = [
        "How many sick days do I get?",
        "What is the sick leave entitlement?",
        "What is the weather like today?",
    ]
    vectors = embed_texts(samples)
    print("Embedding shape:", vectors.shape)

    # Cosine similarity (dot product, since vectors are normalized)
    sim_related = float(np.dot(vectors[0], vectors[1]))
    sim_unrelated = float(np.dot(vectors[0], vectors[2]))

    print(f"\nSimilarity('sick days' vs 'sick leave entitlement'): {sim_related:.3f}")
    print(f"Similarity('sick days' vs 'weather'):                 {sim_unrelated:.3f}")
    print("\n(Notice the related pair scores much higher, even with zero shared words.)")
