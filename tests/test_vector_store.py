"""
tests/test_vector_store.py

Unit tests for app/core/vector_store.py.

These tests use a fake/mock embedding function instead of loading the real
sentence-transformers model, so the tests run fast and don't need internet
access or a GPU. This is a standard testing pattern for ML pipelines:
test the surrounding logic (indexing, search, persistence) independently
from the heavy model itself.

Run with: pytest tests/test_vector_store.py -v
"""

import pytest
import numpy as np
import shutil
from pathlib import Path

import app.core.embeddings as embeddings_module
from app.core.chunking import Chunk


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """
    Replace the real embedding model with a deterministic fake for all
    tests in this file. autouse=True means every test gets this automatically.
    """
    def fake_embed_texts(texts):
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(len(texts), 384)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def fake_embed_single(text):
        return fake_embed_texts([text])[0]

    monkeypatch.setattr(embeddings_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(embeddings_module, "embed_single_text", fake_embed_single)

    # Re-import VectorStore AFTER patching, since it imports these functions directly
    import importlib
    import app.core.vector_store as vs_module
    importlib.reload(vs_module)
    yield vs_module


def make_sample_chunks(n=5):
    return [
        Chunk(chunk_id=f"chunk_{i}", text=f"This is sample chunk number {i}.",
              source_file="sample.txt", chunk_index=i)
        for i in range(n)
    ]


def test_add_chunks_increases_index_size(mock_embeddings):
    store = mock_embeddings.VectorStore()
    chunks = make_sample_chunks(5)
    store.add_chunks(chunks)
    assert store.index.ntotal == 5
    assert len(store.chunks) == 5


def test_search_returns_requested_k(mock_embeddings):
    store = mock_embeddings.VectorStore()
    store.add_chunks(make_sample_chunks(10))
    results = store.search("sample query", k=3)
    assert len(results) == 3


def test_search_caps_k_at_available_chunks(mock_embeddings):
    """If fewer chunks exist than k, search should not error or pad with junk."""
    store = mock_embeddings.VectorStore()
    store.add_chunks(make_sample_chunks(2))
    results = store.search("query", k=10)
    assert len(results) == 2


def test_search_on_empty_store_returns_empty_list(mock_embeddings):
    store = mock_embeddings.VectorStore()
    results = store.search("anything", k=3)
    assert results == []


def test_search_results_are_chunk_score_pairs(mock_embeddings):
    store = mock_embeddings.VectorStore()
    store.add_chunks(make_sample_chunks(3))
    results = store.search("query", k=2)
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)


def test_save_and_load_round_trip(mock_embeddings, tmp_path):
    store = mock_embeddings.VectorStore()
    store.add_chunks(make_sample_chunks(4))

    save_dir = tmp_path / "vector_store_test"
    store.save(str(save_dir))

    reloaded = mock_embeddings.VectorStore.load(str(save_dir))
    assert reloaded.index.ntotal == store.index.ntotal
    assert len(reloaded.chunks) == len(store.chunks)
    assert reloaded.chunks[0].chunk_id == store.chunks[0].chunk_id
