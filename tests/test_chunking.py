"""
tests/test_chunking.py

Unit tests for app/core/chunking.py.

Run with: pytest tests/test_chunking.py -v
"""

import pytest
from app.core.chunking import chunk_text, clean_text, Chunk


def test_clean_text_removes_blank_lines():
    raw = "Line one\n\n\nLine two\n   \nLine three"
    result = clean_text(raw)
    assert "\n\n" not in result
    assert "Line one" in result
    assert "Line two" in result
    assert "Line three" in result


def test_chunk_text_produces_correct_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, source_file="test.txt", chunk_size=500, chunk_overlap=100)

    # step = chunk_size - chunk_overlap = 400, so we expect chunks at
    # [0:500], [400:900], [800:1200(clamped to 1000)]
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_text_no_empty_chunks():
    text = "Short text that is much shorter than the chunk size."
    chunks = chunk_text(text, source_file="test.txt", chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text.strip() != ""


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", source_file="test.txt", chunk_size=100, chunk_overlap=100)


def test_chunk_ids_are_unique_and_sequential():
    text = "B" * 1500
    chunks = chunk_text(text, source_file="doc.txt", chunk_size=500, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # all unique
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_preserves_source_file_name():
    chunks = chunk_text("some sample text here", source_file="handbook.pdf", chunk_size=50, chunk_overlap=10)
    assert all(c.source_file == "handbook.pdf" for c in chunks)
