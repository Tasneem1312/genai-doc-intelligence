"""
tests/test_llm.py

Unit tests for app/core/llm.py — specifically the prompt-building logic,
which is pure string manipulation and doesn't require loading the actual
LLM model (so these tests run instantly with no GPU/network needed).

Run with: pytest tests/test_llm.py -v
"""

from app.core.llm import build_prompt
from app.core.chunking import Chunk


def make_chunk(text, source="doc.txt"):
    return Chunk(chunk_id="c1", text=text, source_file=source, chunk_index=0)


def test_prompt_includes_question():
    chunks = [make_chunk("Sick leave is 12 days per year.")]
    prompt = build_prompt("How many sick days do I get?", chunks)
    assert "How many sick days do I get?" in prompt


def test_prompt_includes_chunk_text():
    chunks = [make_chunk("Sick leave is 12 days per year.")]
    prompt = build_prompt("question", chunks)
    assert "Sick leave is 12 days per year." in prompt


def test_prompt_includes_source_citation():
    chunks = [make_chunk("Some policy text.", source="handbook.pdf")]
    prompt = build_prompt("question", chunks)
    assert "handbook.pdf" in prompt


def test_prompt_respects_max_context_chars():
    long_chunk = make_chunk("X" * 2000)
    prompt = build_prompt("question", [long_chunk], max_context_chars=500)
    # The chunk is longer than max_context_chars, so it should be dropped
    # entirely rather than truncated mid-sentence (per our chunking logic)
    assert "X" * 2000 not in prompt


def test_prompt_has_grounding_instruction():
    """The prompt must instruct the model to say 'I don't know' rather than hallucinate."""
    chunks = [make_chunk("Some text.")]
    prompt = build_prompt("question", chunks)
    assert "don't have enough information" in prompt.lower()


def test_empty_chunks_still_produces_valid_prompt():
    prompt = build_prompt("question", [])
    assert "question" in prompt.lower() or "Question" in prompt
