"""
app/core/llm.py

This module takes (1) a user's question and (2) the retrieved chunks from
the vector store, and asks an LLM to generate a grounded answer.

THE CORE IDEA OF "GROUNDING" (why RAG reduces hallucination):
If you ask a raw LLM "What is our company's sick leave policy?", it has
never seen your company's handbook and will either refuse or, worse,
confidently make something up (a "hallucination").

Instead, we build a prompt that says, in effect: "Here are some excerpts
from the actual document. Answer the question using ONLY this information.
If the answer isn't in these excerpts, say you don't know." This forces
the model to ground its answer in real retrieved text instead of its own
possibly-wrong internal "memory" of similar-sounding facts.

MODEL CHOICE:
This module is built around HuggingFace's `transformers` pipeline using a
small open-source instruction-tuned model (`google/flan-t5-base`), which
is free, runs on CPU, and needs no API key — good for a portfolio project
anyone can clone and run without paying for an OpenAI key.

The function `build_prompt()` is model-agnostic — if you later want to
swap in OpenAI's API or a larger local model, you only need to change
`generate_answer()`, not the rest of the app.
"""

from transformers import pipeline
from app.core.chunking import Chunk

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_generator = None


def get_generator():
    """Lazy-load the text generation pipeline as a singleton."""
    global _generator
    if _generator is None:
        _generator = pipeline("text-generation", model=_MODEL_NAME)
    return _generator


def build_prompt(question: str, retrieved_chunks: list[Chunk], max_context_chars: int = 1500) -> str:
    """
    Construct a grounded prompt from the question + retrieved chunks.

    We cap the total context length (max_context_chars) because LLMs have
    a finite context window — feeding too much text in either wastes
    tokens, slows inference, or in extreme cases gets truncated and
    silently drops the chunk that actually had the answer in it.
    """
    context_parts = []
    running_length = 0

    for chunk in retrieved_chunks:
        if running_length + len(chunk.text) > max_context_chars:
            break
        context_parts.append(f"[Source: {chunk.source_file}]\n{chunk.text}")
        running_length += len(chunk.text)

    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant answering questions using only the "
        "provided context. If the answer is not contained in the context, "
        "say 'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    messages = [{"role": "user", "content": prompt}]
    return messages


def generate_answer(question: str, retrieved_chunks: list[Chunk]) -> dict:
    """
    Full generation step: build the grounded prompt, run it through the
    local LLM, and return both the answer and the sources used (for
    transparency/citation in the API response).
    """
    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    prompt = build_prompt(question, retrieved_chunks)
    generator = get_generator()

    messages = build_prompt(question, retrieved_chunks)
    generator = get_generator()

    result = generator(messages, max_new_tokens=150, do_sample=False)
    answer_text = result[0]["generated_text"][-1]["content"].strip()

    sources = sorted(set(c.source_file for c in retrieved_chunks))

    return {
        "answer": answer_text,
        "sources": sources,
    }


# Quick manual test: python -m app.core.llm
if __name__ == "__main__":
    from app.core.chunking import process_document
    from app.core.vector_store import VectorStore

    chunks = process_document(
        "data/raw_docs/employee_handbook.txt", chunk_size=400, chunk_overlap=80
    )
    store = VectorStore()
    store.add_chunks(chunks)

    question = "How many days of sick leave am I entitled to?"
    retrieved = [chunk for chunk, score in store.search(question, k=3)]

    result = generate_answer(question, retrieved)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
