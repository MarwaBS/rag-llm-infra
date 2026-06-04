"""End-to-end RAG pipeline test — deterministic, no network, no API key."""
from example import DOCS, embed, rag_answer, retrieve


def test_embed_is_deterministic_and_shaped() -> None:
    a = embed(["hello world"])
    b = embed(["hello world"])
    assert a.shape == (1, 128)
    assert (a == b).all()  # reproducible across calls (hashlib, not salted hash())


def test_retrieve_self_match() -> None:
    # Querying with a document's own text must retrieve that document first
    # (self-similarity = 1.0 is the maximum cosine score).
    for doc in DOCS:
        assert retrieve(DOCS, doc, k=1) == [doc]


def test_retrieve_respects_k() -> None:
    assert len(retrieve(DOCS, "vector search", k=3)) == 3


def test_rag_answer_grounds_in_retrieved_context() -> None:
    answer = rag_answer(DOCS, "how do I ground an LLM in documents?")
    assert isinstance(answer, str) and answer
    assert "grounded" in answer
