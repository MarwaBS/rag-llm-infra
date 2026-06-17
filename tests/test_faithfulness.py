"""Unit tests for the groundedness (faithfulness) metric."""

from rag_llm_infra import groundedness


def test_fully_grounded_answer_scores_one() -> None:
    ctx = ["FAISS performs vector similarity search"]
    assert groundedness("vector similarity search", ctx) == 1.0


def test_off_context_claim_lowers_score() -> None:
    ctx = ["FAISS performs vector similarity search"]
    score = groundedness("vector similarity search using bitcoin quantum", ctx)
    assert 0.0 < score < 1.0


def test_empty_answer_is_trivially_grounded() -> None:
    assert groundedness("", ["anything"]) == 1.0


def test_claims_with_no_context_score_zero() -> None:
    assert groundedness("unsupported novel claim", []) == 0.0


def test_result_is_bounded() -> None:
    s = groundedness("some partially supported vector claim", ["vector store"])
    assert 0.0 <= s <= 1.0
