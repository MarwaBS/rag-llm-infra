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


def test_two_char_acronyms_are_groundable() -> None:
    # Regression: len > 2 dropped US/AI/ML, so an answer made only of acronyms had
    # zero content tokens and scored a vacuous 1.0 regardless of the evidence. With
    # len >= 2 the acronyms are scored: supported -> 1.0, unsupported -> < 1.0.
    assert groundedness("AI ML", ["the AI and ML pipeline"]) == 1.0
    assert groundedness("AI ML", ["the cooking recipe"]) == 0.0


def test_known_blind_spots_are_documented_not_hidden() -> None:
    # These assert the LIMITATIONS the docstring is honest about, so the metric's
    # behaviour can't silently change without this test noticing. A lexical
    # bag-of-words proxy cannot catch negation or dilution; do not let a future
    # edit quietly claim it can.
    # Negation-blind: flipping polarity does not change the token overlap.
    assert groundedness("Paris is not the capital", ["Paris is the capital."]) == 1.0
    # Dilution: a fabricated clause appended to a well-supported answer only dents
    # the score (one false token out of many true ones), it does not sink it.
    ctx = ["The Eiffel Tower is a wrought iron lattice tower located in Paris."]
    diluted = groundedness(
        "The Eiffel Tower is a wrought iron lattice tower in Paris and on Mars", ctx
    )
    assert diluted > 0.8  # 6 of 7 content tokens grounded; only "mars" is false
