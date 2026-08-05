"""The generation gate must pass, and must be able to fail.

The fixtures are the part that can quietly stop working: an answer copied out of
its own evidence scores 1.0 by set identity, and one sharing no vocabulary scores
0.0 the same way. Either sits at a theoretical extreme, where a metric that
discriminates and a metric that has stopped produce the same number. The first
three tests pin the fixtures away from those extremes; the rest substitute
metrics that do not discriminate and require the gate to go red.
"""

import eval.generation_eval as g
from eval.generation_eval import (
    GROUNDED_MIN,
    HALLUCINATED_MAX,
    MARGIN_MIN,
    evaluate,
    main,
)

from rag_llm_infra import groundedness

FIXTURES = g.FAITHFUL_ANSWERS + g.HALLUCINATED_ANSWERS


def test_no_fixture_answer_is_lifted_from_its_own_evidence() -> None:
    contexts = g.retrieve(g.QUERY)
    for answer in FIXTURES:
        for context in contexts:
            assert answer not in context, answer
            assert context not in answer, answer


def test_no_fixture_answer_scores_a_theoretical_extreme() -> None:
    contexts = g.retrieve(g.QUERY)
    for answer in FIXTURES:
        assert 0.0 < groundedness(answer, contexts) < 1.0, answer


def test_the_gate_holds_both_populations_at_their_worst() -> None:
    contexts = g.retrieve(g.QUERY)
    m = evaluate()
    assert m["grounded"] == min(groundedness(a, contexts) for a in g.FAITHFUL_ANSWERS)
    assert m["hallucinated"] == max(
        groundedness(a, contexts) for a in g.HALLUCINATED_ANSWERS
    )


def test_generation_gate_two_sided() -> None:
    m = evaluate()
    assert m["grounded"] >= GROUNDED_MIN, m
    assert m["hallucinated"] <= HALLUCINATED_MAX, m
    assert m["margin"] >= MARGIN_MIN, m


def test_gate_passes_end_to_end() -> None:
    assert main() == 0


def test_gate_fails_a_metric_that_flags_nothing(monkeypatch) -> None:
    """Lifting every score by 0.33 leaves an unsupported answer scoring 0.33 —
    a metric that flags nothing. The gate must not pass it."""
    real = g.groundedness
    monkeypatch.setattr(g, "groundedness", lambda a, c: min(1.0, real(a, c) + 0.33))
    assert main() == 1


def test_gate_fails_if_metric_collapses(monkeypatch) -> None:
    """A metric that returns the same number for fact and fiction must trip it."""
    monkeypatch.setattr(g, "groundedness", lambda a, c: 1.0)
    assert main() == 1
