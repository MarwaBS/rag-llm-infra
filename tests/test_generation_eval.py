"""The generation (faithfulness) eval must pass its two-sided gate — and the gate
must be able to FAIL, which is the whole point: the previous version was a
tautology (grounded == join(contexts), hallucinated == grounded + extra) that
could never fail on a real metric."""

import eval.generation_eval as g
from eval.generation_eval import (
    GROUNDED_MIN,
    HALLUCINATED_MAX,
    MARGIN_MIN,
    evaluate,
    main,
)


def test_generation_gate_two_sided() -> None:
    m = evaluate()
    assert m["grounded"] >= GROUNDED_MIN, m
    assert m["hallucinated"] <= HALLUCINATED_MAX, (
        m
    )  # absolute ceiling, not just a margin
    assert m["margin"] >= MARGIN_MIN, m


def test_gate_passes_end_to_end() -> None:
    assert main() == 0


def test_gate_fails_on_high_scoring_hallucination(monkeypatch) -> None:
    """A metric that scores a hallucination 0.80 still has a positive margin vs a
    0.95 grounded score — the OLD margin>0 gate passed it. The two-sided gate
    MUST fail it because 0.80 exceeds the hallucinated ceiling. This proves the
    gate genuinely discriminates rather than being a tautology."""
    scores = iter([0.95, 0.80])  # evaluate() scores grounded first, then hallucinated
    monkeypatch.setattr(g, "groundedness", lambda answer, contexts: next(scores))
    assert main() == 1


def test_gate_fails_if_metric_collapses(monkeypatch) -> None:
    """A metric that returns 1.0 for everything (can't tell fact from fiction)
    must trip the gate."""
    monkeypatch.setattr(g, "groundedness", lambda answer, contexts: 1.0)
    assert main() == 1
