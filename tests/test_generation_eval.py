"""The generation (faithfulness) eval must pass its two-sided gate — and the gate
must be able to FAIL, which the last two tests prove by substituting a metric
that does not discriminate."""

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
    """A metric scoring the hallucination 0.80 against a 0.95 grounded answer
    clears the floor and keeps a positive margin. The gate must still reject it:
    0.80 is above the ceiling and the 0.15 margin is below the minimum."""
    scores = iter([0.95, 0.80])  # evaluate() scores grounded first, then hallucinated
    monkeypatch.setattr(g, "groundedness", lambda answer, contexts: next(scores))
    assert main() == 1


def test_gate_fails_if_metric_collapses(monkeypatch) -> None:
    """A metric that returns 1.0 for everything (can't tell fact from fiction)
    must trip the gate."""
    monkeypatch.setattr(g, "groundedness", lambda answer, contexts: 1.0)
    assert main() == 1
