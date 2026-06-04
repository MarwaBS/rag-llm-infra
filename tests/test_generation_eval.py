"""The generation (faithfulness) eval must pass its gate."""
from eval.generation_eval import GROUNDED_MIN, evaluate


def test_generation_gate() -> None:
    m = evaluate()
    assert m["grounded"] >= GROUNDED_MIN, m
    assert m["margin"] > 0.0, m  # metric must flag the hallucinated control
