"""The retrieval eval must meet its thresholds — this is the CI quality gate."""

from eval.retrieval_eval import THRESHOLDS, evaluate


def test_retrieval_meets_thresholds() -> None:
    m = evaluate()
    assert m["recall@1"] >= THRESHOLDS["recall@1"], m
    assert m["mrr"] >= THRESHOLDS["mrr"], m


def test_metrics_are_bounded() -> None:
    m = evaluate()
    assert 0.0 <= m["recall@1"] <= 1.0
    assert 0.0 <= m["mrr"] <= 1.0
