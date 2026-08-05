"""No eval floor is a number somebody picked, and none of them is stale.

Re-running the producer must reproduce the committed artefact byte for byte, so a
fixture edit that moves a measurement cannot leave the floors behind it.
"""

import json
from pathlib import Path

import pytest
from eval import generation_eval, retrieval_eval
from scripts import derive_eval_floors

ARTEFACT = Path(__file__).resolve().parent.parent / "eval" / "eval_floors.json"


def test_rerunning_the_producer_reproduces_the_committed_artefact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fresh = tmp_path / "eval_floors.json"
    monkeypatch.setattr(derive_eval_floors, "ARTEFACT", fresh)
    assert derive_eval_floors.main() == 0
    assert fresh.read_bytes() == ARTEFACT.read_bytes()


def test_every_floor_the_gates_use_comes_from_the_artefact() -> None:
    committed = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert retrieval_eval.THRESHOLDS == committed["retrieval"]["floors"]
    generation = committed["generation"]["floors"]
    assert generation_eval.GROUNDED_MIN == generation["grounded_min"]
    assert generation_eval.HALLUCINATED_MAX == generation["hallucinated_max"]
    assert generation_eval.MARGIN_MIN == generation["margin_min"]


def test_each_derivation_rule_is_recorded_beside_the_numbers() -> None:
    committed = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    for section in ("generation", "retrieval"):
        assert committed[section]["rule"].strip()
        assert committed[section]["measured"]
