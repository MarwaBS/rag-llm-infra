"""The producer refuses to derive floors from a metric that does not separate.

Floors derived from the metric they gate cannot also police it: distort the
metric, re-derive, and the floors move out of its own way. The 0.5 anchor is the
midpoint of groundedness's documented [0,1] range, so it does not move when the
metric does. Nothing else in the suite executes this branch.
"""

from __future__ import annotations

import pytest
from eval import generation_eval
from scripts import derive_eval_floors


def test_it_refuses_when_the_populations_do_not_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A faithful answer below 0.5 puts both populations on one side of it."""
    monkeypatch.setattr(
        generation_eval,
        "FAITHFUL_ANSWERS",
        [*generation_eval.FAITHFUL_ANSWERS, "cartography of the abyssal seafloor"],
    )
    with pytest.raises(SystemExit) as raised:
        derive_eval_floors._generation()
    assert "refusing to derive" in str(raised.value)


def test_the_control_shows_it_derives_on_the_shipped_fixtures() -> None:
    floors = derive_eval_floors._generation()["floors"]
    assert 0.0 < floors["grounded_min"] < 1.0
