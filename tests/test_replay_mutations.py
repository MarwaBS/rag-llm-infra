"""The replay decides whether every other gate can fail, so its verdict is pinned.

Nothing else in the suite reads this module. A runner that reports every gate red
prints the same `PASS` line as one that ran them. The registry it certifies
cannot tell the difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import replay_mutations as replay

GATE_OK = "python gate_ok.py"
GATE_FAILS = "python gate_fails.py"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A miniature repo: one subject file and two gates that read it."""
    (tmp_path / "subject.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "gate_ok.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / "gate_fails.py").write_text(
        "import subject; raise SystemExit(0 if subject.VALUE == 1 else 1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(replay, "REPO", tmp_path)
    monkeypatch.setattr(replay, "REGISTRY", tmp_path / "mutations.json")
    return tmp_path


def _write_registry(root: Path, entries: list[dict]) -> None:
    (root / "mutations.json").write_text(
        json.dumps({"mutations": entries}), encoding="utf-8"
    )


def _entry(**over: object) -> dict:
    base = {
        "id": "T1",
        "why": "the subject's value changes",
        "file": "subject.py",
        "find": "VALUE = 1",
        "replace": "VALUE = 2",
        "gates": [GATE_FAILS],
    }
    return {**base, **over}


def test_a_gate_that_passes_and_one_that_fails_are_told_apart(sandbox: Path) -> None:
    assert replay.gate_caught(GATE_OK) is False
    (sandbox / "subject.py").write_text("VALUE = 9\n", encoding="utf-8")
    assert replay.gate_caught(GATE_FAILS) is True


def test_a_gate_that_could_not_run_earns_no_credit(sandbox: Path) -> None:
    """pytest exits 4 on a bad invocation and 2 on a collection error. Counting
    either as a caught defect credits a guard for a red it never produced."""
    assert replay.gate_caught("pytest --no-such-flag") is False


def test_a_file_that_stops_importing_is_rejected_not_credited(sandbox: Path) -> None:
    assert replay.imports_cleanly("subject.py")
    broken = _entry(replace="VALUE = 1\nraise RuntimeError('boom')")
    _write_registry(sandbox, [broken])
    assert replay.replay(broken)[0] == "unloadable"
    assert replay.main() == 2


def test_a_json_file_that_stops_parsing_is_rejected_too(sandbox: Path) -> None:
    """The registry may target data as well as code. Invalid JSON reddens every
    gate that reads it without removing any behaviour."""
    (sandbox / "data.json").write_text('{"k": 1}\n', encoding="utf-8")
    assert replay.imports_cleanly("data.json")
    broken = _entry(id="J1", file="data.json", find='{"k": 1}', replace='{"k": ')
    _write_registry(sandbox, [broken])
    assert replay.replay(broken)[0] == "unloadable"
    assert replay.main() == 2


def test_a_defect_its_gate_catches_is_reported_caught(sandbox: Path) -> None:
    _write_registry(sandbox, [_entry()])
    assert replay.main() == 0
    assert (sandbox / "subject.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_a_defect_no_gate_catches_fails_the_run(sandbox: Path) -> None:
    _write_registry(sandbox, [_entry(gates=[GATE_OK])])
    assert replay.main() == 1


def test_a_runner_stuck_on_red_is_caught_by_the_control(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The survivor list can only see a runner stuck on green. This is the other
    half: a control changes nothing, so reporting it caught means no gate ran."""
    control = _entry(id="C0", expect="survives", replace="VALUE = 1  # reworded")
    _write_registry(sandbox, [_entry(), control])
    assert replay.main() == 0

    monkeypatch.setattr(replay, "gate_caught", lambda gate: True)
    assert replay.main() == 1


def test_a_runner_stuck_on_green_fails_on_the_defects(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_registry(sandbox, [_entry()])
    monkeypatch.setattr(replay, "gate_caught", lambda gate: False)
    assert replay.main() == 1


def test_the_printed_count_is_the_same_list_the_exit_code_is(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anyone watching CI reads the summary line, not the exit code."""
    _write_registry(sandbox, [_entry(), _entry(id="T2", gates=[GATE_OK])])
    assert replay.main() == 1
    out = capsys.readouterr().out
    assert "1/2 caught" in out
    assert "FAIL: nothing holds T2" in out


def test_a_stale_anchor_stops_the_run(sandbox: Path) -> None:
    _write_registry(sandbox, [_entry(find="NOT PRESENT")])
    assert replay.main() == 2
