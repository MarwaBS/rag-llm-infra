"""The mutation registry must stay applicable to the code it targets.

Replaying it is a separate CI step (`python -m scripts.replay_mutations`) because
each entry runs the suite. What is cheap enough to run here is the part that rots
silently: an anchor that no longer matches makes an entry unappliable, and a
registry of unappliable entries replays green while testing nothing.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REGISTRY = json.loads((REPO / "tests" / "mutations.json").read_text(encoding="utf-8"))[
    "mutations"
]

EXPECTED_IDS = frozenset(
    {
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
        "M11", "M12", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20",
        "M21", "M22", "M23", "M24", "M25", "M26", "M27", "M28", "M29", "M30",
        "M31", "M32", "M33",
    }
)  # fmt: skip


def test_the_registry_holds_exactly_these_defects() -> None:
    """The replay reports PASS over whatever it is handed, so a registry that
    loses entries reports the same green as one that caught them all."""
    assert {entry["id"] for entry in REGISTRY} == EXPECTED_IDS


def test_every_id_is_unique() -> None:
    ids = [entry["id"] for entry in REGISTRY]
    assert len(set(ids)) == len(ids)


def _ci_gate_commands() -> list[str]:
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    return [
        m.group(1).strip() for m in re.finditer(r"^ +run: (?!\|)(.+)$", workflow, re.M)
    ]


def _scores_this_repo(command: str) -> bool:
    return command.startswith(("pytest", "python -m eval."))


def _gate_key(command: str) -> str:
    return "pytest" if command.startswith("pytest") else command


def test_every_ci_gate_that_scores_this_repo_carries_a_defect() -> None:
    """ruff, mypy and `python -m build` are third-party tools whose failure modes
    are theirs, and `example.py` exits on an exception rather than on a score.
    The rest read a number out of this code, so each one can be shown to go red.
    """
    commands = _ci_gate_commands()
    assert commands, "no gate step parsed out of ci.yml"
    required = {_gate_key(c) for c in commands if _scores_this_repo(c)}
    assert required, "no scoring gate parsed out of ci.yml"
    covered = {_gate_key(gate) for entry in REGISTRY for gate in entry["gates"]}
    assert required <= covered, f"no registered defect for {sorted(required - covered)}"


@pytest.mark.parametrize("entry", REGISTRY, ids=lambda e: e["id"])
def test_the_anchor_matches_exactly_once_and_the_mutation_changes_something(
    entry: dict,
) -> None:
    text = (REPO / entry["file"]).read_text(encoding="utf-8")
    assert text.count(entry["find"]) == 1, entry["file"]
    assert entry["replace"] != entry["find"]
    assert entry["gates"]
    assert entry["why"].strip()


@pytest.mark.parametrize("entry", REGISTRY, ids=lambda e: e["id"])
def test_the_mutated_file_still_parses(entry: dict) -> None:
    """The replay decides on exit code, so a mutation that does not parse looks
    caught while testing nothing. Every entry must produce runnable code."""
    path = REPO / entry["file"]
    mutated = path.read_text(encoding="utf-8").replace(
        entry["find"], entry["replace"], 1
    )
    if path.suffix == ".py":
        ast.parse(mutated)
    elif path.suffix == ".json":
        json.loads(mutated)
