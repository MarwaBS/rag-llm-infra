"""The mutation registry must stay applicable to the code it targets.

Replaying it is a separate CI step (`python -m scripts.replay_mutations`) because
each entry runs the suite. What is cheap enough to run here is the part that rots
silently. An anchor that no longer matches makes an entry unappliable, and a
registry of unappliable entries replays green while testing nothing.
"""

import ast
import json
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
        "M31", "M32", "M33", "M34", "M35", "M36", "M37", "M38", "M39",
        "M40", "M41", "M42", "M43", "M44", "M45", "M46", "M47", "M48",
        "M49", "M50", "M51", "M52", "M53", "M54", "M55", "M56", "M57", "M58",
        "M59", "M60", "M61", "M62", "M63", "M64", "M65", "M66", "M67", "M68",
        "M69", "M70", "M71", "M72", "M73", "M74", "M75", "M76", "M77", "M78",
        "M79", "M80", "M81", "M82", "M83",
        "C1",
    }
)  # fmt: skip


def test_the_registry_holds_exactly_these_defects() -> None:
    """The replay reports PASS over whatever it is handed, so a registry that
    loses entries reports the same green as one that caught them all."""
    assert {entry["id"] for entry in REGISTRY} == EXPECTED_IDS


def test_every_id_is_unique() -> None:
    ids = [entry["id"] for entry in REGISTRY]
    assert len(set(ids)) == len(ids)


CONTROL = {
    "id": "C1",
    "file": "tests/test_demo_embedder.py",
    "find": '("0", "1", "12345")',
    "replace": '("0", "1", "31337")',
}


def _scoring_modules() -> list[str]:
    """Every runnable module in `eval/`.

    The gates that score this repo are `pytest` and the eval modules. Reading the
    package is structural, so a new eval module is in scope the moment it exists,
    whichever workflow ends up calling it. Bounded: a scoring gate that is not an
    eval module, some future `scripts/score_*.py`, is outside this check.
    """
    return sorted(
        path.stem
        for path in (REPO / "eval").glob("*.py")
        if not path.name.startswith("_")
        and "\ndef main(" in path.read_text(encoding="utf-8")
    )


def test_every_eval_gate_carries_a_defect() -> None:
    modules = _scoring_modules()
    assert modules, "no runnable eval module found"
    covered = {gate for entry in REGISTRY for gate in entry["gates"]}
    missing = [m for m in modules if f"python -m eval.{m}" not in covered]
    assert not missing, f"no registered defect for {missing}"


def test_the_suite_carries_defects() -> None:
    assert any(g.startswith("pytest") for e in REGISTRY for g in e["gates"])


def test_the_replay_carries_defects_of_its_own() -> None:
    """It decides whether every other gate went red, so a corruption of its
    verdict certifies the whole registry without running any of it."""
    assert any(entry["file"] == "scripts/replay_mutations.py" for entry in REGISTRY)


def test_the_only_control_is_this_one() -> None:
    """A control is declared, not detected. Survival is what a control and an
    unheld guard have in common. Pinning it whole is what stops a real guard
    being parked in the registry and reported as intended behaviour."""
    controls = [entry for entry in REGISTRY if entry.get("expect") == "survives"]
    assert len(controls) == 1
    assert {key: controls[0][key] for key in CONTROL} == CONTROL


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
