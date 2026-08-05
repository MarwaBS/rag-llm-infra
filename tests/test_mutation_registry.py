"""The mutation registry must stay applicable to the code it targets.

Replaying it is a separate CI step (`python -m scripts.replay_mutations`) because
each entry runs the suite. What is cheap enough to run here is the part that rots
silently: an anchor that no longer matches makes an entry unappliable, and a
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


def test_the_registry_is_not_empty() -> None:
    assert REGISTRY


def test_every_id_is_unique() -> None:
    ids = [entry["id"] for entry in REGISTRY]
    assert len(set(ids)) == len(ids)


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
