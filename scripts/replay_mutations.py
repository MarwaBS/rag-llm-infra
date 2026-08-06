"""Replay every defect in `tests/mutations.json` and require each to be caught.

    python -m scripts.replay_mutations

One at a time: apply the mutation, run its gates, restore the file's original
bytes. A mutation that leaves every gate green is a guard nobody is holding, so
this exits non-zero and names it.

Serial by construction — two mutations live in one working tree would each
report the other's result.

Entries marked `"expect": "survives"` are controls. They change nothing any gate
can see, so reporting one as caught means the gates are not being run. That is
the half of the verdict the survivor list cannot reach: a runner stuck on "green"
fails on the defects, a runner stuck on "red" fails on the controls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "tests" / "mutations.json"
TIMEOUT_S = 300

PASSED, FAILED, ERRORED, TIMEOUT = "passed", "failed", "errored", "timeout"


def verdict(gate: str) -> str:
    """How one gate command ended.

    pytest exits 1 when a test failed and 2, 3 or 4 when collection, an internal
    error or a bad invocation stopped it. Only the first is a guard doing its
    job, so the others are `errored` and earn no credit.
    """
    argv = gate.split()
    if argv[0] == "pytest":
        argv = [sys.executable, "-m", "pytest", "-q", "--no-header", *argv[1:]]
    elif argv[0] == "python":
        argv = [sys.executable, *argv[1:]]
    try:
        done = subprocess.run(
            argv, cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT
    if done.returncode == 0:
        return PASSED
    return FAILED if done.returncode == 1 else ERRORED


def imports_cleanly(rel: str) -> bool:
    """Whether the mutated file still loads.

    A mutation that breaks the import reddens every gate that touches it without
    removing any behaviour, so it would be credited for a guard it never tested.
    """
    path = REPO / rel
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return True
    if path.suffix != ".py":
        return True
    module = ".".join(path.relative_to(REPO).with_suffix("").parts)
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    return done.returncode == 0


def replay(entry: dict) -> tuple[str, str | None]:
    """Apply one entry, run its gates, restore. Returns (outcome, gate)."""
    path = REPO / entry["file"]
    original = path.read_bytes()
    # Universal newlines, so an anchor spanning lines matches the same way here
    # as in tests/test_mutation_registry.py on a CRLF checkout. The file is
    # restored from its original bytes either way.
    text = path.read_text(encoding="utf-8")
    if text.count(entry["find"]) != 1:
        return "stale", None
    try:
        path.write_text(
            text.replace(entry["find"], entry["replace"], 1),
            encoding="utf-8",
            newline="\n",
        )
        if not imports_cleanly(entry["file"]):
            return "unloadable", None
        for gate in entry["gates"]:
            if verdict(gate) == FAILED:
                return "caught", gate
    finally:
        path.write_bytes(original)
    return "survived", None


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["mutations"]
    survivors: list[str] = []
    controls_caught: list[str] = []
    caught = 0
    for entry in registry:
        outcome, gate = replay(entry)
        expected_to_survive = entry.get("expect") == "survives"
        if outcome in ("stale", "unloadable"):
            print(f"{entry['id']}: {outcome} against {entry['file']}")
            return 2
        if outcome == "caught" and expected_to_survive:
            controls_caught.append(entry["id"])
            print(
                f"  CONTROL CAUGHT {entry['id']}  by `{gate}` — the gates are not running"
            )
        elif outcome == "caught":
            caught += 1
            print(f"  CAUGHT   {entry['id']}  by `{gate}`  — {entry['why']}")
        elif expected_to_survive:
            print(f"  CONTROL  {entry['id']}  survived, as it must  — {entry['why']}")
        else:
            survivors.append(entry["id"])
            print(f"  SURVIVED {entry['id']}  — {entry['why']}")

    defects = [e for e in registry if e.get("expect") != "survives"]
    print(f"\n{caught}/{len(defects)} caught, {len(registry) - len(defects)} controls")
    if controls_caught:
        print(f"FAIL: control {', '.join(controls_caught)} reported caught")
        return 1
    if survivors:
        print(f"FAIL: nothing holds {', '.join(survivors)}")
        return 1
    print("PASS: every registered defect turns a gate red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
