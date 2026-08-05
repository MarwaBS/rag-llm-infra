"""Replay every defect in `tests/mutations.json` and require each to be caught.

    python -m scripts.replay_mutations

One at a time: apply the mutation, run its gates, restore the file's original
bytes. A mutation that leaves every gate green is a guard nobody is holding, so
this exits non-zero and names it.

Serial by construction — two mutations live in one working tree would each
report the other's result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "tests" / "mutations.json"
TIMEOUT_S = 300


def _run(gate: str) -> bool:
    """True when the gate passes."""
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
        return True  # no verdict inside the cap is not a red gate
    return done.returncode == 0


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["mutations"]
    survivors: list[str] = []
    for entry in registry:
        path = REPO / entry["file"]
        original = path.read_bytes()
        # Universal newlines, so an anchor spanning lines matches the same way
        # here as in tests/test_mutation_registry.py on a CRLF checkout. The
        # file is restored from its original bytes either way.
        text = path.read_text(encoding="utf-8")
        if text.count(entry["find"]) != 1:
            print(f"{entry['id']}: anchor is not unique in {entry['file']} — stale")
            return 2
        try:
            path.write_text(
                text.replace(entry["find"], entry["replace"], 1),
                encoding="utf-8",
                newline="\n",
            )
            caught = next((g for g in entry["gates"] if not _run(g)), None)
        finally:
            path.write_bytes(original)
        if caught:
            print(f"  CAUGHT   {entry['id']}  by `{caught}`  — {entry['why']}")
        else:
            survivors.append(entry["id"])
            print(f"  SURVIVED {entry['id']}  — {entry['why']}")

    print(f"\n{len(registry) - len(survivors)}/{len(registry)} caught")
    if survivors:
        print(f"FAIL: nothing holds {', '.join(survivors)}")
        return 1
    print("PASS: every registered defect turns a gate red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
