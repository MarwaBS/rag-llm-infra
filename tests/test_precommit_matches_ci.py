"""The pre-commit hooks run what CI runs, and the pinned rev satisfies the floor.

A config carrying `ruff` but not `ruff-format` lets a clean pre-commit pass fail
CI's `ruff format --check`, which is worse than having no hook: it teaches the
contributor that local green means anything.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
CI = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _ruff_repo() -> dict:
    matches = [r for r in CONFIG["repos"] if "ruff-pre-commit" in r["repo"]]
    assert len(matches) == 1, CONFIG["repos"]
    return matches[0]


def _ci_commands() -> list[str]:
    return [
        line.strip()
        for job in CI["jobs"].values()
        for step in job.get("steps", [])
        for line in (step.get("run") or "").splitlines()
        if line.strip()
    ]


def test_every_ruff_command_ci_runs_has_a_hook() -> None:
    hooks = {h["id"] for h in _ruff_repo()["hooks"]}
    needed = set()
    for command in _ci_commands():
        if command.startswith("ruff format"):
            needed.add("ruff-format")
        elif command.startswith("ruff "):
            needed.add("ruff")
    assert needed, "no ruff command found in CI"
    assert needed <= hooks, f"CI runs {sorted(needed - hooks)} with no hook"


def test_the_pinned_hook_satisfies_the_declared_floor() -> None:
    """A rev below the floor runs different rules than the gate does."""
    dev = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]["dev"]
    floor = next(Requirement(r) for r in dev if Requirement(r).name == "ruff")
    pinned = Version(_ruff_repo()["rev"].lstrip("v"))
    assert floor.specifier.contains(str(pinned)), f"rev {pinned} vs {floor}"
