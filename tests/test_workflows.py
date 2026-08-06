"""What the workflows must contain to be the gates the docs claim.

Bounded: this reads the YAML, it does not run GitHub Actions. It catches a step
that was deleted or a tag that drifted back; it cannot tell you the container
job passes. That is the one job with no local equivalent on this machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted(
    (Path(__file__).resolve().parent.parent / ".github/workflows").glob("*.yml")
)
SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _steps(path: Path) -> list[dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [s for job in document["jobs"].values() for s in job.get("steps", [])]


def _commands(path: Path) -> str:
    return "\n".join(s.get("run") or "" for s in _steps(path))


def test_there_are_workflows_to_check() -> None:
    assert {p.name for p in WORKFLOWS} == {"ci.yml", "release.yml"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    """A tag can be repointed at other code; a commit cannot."""
    floating = [
        s["uses"] for s in _steps(path) if "uses" in s and not SHA.match(s["uses"])
    ]
    assert not floating, floating


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_dependency_cve_gate_runs_and_is_strict(path: Path) -> None:
    """Without --strict, pip-audit exits 0 when a dependency cannot be resolved."""
    commands = _commands(path)
    assert "pip-audit" in commands
    assert "--strict" in commands


def test_the_image_is_started_and_exercised_not_only_built() -> None:
    """An image that is built and never run is the defect this job exists for."""
    commands = _commands(
        Path(__file__).resolve().parent.parent / ".github/workflows/ci.yml"
    )
    assert "docker build" in commands
    assert "docker run -d" in commands
    assert "/health" in commands
    assert "/index" in commands and "/query" in commands
    assert "State.Health.Status" in commands
