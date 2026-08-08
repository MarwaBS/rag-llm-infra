"""What the workflows must contain to be the gates the docs claim.

Bounded: this reads the YAML, it does not run GitHub Actions. It catches a
deleted step, a drifted tag and a dropped floor. It says nothing about the
container job, which only CI can execute.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github/workflows/ci.yml"
# Actions runs both extensions, so checking only one leaves the other ungated.
WORKFLOWS = sorted(
    p for p in (ROOT / ".github/workflows").iterdir() if p.suffix in {".yml", ".yaml"}
)
EXPECTED_WORKFLOWS = {"ci.yml", "release.yml"}
SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")
COVERAGE_FLOOR = 90


def _steps(path: Path) -> list[dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [s for job in document["jobs"].values() for s in job.get("steps", [])]


def _named(path: Path, name: str) -> str:
    """The commands of the step with this name. Missing the step is the defect."""
    matches = [s for s in _steps(path) if s.get("name") == name]
    assert len(matches) == 1, f"{path.name}: {len(matches)} steps named {name!r}"
    return matches[0].get("run") or ""


def _invokes(run: str, program: str) -> bool:
    """Whether some line actually starts by running `program`.

    Containment credits `echo "would run pip-audit --strict"`, which is a label,
    not an invocation.
    """
    return any(
        line.strip().split(" ")[0] == program
        for line in run.splitlines()
        if line.strip()
    )


def test_the_workflow_set_is_exactly_these_files() -> None:
    """A `.yaml` sibling would run in CI and be checked by nothing here."""
    assert {p.name for p in WORKFLOWS} == EXPECTED_WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    """A tag can be repointed at other code; a commit cannot."""
    floating = [
        s["uses"] for s in _steps(path) if "uses" in s and not SHA.match(s["uses"])
    ]
    assert not floating, floating


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_coverage_floor_is_not_lowered(path: Path) -> None:
    floors = [
        int(m)
        for s in _steps(path)
        for m in re.findall(r"--cov-fail-under=(\d+)", s.get("run") or "")
    ]
    assert floors, f"{path.name}: no coverage floor found"
    assert all(f >= COVERAGE_FLOOR for f in floors), floors


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_dependency_cve_gate_runs_and_is_strict(path: Path) -> None:
    """Without --strict, pip-audit exits 0 when a dependency cannot be resolved."""
    run = _named(path, "Dependency CVE gate (pip-audit)")
    assert _invokes(run, "pip-audit"), run
    assert "--strict" in run


@pytest.mark.parametrize("module", ["eval.retrieval_eval", "eval.generation_eval"])
def test_ci_invokes_both_eval_gates(module: str) -> None:
    """Neither eval threshold has a unit test behind it, so deleting the step
    removes the only thing enforcing it. Matched on the invocation rather than
    the step's name, which a rename would otherwise defeat."""
    invoked = any(
        line.split()[:3] == ["python", "-m", module]
        for step in _steps(CI)
        for line in (step.get("run") or "").splitlines()
        if line.split()
    )
    assert invoked, module
