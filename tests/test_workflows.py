"""What the workflows must contain to be the gates the docs claim.

Bounded: this reads the YAML, it does not run GitHub Actions. It catches a
deleted step, a drifted tag, a discarded exit code and a lowered floor; it cannot
tell you the container job passes. That job has no local equivalent here.
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
COVERAGE_FLOOR = 85
NO_OPS = {"true", ":", "exit 0"}


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
def test_no_gate_discards_its_exit_code(path: Path) -> None:
    """A step that cannot fail is a log line. `|| true`, `|| :` and a trailing
    `exit 0` all do that, so the rule is about the no-op, not one spelling."""
    swallowed = []
    for step in _steps(path):
        if step.get("name") == "Stop it":  # cleanup, deliberately best-effort
            continue
        lines = [
            ln.strip() for ln in (step.get("run") or "").splitlines() if ln.strip()
        ]
        for line in lines:
            if "||" in line and line.split("||")[-1].strip().rstrip(";") in NO_OPS:
                swallowed.append((step.get("name"), line))
        if lines and lines[-1] in NO_OPS:
            swallowed.append((step.get("name"), lines[-1]))
    assert not swallowed, swallowed
    assert not [s.get("name") for s in _steps(path) if s.get("continue-on-error")]


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


def test_the_image_job_runs_the_image_it_just_built() -> None:
    tag = "rag-llm-infra:ci"
    assert tag in _named(CI, "Build the image")
    started = _named(CI, "Start it")
    assert "docker run -d" in started and tag in started


def test_the_image_job_proves_the_credential_is_enforced() -> None:
    step = _named(CI, "The running image enforces the credential")
    assert "curl" in step and "/index" in step
    assert _invokes(step, "test"), "the status code is captured but never compared"
    assert '"401"' in step


def test_the_image_job_exercises_a_real_query() -> None:
    step = _named(CI, "The running image answers a real query")
    assert "/index" in step and "/query" in step
    assert "grep -q" in step, "the response is fetched but never checked"


def test_the_image_job_waits_for_the_containers_own_healthcheck() -> None:
    assert "State.Health.Status" in _named(CI, "The container reports itself healthy")
