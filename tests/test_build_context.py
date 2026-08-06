"""The image builds from the admitted files, and only those.

`COPY . .` copies the build context, so what `.dockerignore` admits is what
reaches the builder stage. This assembles a directory from the admitted paths
alone and builds the wheel in it. Too tight and the build fails here rather than
in the image; too loose and the excluded paths ship.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RULES = (REPO / ".dockerignore").read_text(encoding="utf-8")

ADMITTED = [line[1:] for line in RULES.splitlines() if line.startswith("!")]

EXCLUDED = [
    ".venv/pyvenv.cfg",
    ".git/HEAD",
    "dist/anything.whl",
    "tests/test_build_context.py",
    ".env",
    "benchmarks/topk_tie_cost.py",
]


def _admitted(relative: str) -> bool:
    """`*` denies everything; a `!` line re-admits a path or a directory."""
    assert RULES.splitlines()[3] == "*", "the rules must start from deny-everything"
    return any(relative == rule or relative.startswith(rule) for rule in ADMITTED)


@pytest.mark.parametrize("relative", EXCLUDED)
def test_the_context_excludes_what_must_not_reach_the_image(relative: str) -> None:
    assert not _admitted(relative)


@pytest.mark.parametrize("relative", ["pyproject.toml", "README.md", "LICENSE"])
def test_every_admitted_path_exists(relative: str) -> None:
    assert (REPO / relative).exists()


def test_the_package_sources_are_admitted() -> None:
    assert _admitted("src/rag_llm_infra/__init__.py")


@pytest.mark.slow
def test_the_admitted_files_alone_build_the_wheel(tmp_path: Path) -> None:
    """The image installs the project from the context. If the context is short
    of a file the build backend needs, the image fails to build, not the suite."""
    for rule in ADMITTED:
        source = REPO / rule.rstrip("/")
        target = tmp_path / rule.rstrip("/")
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copy2(source, target)

    done = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path / "out")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert list((tmp_path / "out").glob("*.whl")), done.stdout[-800:]
