"""The declared floors and bounds, held where they are declared.

`pip-audit` reads the installed environment, so it cannot see a floor deleted
from `pyproject.toml` — it only notices once a resolve happens to land below it.
The size constants have the mirrored problem: the tests that exercise them derive
their payloads from the constant, so any value passes.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from rag_llm_infra.serve import DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_CORPUS_DOCS

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]

# Package -> (floor, the advisory that set it, the extra that pulls it in).
SECURITY_FLOORS = {
    "starlette": ("1.3.1", "PYSEC-2026-248/-249", "serve"),
    "h2": ("4.4.1", "CVE-2026-71554", "qdrant"),
}


def _specifier(extra: str, package: str) -> str | None:
    for requirement in PROJECT["optional-dependencies"][extra]:
        if re.match(rf"^{package}\b", requirement):
            return requirement
    return None


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_security_floor_is_declared_where_the_extra_pulls_it(package: str) -> None:
    floor, advisory, extra = SECURITY_FLOORS[package]
    declared = _specifier(extra, package)
    assert declared, f"{package} floor gone from [{extra}] ({advisory})"
    assert f">={floor}" in declared, f"{declared!r} is below the {advisory} fix"


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_dev_extra_carries_the_same_floor(package: str) -> None:
    """CI installs `[dev]`, so a floor only in `[serve]` is not what CI audits."""
    floor, advisory, _ = SECURITY_FLOORS[package]
    declared = _specifier("dev", package)
    assert declared and f">={floor}" in declared, (
        f"{package}: {declared!r} ({advisory})"
    )


def test_the_body_bound_is_the_documented_one() -> None:
    """`1 MiB` is written in the README, SECURITY.md and the module docstring."""
    assert DEFAULT_MAX_BODY_BYTES == 1024 * 1024


def test_the_corpus_bound_is_the_documented_one() -> None:
    assert DEFAULT_MAX_CORPUS_DOCS == 20_000


@pytest.mark.parametrize("document", ["README.md", "SECURITY.md"])
def test_the_documents_state_the_bound_the_code_uses(document: str) -> None:
    mib = DEFAULT_MAX_BODY_BYTES // (1024 * 1024)
    assert f"{mib} MiB" in (ROOT / document).read_text(encoding="utf-8")
