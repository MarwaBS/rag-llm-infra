"""The declared floors and bounds, held where they are declared.

`pip-audit` reads the installed environment, so it cannot see a floor deleted
from `pyproject.toml` — it only notices once a resolve happens to land below it.
The size constants have the mirrored problem: the tests that exercise them derive
their payloads from the constant, so any value passes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

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


def _requirement(extra: str, package: str) -> Requirement | None:
    for line in PROJECT["optional-dependencies"][extra]:
        parsed = Requirement(line)
        if canonicalize_name(parsed.name) == canonicalize_name(package):
            return parsed
    return None


def _assert_floor_holds(extra: str, package: str) -> None:
    """Parse the specifier rather than match its text.

    `starlette>=1.3.1; python_version<'3.0'` contains the floor and applies to no
    interpreter this project admits. A marker makes the constraint conditional,
    so any marker at all disqualifies it.
    """
    floor, advisory, _ = SECURITY_FLOORS[package]
    declared = _requirement(extra, package)
    assert declared, f"{package} floor gone from [{extra}] ({advisory})"
    assert declared.marker is None, f"{declared} is conditional ({advisory})"
    assert declared.specifier.contains(floor), f"{declared} excludes the fixed {floor}"
    fixed = Version(floor)
    below = f"{fixed.major}.{fixed.minor}.{max(fixed.micro - 1, 0)}"
    assert not declared.specifier.contains(below), (
        f"{declared} admits {below}, below the {advisory} fix"
    )


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_security_floor_is_declared_where_the_extra_pulls_it(package: str) -> None:
    _assert_floor_holds(SECURITY_FLOORS[package][2], package)


@pytest.mark.parametrize("package", sorted(SECURITY_FLOORS))
def test_the_dev_extra_carries_the_same_floor(package: str) -> None:
    """CI installs `[dev]`, so a floor only in `[serve]` is not what CI audits."""
    _assert_floor_holds("dev", package)


def test_the_body_bound_is_the_documented_one() -> None:
    """`1 MiB` is written in the README, SECURITY.md and the module docstring."""
    assert DEFAULT_MAX_BODY_BYTES == 1024 * 1024


def test_the_corpus_bound_is_the_documented_one() -> None:
    assert DEFAULT_MAX_CORPUS_DOCS == 20_000


@pytest.mark.parametrize("document", ["README.md", "SECURITY.md"])
def test_the_documents_state_the_body_bound_the_code_uses(document: str) -> None:
    mib = DEFAULT_MAX_BODY_BYTES // (1024 * 1024)
    assert f"{mib} MiB" in (ROOT / document).read_text(encoding="utf-8")


@pytest.mark.parametrize("document", ["README.md", "SECURITY.md"])
def test_the_documents_state_the_corpus_bound_the_code_uses(document: str) -> None:
    """An operator who gets a 413 on an 80 KB body needs the reason written
    somewhere. The byte bound was tied to the docs and this one was not."""
    assert str(DEFAULT_MAX_CORPUS_DOCS) in (ROOT / document).read_text(encoding="utf-8")
