"""The shipped version is declared once in effect, and it is documented.

`pyproject.toml` and `__init__.py` each carry the version as a literal, and the
release workflow only compares the tag against `pyproject.toml`. Nothing tied the
two together, so twenty commits of public-surface change built a second, different
`0.1.2` under a version already on PyPI.
"""

import re
import tomllib
from pathlib import Path

import rag_llm_infra

ROOT = Path(__file__).resolve().parent.parent


def _declared() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def test_both_declarations_of_the_version_agree() -> None:
    assert rag_llm_infra.__version__ == _declared()


def test_the_changelog_has_a_section_for_the_version_being_shipped() -> None:
    headings = re.findall(
        r"^## \[([^\]]+)\]", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M
    )
    assert _declared() in headings, f"no CHANGELOG section for {_declared()}"
