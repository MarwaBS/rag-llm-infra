"""The shipped version is declared once in effect, and it is documented.

`pyproject.toml` and `__init__.py` each carry the version as a literal. The
release workflow compares the tag against `__version__` alone; the wheel takes
its version from `pyproject.toml`. Without these two assertions the two literals
can ship a release apart.
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


def test_every_documented_version_carries_a_compare_link() -> None:
    """A section without one leaves a reader no way to see what changed."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## \[([^\]]+)\]", text, re.M)
    linked = set(re.findall(r"^\[([^\]]+)\]: https", text, re.M))
    assert sections
    assert not [s for s in sections if s not in linked]


def test_the_changelog_has_a_section_for_the_version_being_shipped() -> None:
    headings = re.findall(
        r"^## \[([^\]]+)\]", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M
    )
    assert _declared() in headings, f"no CHANGELOG section for {_declared()}"
