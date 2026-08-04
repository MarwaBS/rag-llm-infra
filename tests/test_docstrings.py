"""An import a reader can copy out of a docstring must actually import.

Stated as a property over every docstring in the package, not as the lines that
happen to be wrong today.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "rag_llm_infra"


def _docstrings() -> list[tuple[str, str]]:
    """Every docstring in the package, tagged with the file it came from."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                doc = ast.get_docstring(node)
                if doc:
                    found.append((path.name, doc))
    return found


def _quoted_imports() -> list[tuple[str, str]]:
    """`from X import Y` / `import X` lines quoted inside docstrings."""
    pattern = re.compile(
        r"^\s*(?:from\s+[\w.]+\s+import\s+[\w, ]+|import\s+[\w.]+)\s*$"
    )
    return [
        (filename, line.strip())
        for filename, doc in _docstrings()
        for line in doc.splitlines()
        if pattern.match(line)
    ]


def test_the_package_quotes_at_least_one_import() -> None:
    # Without this the parametrized test below would pass on an empty set.
    assert _quoted_imports(), "no import lines found to check — the matcher is dead"


@pytest.mark.parametrize("filename,statement", _quoted_imports())
def test_every_import_shown_in_a_docstring_resolves(
    filename: str, statement: str
) -> None:
    module = (
        statement.split()[1]
        if statement.startswith("from ")
        else statement.split()[1].split(".")[0]
    )
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(f"{filename} shows `{statement}`, which does not import: {exc}")
