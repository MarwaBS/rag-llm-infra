"""A broken optional dependency degrades; it does not kill the import.

A native extension whose runtime is missing raises `OSError` at import, not
`ImportError`. Catching only `ImportError` turns "FAISS is unusable here" into
"this package will not import at all".
"""

from __future__ import annotations

import subprocess
import sys

import pytest

OPTIONAL = [
    "faiss",
    "psutil",
    "qdrant_client",
    "sentence_transformers",
    "opentelemetry",
]

# A meta-path hook that lets the module be found, then fails while loading it.
_BREAK = """
import sys, importlib.abc, importlib.machinery

TARGET = {target!r}

class Breaker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == TARGET or name.startswith(TARGET + "."):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        raise OSError(126, "simulated: shared library not found")

    def exec_module(self, module):
        raise OSError(126, "simulated: shared library not found")

for name in list(sys.modules):
    if name == TARGET or name.startswith(TARGET + "."):
        del sys.modules[name]
sys.meta_path.insert(0, Breaker())

import rag_llm_infra
from rag_llm_infra import get_vector_store, get_llm
store = get_vector_store("auto")
store.add(__import__("numpy").eye(3, dtype="float32"))

# A structured log line must still come out: the trace-context lookup runs
# inside format(), so a fault there costs the record, not just the trace id.
import io, json, logging
from rag_llm_infra.log_config import _JsonFormatter
buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
handler.setFormatter(_JsonFormatter())
log = logging.getLogger("probe")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.info("hello", extra={{"request_id": "r1"}})
emitted = json.loads(buffer.getvalue().strip())
assert emitted["msg"] == "hello" and emitted["request_id"] == "r1", emitted

print("OK", rag_llm_infra.__version__, store.backend_name, store.size)
"""


@pytest.mark.parametrize("target", OPTIONAL)
def test_the_package_still_imports_and_retrieves(target: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", _BREAK.format(target=target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, f"{target}: {done.stderr[-800:]}"
    assert done.stdout.startswith("OK "), done.stdout


def test_the_tracing_entry_points_degrade_when_otel_is_unloadable() -> None:
    """Importing the package was already covered; calling into tracing was not.
    An unloadable OTel raises OSError, which `except ImportError` alone lets
    through, so each entry point has to be driven, not just the import."""
    probe = _BREAK.format(target="opentelemetry").replace(
        "from rag_llm_infra import get_vector_store, get_llm",
        "from rag_llm_infra import tracing\n"
        "tracing.configure_tracing()\n"
        "tracing.get_tracer()\n"
        "assert tracing.current_trace_context() == "
        '{"trace_id": "", "span_id": ""}\n'
        "from rag_llm_infra import get_vector_store, get_llm",
    )
    # A replace whose anchor moved returns the template untouched, and the
    # template alone prints OK: the assertions below would pass having called
    # nothing.
    assert "configure_tracing" in probe, "the anchor moved; this probe is empty"
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.startswith("OK "), done.stdout


def test_a_raising_trace_lookup_does_not_cost_the_log_record(monkeypatch) -> None:
    """`current_trace_context` now degrades instead of raising, so only a
    direct fault reaches the handler in `log_config` that exists to keep the
    record. Without one, narrowing that handler changes nothing observable."""
    import io
    import json
    import logging

    from rag_llm_infra import tracing
    from rag_llm_infra.log_config import _JsonFormatter

    def _raise() -> dict[str, str]:
        raise OSError(126, "simulated: tracer unloadable")

    monkeypatch.setattr(tracing, "current_trace_context", _raise)

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(_JsonFormatter())
    log = logging.getLogger("trace_fault_probe")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info("hello")
    finally:
        log.removeHandler(handler)

    emitted = json.loads(buffer.getvalue().strip())
    assert emitted["msg"] == "hello", emitted
    assert emitted["trace_id"] == "", emitted


def test_the_control_shows_the_breaker_really_breaks_the_import() -> None:
    """Without it the probe proves nothing: the module might just be absent."""
    probe = _BREAK.format(target="faiss").replace(
        "import rag_llm_infra\n", "import faiss\nimport rag_llm_infra\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert done.returncode != 0
    assert "simulated" in done.stderr
