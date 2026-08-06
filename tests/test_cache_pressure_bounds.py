"""Memory pressure only ever shrinks the cache, and the limit comes back.

The trim derives its new limit from the configured ceiling. Deriving it from the
current one instead compounds on every check, and any fixed floor above a small
configured limit raises the cap exactly when memory is scarce.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

import rag_llm_infra.evidence_index as ei
from rag_llm_infra.evidence_index import CONFIG


class _FakeEmbedder:
    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003
        return np.ones((len(texts), 4), dtype="float32")


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch):
    # psutil is faked below, so the trim must run whether or not it is installed.
    monkeypatch.setattr(ei, "PSUTIL_AVAILABLE", True)
    monkeypatch.setitem(CONFIG, "adaptive_cache", True)
    monkeypatch.setitem(CONFIG, "memory_warning_threshold", 0.8)
    return ei.EmbeddingEngine(model=_FakeEmbedder())


def _set_memory(monkeypatch: pytest.MonkeyPatch, percent: float) -> None:
    monkeypatch.setattr(
        ei,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=percent)),
    )


@pytest.mark.parametrize("configured", [1, 2, 50, 150, 199, 200, 2000])
def test_pressure_never_raises_the_configured_limit(
    engine, monkeypatch: pytest.MonkeyPatch, configured: int
) -> None:
    engine._configured_cache_size = configured
    engine._max_cache_size = configured
    _set_memory(monkeypatch, 99.0)
    engine._last_memory_check = 0.0
    engine._check_memory_pressure()
    assert engine._max_cache_size <= configured


def test_the_limit_returns_when_pressure_clears(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine._configured_cache_size = 50
    engine._max_cache_size = 50
    _set_memory(monkeypatch, 99.0)
    engine._last_memory_check = 0.0
    engine._check_memory_pressure()
    assert engine._max_cache_size == 25

    _set_memory(monkeypatch, 10.0)
    engine._last_memory_check = 0.0
    engine._check_memory_pressure()
    assert engine._max_cache_size == 50


def test_repeated_pressure_does_not_compound(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Halving the current limit each time walks it to the floor over a long
    incident; halving the configured one settles."""
    engine._configured_cache_size = 2000
    engine._max_cache_size = 2000
    _set_memory(monkeypatch, 99.0)
    for _ in range(5):
        engine._last_memory_check = 0.0
        engine._check_memory_pressure()
    assert engine._max_cache_size == 1000


def test_the_control_shows_the_trim_runs_at_all(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine._configured_cache_size = 2000
    engine._max_cache_size = 2000
    _set_memory(monkeypatch, 10.0)
    engine._last_memory_check = 0.0
    engine._check_memory_pressure()
    assert engine._max_cache_size == 2000
    assert engine._last_memory_check > 0.0
    assert time.time() - engine._last_memory_check < 30
