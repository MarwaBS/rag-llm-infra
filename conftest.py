import os
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env_root() -> Generator[None, None, None]:
    """Set consistent, test-safe environment defaults for the whole suite."""
    monkeypatch = pytest.MonkeyPatch()
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-key-for-testing")

    monkeypatch.setenv("DEBUG_MODE", "true")
    monkeypatch.setenv("DISABLE_SENTENCE_TRANSFORMERS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    yield
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def mock_torch_for_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid GPU usage if torch is present during tests."""
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Torch absent in CI is the expected path; skip the monkeypatch silently.
    except Exception:  # nosec B110
        pass
