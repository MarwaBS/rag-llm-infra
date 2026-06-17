"""Tests for evidence_index.py — RWLock, EmbeddingEngine, config.

EmbeddingEngine is exercised for real by injecting a deterministic fake embedder
(``model=...``), so the cache, cache-key correctness, eviction, stats, and the
encode-outside-lock concurrency are all covered without sentence-transformers.
"""

import threading
import time

import numpy as np
import pytest

from rag_llm_infra.evidence_index import CONFIG, RWLock


class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer. Records each encode batch
    and maps each text to a vector that differs by case/length, so a wrong cache
    hit (e.g. "US" vs "us") is detectable."""

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        return np.array(
            [[float(len(t)), float(sum(map(ord, t)) % 100), 0.0, 0.0] for t in texts],
            dtype="float32",
        )

    @property
    def total_encoded(self) -> int:
        return sum(len(batch) for batch in self.encode_calls)


def _engine(**kwargs):
    from rag_llm_infra.evidence_index import EmbeddingEngine

    return EmbeddingEngine(model=_FakeEmbedder(), **kwargs)


class TestConfig:
    def test_config_has_required_keys(self):
        assert "max_embedding_cache" in CONFIG
        assert "memory_warning_threshold" in CONFIG
        assert "adaptive_cache" in CONFIG
        assert "embedding_model_revision" in CONFIG

    def test_config_values_are_sane(self):
        assert CONFIG["max_embedding_cache"] > 0
        assert 0 < CONFIG["memory_warning_threshold"] <= 1.0


class TestFeatureFlags:
    def test_flags_are_booleans(self):
        from rag_llm_infra.evidence_index import (
            FAISS_AVAILABLE,
            PSUTIL_AVAILABLE,
            SENTENCE_TRANSFORMERS_AVAILABLE,
        )

        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)
        assert isinstance(PSUTIL_AVAILABLE, bool)
        assert isinstance(FAISS_AVAILABLE, bool)


class TestRWLock:
    def test_read_lock_context_manager(self):
        lock = RWLock()
        with lock.read_lock:
            assert lock._readers == 1
        assert lock._readers == 0

    def test_write_lock_context_manager(self):
        lock = RWLock()
        with lock.write_lock:
            assert lock._writer is True
        assert lock._writer is False

    def test_concurrent_reads(self):
        lock = RWLock()
        results = []

        def reader(idx):
            with lock.read_lock:
                results.append(f"read-{idx}")
                time.sleep(0.01)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 3

    def test_write_excludes_reads(self):
        lock = RWLock()
        sequence = []

        def writer():
            with lock.write_lock:
                sequence.append("write-start")
                time.sleep(0.05)
                sequence.append("write-end")

        def reader():
            time.sleep(0.01)
            with lock.read_lock:
                sequence.append("read")

        t1, t2 = threading.Thread(target=writer), threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert sequence.index("write-end") < sequence.index("read")

    def test_writer_not_starved_by_continuous_readers(self):
        """A waiting writer must not be starved by a stream of new readers
        (writer preference). The old lock let new readers keep _readers > 0 so a
        waiting writer never proceeded."""
        lock = RWLock()
        lock.acquire_read()  # an existing reader holds the lock

        writer_done = threading.Event()

        def writer():
            lock.acquire_write()
            writer_done.set()
            lock.release_write()

        wt = threading.Thread(target=writer)
        wt.start()

        deadline = time.time() + 2.0
        while lock._writers_waiting == 0 and time.time() < deadline:
            time.sleep(0.005)
        assert lock._writers_waiting == 1  # writer is queued

        reader2_done = threading.Event()

        def reader2():
            lock.acquire_read()
            reader2_done.set()
            lock.release_read()

        rt = threading.Thread(target=reader2)
        rt.start()
        time.sleep(0.05)
        # The new reader must yield to the waiting writer, not jump the queue.
        assert not reader2_done.is_set()
        assert not writer_done.is_set()  # writer still blocked by the first reader

        lock.release_read()  # release the original reader -> writer can proceed
        assert writer_done.wait(timeout=2.0)
        assert reader2_done.wait(timeout=2.0)
        wt.join()
        rt.join()


class TestEmbeddingEngine:
    def test_raises_without_model_and_without_sentence_transformers(self):
        from rag_llm_infra.evidence_index import (
            SENTENCE_TRANSFORMERS_AVAILABLE,
            EmbeddingEngine,
        )

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip(
                "sentence-transformers installed; the no-model error path is unreachable"
            )
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            EmbeddingEngine()

    def test_injected_model_embeds(self):
        eng = _engine()
        out = eng.embed_batch(["alpha", "beta gamma"])
        assert out.shape == (2, 4)

    def test_cache_hit_skips_recompute(self):
        fake = _FakeEmbedder()
        from rag_llm_infra.evidence_index import EmbeddingEngine

        eng = EmbeddingEngine(model=fake)
        eng.embed_batch(["hello world"])
        eng.embed_batch(["hello world"])  # identical -> cache hit
        assert fake.total_encoded == 1  # encoded once, not twice
        assert eng.get_stats()["cache_hits"] == 1

    def test_duplicate_texts_in_one_batch_encoded_once(self):
        """Regression: a text repeated within a single embed_batch used to be
        re-encoded once per occurrence (dedup happened only against the cache, not
        within the batch). Identical misses must collapse to one encode, and every
        occurrence must still receive the (identical) vector."""
        fake = _FakeEmbedder()
        from rag_llm_infra.evidence_index import EmbeddingEngine

        eng = EmbeddingEngine(model=fake)
        out = eng.embed_batch(["dup", "unique", "dup", "dup"])
        assert out.shape == (4, 4)
        # Two unique texts -> two encoded, not four.
        assert fake.total_encoded == 2, f"expected 2 encodes, got {fake.total_encoded}"
        # All three "dup" rows are the same vector.
        assert np.array_equal(out[0], out[2])
        assert np.array_equal(out[0], out[3])

    def test_cache_key_is_case_and_space_sensitive(self):
        """Regression: the key used to lowercase + collapse whitespace, so "US"
        and "us" collided and the second lookup returned the WRONG vector. They
        must both be encoded and yield distinct embeddings."""
        fake = _FakeEmbedder()
        from rag_llm_infra.evidence_index import EmbeddingEngine

        eng = EmbeddingEngine(model=fake)
        a = eng.embed_batch(["US"])
        b = eng.embed_batch(["us"])
        assert fake.total_encoded == 2, (
            "case-different texts must not share a cache entry"
        )
        assert not np.array_equal(a[0], b[0])

    def test_eviction_caps_cache_size(self):
        eng = _engine()
        eng._max_cache_size = 3
        eng.embed_batch([f"text number {i}" for i in range(10)])
        assert eng.get_stats()["cache_size"] <= 3

    def test_stats_shape(self):
        eng = _engine()
        eng.embed_batch(["a", "b"])
        eng.embed_batch(["a"])  # one hit
        stats = eng.get_stats()
        assert stats["total_requests"] == 3
        assert stats["cache_hits"] == 1
        assert 0.0 <= stats["cache_hit_rate"] <= 1.0

    def test_empty_input_returns_empty(self):
        eng = _engine()
        out = eng.embed_batch([])
        assert out.shape == (0, 0)

    def test_concurrent_embeds_are_consistent(self):
        # Slow encode so threads genuinely overlap; cache hits must not be
        # serialized behind a miss's inference. Just assert no errors + correct
        # vectors out (the encode-outside-lock change must stay correct).
        class _SlowFake(_FakeEmbedder):
            def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
                time.sleep(0.01)
                return super().encode(texts, convert_to_numpy, show_progress_bar)

        from rag_llm_infra.evidence_index import EmbeddingEngine

        eng = EmbeddingEngine(model=_SlowFake())
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(10):
                    out = eng.embed_batch(["shared one", "shared two"])
                    assert out.shape == (2, 4)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
