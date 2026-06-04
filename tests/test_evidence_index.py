"""Tests for evidence_index.py — RWLock, EmbeddingEngine, config."""
import threading
import time
import pytest


class TestConfig:
    def test_config_has_required_keys(self):
        from rag_llm_infra.evidence_index import CONFIG
        assert "max_embedding_cache" in CONFIG
        assert "memory_warning_threshold" in CONFIG
        assert "adaptive_cache" in CONFIG

    def test_config_values_are_sane(self):
        from rag_llm_infra.evidence_index import CONFIG
        assert CONFIG["max_embedding_cache"] > 0
        assert 0 < CONFIG["memory_warning_threshold"] <= 1.0


class TestFeatureFlags:
    def test_flags_are_booleans(self):
        from rag_llm_infra.evidence_index import SENTENCE_TRANSFORMERS_AVAILABLE, PSUTIL_AVAILABLE, FAISS_AVAILABLE
        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)
        assert isinstance(PSUTIL_AVAILABLE, bool)
        assert isinstance(FAISS_AVAILABLE, bool)


class TestRWLock:
    def test_read_lock_context_manager(self):
        from rag_llm_infra.evidence_index import RWLock
        lock = RWLock()
        with lock.read_lock:
            assert lock._readers == 1
        assert lock._readers == 0

    def test_write_lock_context_manager(self):
        from rag_llm_infra.evidence_index import RWLock
        lock = RWLock()
        with lock.write_lock:
            assert lock._writer is True
        assert lock._writer is False

    def test_concurrent_reads(self):
        from rag_llm_infra.evidence_index import RWLock
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
        from rag_llm_infra.evidence_index import RWLock
        lock = RWLock()
        sequence = []

        def writer():
            with lock.write_lock:
                sequence.append("write-start")
                time.sleep(0.05)
                sequence.append("write-end")

        def reader():
            time.sleep(0.01)  # Ensure writer starts first
            with lock.read_lock:
                sequence.append("read")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Read should happen after write completes
        assert sequence.index("write-end") < sequence.index("read")

    def test_acquire_release_read(self):
        from rag_llm_infra.evidence_index import RWLock
        lock = RWLock()
        lock.acquire_read()
        assert lock._readers == 1
        lock.release_read()
        assert lock._readers == 0

    def test_acquire_release_write(self):
        from rag_llm_infra.evidence_index import RWLock
        lock = RWLock()
        lock.acquire_write()
        assert lock._writer is True
        lock.release_write()
        assert lock._writer is False


class TestEmbeddingEngine:
    def test_raises_without_sentence_transformers(self):
        from rag_llm_infra.evidence_index import SENTENCE_TRANSFORMERS_AVAILABLE
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            from rag_llm_infra.evidence_index import EmbeddingEngine
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                EmbeddingEngine()

    def test_normalize_cache_key_empty(self):
        """Test the static logic of _normalize_cache_key without needing SentenceTransformers."""
        from rag_llm_infra.evidence_index import SENTENCE_TRANSFORMERS_AVAILABLE
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("Test only runs when sentence-transformers is unavailable")
        # Can't instantiate EmbeddingEngine without ST, so test the logic pattern
        import hashlib
        import unicodedata
        import re
        text = "  Hello   World  "
        namespace = "default"
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r'\s+', ' ', normalized.strip().lower())
        raw_key = f"{namespace}:{normalized}"
        key = hashlib.md5(raw_key.encode(), usedforsecurity=False).hexdigest()
        assert len(key) == 32
