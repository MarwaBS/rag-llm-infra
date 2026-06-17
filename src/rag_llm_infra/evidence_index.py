"""
Embedding index — configuration, feature flags, concurrency utilities, and embedding engine.

Contains the CONFIG dict, optional library feature flags (FAISS, SentenceTransformers, psutil),
an RWLock for concurrent read/write access, and EmbeddingEngine for sentence-level
embeddings with adaptive, memory-pressure-aware caching.
"""

import os
import re
import hashlib
import logging
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import List, Dict, Any, Optional
import numpy as np

# FAISS import for cache compatibility checks
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================
CONFIG: Dict[str, Any] = {
    "max_embedding_cache": int(os.getenv("EVIDENCE_MAX_CACHE", "2000")),
    "memory_warning_threshold": float(os.getenv("EVIDENCE_MEMORY_WARN", "0.8")),
    "adaptive_cache": os.getenv("EVIDENCE_ADAPTIVE_CACHE", "true").lower() == "true",
    # Pin the embedding model revision for reproducible loads. Defaults to
    # "main"; production should set an immutable commit SHA via the env var.
    "embedding_model_revision": os.getenv("EVIDENCE_EMBEDDING_REVISION", "main"),
}

# ===============================
# DEPENDENCY ROBUSTNESS
# ===============================
SENTENCE_TRANSFORMERS_AVAILABLE = False
PSUTIL_AVAILABLE = False

try:
    if os.getenv("DISABLE_SENTENCE_TRANSFORMERS") == "1":
        raise ImportError("SentenceTransformers disabled for tests")
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    logger.debug("SentenceTransformers unavailable; embedding features disabled: %s", e)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    logger.debug("psutil unavailable; adaptive memory-pressure trimming disabled.")


# ===============================
# CONCURRENCY UTILITIES
# ===============================
class RWLock:
    """Reader-writer lock: concurrent reads, exclusive writes, writer-preferring.

    Writer preference avoids writer starvation: once a writer is waiting, new
    readers queue behind it (the previous version let a steady stream of readers
    keep `_readers > 0` forever, so a waiting writer could never proceed). Used
    by EmbeddingEngine — short read-locked cache lookups, exclusive write-locked
    inserts/trims — with the slow `model.encode` happening outside the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            # Yield to any waiting or active writer (writer preference).
            while self._writer or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    @property
    def read_lock(self) -> Any:
        class _Read:
            def __init__(self, rw: "RWLock") -> None:
                self._rw = rw

            def __enter__(self) -> None:
                self._rw.acquire_read()

            def __exit__(self, *a: Any) -> None:
                self._rw.release_read()

        return _Read(self)

    @property
    def write_lock(self) -> Any:
        class _Write:
            def __init__(self, rw: "RWLock") -> None:
                self._rw = rw

            def __enter__(self) -> None:
                self._rw.acquire_write()

            def __exit__(self, *a: Any) -> None:
                self._rw.release_write()

        return _Write(self)


# ===============================
# EMBEDDING ENGINE
# ===============================
class EmbeddingEngine:
    """Sentence embeddings with concurrent-read caching and memory-pressure trimming.

    Concurrency: a reader-writer lock (``RWLock``) guards the cache. Lookups take
    the read lock (so cache hits run concurrently), and the slow
    ``model.encode`` of cache misses runs OUTSIDE the lock — only the resulting
    inserts take the exclusive write lock. The previous version held one lock
    across the whole call, so every cache hit blocked behind another thread's
    inference. Eviction is insertion-order (oldest first); a read does not
    refresh recency, which is what lets lookups avoid the write lock.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        model: Optional[Any] = None,
        revision: Optional[str] = None,
    ) -> None:
        """
        model: inject a pre-built embedder (anything with
            ``encode(list[str], convert_to_numpy=..., show_progress_bar=...) ->
            ndarray``). Lets the engine be exercised without sentence-transformers
            and lets callers supply a custom model. When None, a
            ``SentenceTransformer`` is loaded.
        revision: pin the model revision for reproducible loads. Defaults to
            ``CONFIG['embedding_model_revision']`` (env EVIDENCE_EMBEDDING_REVISION).
        """
        if model is not None:
            self.model = model
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Error: 'sentence-transformers' package not installed. "
                    "EmbeddingEngine requires this package for embeddings, or pass "
                    "model=<your embedder>. Install with: pip install sentence-transformers"
                )
            rev = revision or CONFIG["embedding_model_revision"]
            self.model = SentenceTransformer(model_name, revision=rev)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = RWLock()
        self._stats_lock = threading.Lock()
        self._max_cache_size = CONFIG["max_embedding_cache"]
        self._total_requests = 0
        self._cache_hits = 0
        self._last_memory_check = time.time()
        logger.info(
            f"EmbeddingEngine initialized with adaptive cache (max: {self._max_cache_size})"
        )

    def _normalize_cache_key(self, text: str, namespace: str) -> str:
        if not text:
            return ""
        # NFKC unicode canonicalization ONLY. Do NOT lowercase or collapse
        # whitespace: the embedding model is case- and spacing-sensitive ("US"
        # and "us", "a b" and "a  b" embed differently), so the key must keep
        # them distinct — otherwise a lookup returns the wrong cached vector.
        normalized = unicodedata.normalize("NFKC", text)
        raw_key = f"{namespace}:{normalized}"
        return hashlib.md5(raw_key.encode(), usedforsecurity=False).hexdigest()

    def _check_memory_pressure(self) -> None:
        if not CONFIG["adaptive_cache"] or not PSUTIL_AVAILABLE:
            return
        if time.time() - self._last_memory_check < 30:
            return
        # Reset the throttle as soon as we pass it, regardless of whether
        # pressure is found. The old code only updated the timestamp inside the
        # pressure branch, so once 30s elapsed without pressure, psutil was
        # polled on EVERY subsequent call.
        self._last_memory_check = time.time()
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > CONFIG["memory_warning_threshold"] * 100:
                reduction = max(100, int(self._max_cache_size * 0.5))
                with self._lock.write_lock:
                    while len(self._cache) > reduction:
                        self._cache.popitem(last=False)
                self._max_cache_size = reduction
        # Memory-pressure cache trim is best-effort; never break ingest on failure.
        except Exception:  # nosec B110
            pass

    def embed_batch(
        self, texts: List[str], namespace: str = "default"
    ) -> "np.ndarray[Any, Any]":
        if not texts:
            return np.empty((0, 0), dtype="float32")
        self._check_memory_pressure()
        keys = [self._normalize_cache_key(t, namespace) for t in texts]
        results: List[Any] = [None] * len(texts)

        # Read phase — concurrent readers share the cache.
        with self._lock.read_lock:
            for i, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    results[i] = cached

        hits = sum(1 for r in results if r is not None)
        with self._stats_lock:
            self._total_requests += len(texts)
            self._cache_hits += hits

        # Compute misses OUTSIDE any lock — model.encode is the slow part and
        # must not block concurrent cache hits.
        miss_indices = [i for i, r in enumerate(results) if r is None]
        if miss_indices:
            computed = self.model.encode(
                [texts[i] for i in miss_indices],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            with self._lock.write_lock:
                for i, emb in zip(miss_indices, computed):
                    self._cache[keys[i]] = emb
                    results[i] = emb
                while len(self._cache) > self._max_cache_size:
                    self._cache.popitem(last=False)
        return np.stack(results)

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache statistics for monitoring."""
        with self._stats_lock:
            total_requests = self._total_requests
            cache_hits = self._cache_hits
        with self._lock.read_lock:
            cache_size = len(self._cache)
        return {
            "cache_size": cache_size,
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / max(1, total_requests), 4),
        }
