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
CONFIG = {
    'max_embedding_cache': int(os.getenv('EVIDENCE_MAX_CACHE', '2000')),
    'memory_warning_threshold': float(os.getenv('EVIDENCE_MEMORY_WARN', '0.8')),
    'adaptive_cache': os.getenv('EVIDENCE_ADAPTIVE_CACHE', 'true').lower() == 'true',
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
    """Reader-writer lock allowing concurrent reads and exclusive writes."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writer = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    @property
    def read_lock(self) -> Any:
        class _Read:
            def __init__(self, rw: "RWLock") -> None: self._rw = rw
            def __enter__(self) -> None: self._rw.acquire_read()
            def __exit__(self, *a: Any) -> None: self._rw.release_read()
        return _Read(self)

    @property
    def write_lock(self) -> Any:
        class _Write:
            def __init__(self, rw: "RWLock") -> None: self._rw = rw
            def __enter__(self) -> None: self._rw.acquire_write()
            def __exit__(self, *a: Any) -> None: self._rw.release_write()
        return _Write(self)


# ===============================
# EMBEDDING ENGINE
# ===============================
class EmbeddingEngine:
    """Manages sentence embeddings with adaptive caching and memory pressure handling."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Error: 'sentence-transformers' package not installed. "
                "EmbeddingEngine requires this package for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._max_cache_size = CONFIG['max_embedding_cache']
        self._total_requests = 0
        self._cache_hits = 0
        self._last_memory_check = time.time()
        logger.info(f"EmbeddingEngine initialized with adaptive cache (max: {self._max_cache_size})")

    def _normalize_cache_key(self, text: str, namespace: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r'\s+', ' ', normalized.strip().lower())
        raw_key = f"{namespace}:{normalized}"
        return hashlib.md5(raw_key.encode(), usedforsecurity=False).hexdigest()

    def _check_memory_pressure(self) -> None:
        if not CONFIG['adaptive_cache'] or not PSUTIL_AVAILABLE:
            return
        if time.time() - self._last_memory_check < 30:
            return
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > CONFIG['memory_warning_threshold'] * 100:
                reduction = max(100, int(self._max_cache_size * 0.5))
                with self._lock:
                    while len(self._cache) > reduction:
                        self._cache.popitem(last=False)
                self._max_cache_size = reduction
                self._last_memory_check = time.time()
        # Memory-pressure cache trim is best-effort; never break ingest on failure.
        except Exception:  # nosec B110
            pass

    def embed_batch(self, texts: List[str], namespace: str = "default") -> "np.ndarray[Any, Any]":
        if not texts:
            return np.array([])
        self._check_memory_pressure()
        self._total_requests += len(texts)
        with self._lock:
            to_compute: List[str] = []
            compute_indices: List[int] = []
            results: List[Any] = [None] * len(texts)
            for i, text in enumerate(texts):
                key = self._normalize_cache_key(text, namespace)
                if key in self._cache:
                    self._cache_hits += 1
                    self._cache.move_to_end(key)
                    results[i] = self._cache[key]
                else:
                    to_compute.append(text)
                    compute_indices.append(i)
            if to_compute:
                computed = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
                for idx, text, emb in zip(compute_indices, to_compute, computed):
                    key = self._normalize_cache_key(text, namespace)
                    self._cache[key] = emb
                    results[idx] = emb
                while len(self._cache) > self._max_cache_size:
                    self._cache.popitem(last=False)
            return np.stack(results)

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache statistics for monitoring."""
        with self._lock:
            total_requests = max(1, self._total_requests)
            return {
                'cache_size': len(self._cache),
                'total_requests': self._total_requests,
                'cache_hits': self._cache_hits,
                'cache_hit_rate': round(self._cache_hits / total_requests, 4),
            }
