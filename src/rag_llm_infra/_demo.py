"""Deterministic, dependency-light hashing embedder for the example, the eval
harness, and the serving demo — one shared implementation.

Bag-of-tokens hashing into a fixed-width vector. Reproducible across processes
(uses hashlib, not the salted built-in `hash()`). NOT for production — swap for
`EmbeddingEngine` (real sentence embeddings).
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

EMBED_DIM = 128


def embed(texts: List[str]) -> np.ndarray:
    """Embed `texts` into an `(N, EMBED_DIM)` float32 matrix."""
    vecs = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBED_DIM
            vecs[row, bucket] += 1.0
    return vecs
