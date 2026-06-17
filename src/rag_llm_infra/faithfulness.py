"""Groundedness (faithfulness) metric for RAG output.

`groundedness(answer, contexts)` returns the fraction of the answer's content
words that are supported by the retrieved context — a cheap, deterministic
faithfulness signal you can run on every generation and gate CI on.

It is a *lexical* proxy: it catches answers that introduce vocabulary absent from
the evidence (a common hallucination signature). It does not judge semantics —
pair it with an LLM-judge for nuanced faithfulness.
"""

from __future__ import annotations

import re
from typing import Sequence, Set

# Closed-class stop words carry no groundable claim; content words do.
_STOP: frozenset = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "to",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "with",
        "for",
        "on",
        "by",
        "as",
        "from",
        "at",
        "into",
        "over",
        "than",
        "then",
        "so",
        "such",
        "but",
        "not",
        "no",
        "can",
        "will",
        "they",
        "their",
    }
)


def _content_tokens(text: str) -> Set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOP and len(t) > 2
    }


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of the answer's content tokens present in the union of `contexts`.

    Returns 1.0 when the answer makes no groundable claim (no content tokens),
    and 0.0 when it makes claims but no context supports them. Result is in [0, 1].
    """
    claim = _content_tokens(answer)
    if not claim:
        return 1.0
    support: Set[str] = set()
    for context in contexts:
        support |= _content_tokens(context)
    return len(claim & support) / len(claim)
