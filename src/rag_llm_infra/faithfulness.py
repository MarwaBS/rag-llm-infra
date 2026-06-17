"""Groundedness (faithfulness) metric for RAG output.

`groundedness(answer, contexts)` returns the fraction of the answer's content
words that are supported by the retrieved context — a cheap, deterministic
faithfulness signal you can run on every generation and gate CI on.

It is a *lexical* proxy: it catches answers that introduce vocabulary absent from
the evidence (a common hallucination signature). It does not judge semantics, so
it has three known blind spots by construction — treat it as a cheap tripwire, not
a faithfulness guarantee, and pair it with an LLM-judge for nuanced cases:

  - **Negation-blind.** "Paris is not the capital" scores the same as "Paris is
    the capital" — `not` is a closed-class stop word, so flipping the claim's
    polarity is invisible to a bag-of-words overlap.
  - **Dilution.** A true claim padded with a false one keeps most of its tokens
    grounded, so a single fabricated clause only dents the score rather than
    sinking it.
  - **Vocabulary, not propositions.** It scores token presence, not whether the
    evidence actually *asserts* the answer's claim.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

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


def _content_tokens(text: str) -> set[str]:
    # len >= 2 (not > 2) so two-character acronyms — US, AI, ML, UK — count as
    # groundable content. Dropping them made answers built from acronyms score a
    # vacuous 1.0 (no content tokens at all). Single characters stay out: they are
    # almost always list markers or stray letters, not claims.
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOP and len(t) >= 2
    }


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of the answer's content tokens present in the union of `contexts`.

    Returns 1.0 when the answer makes no groundable claim (no content tokens),
    and 0.0 when it makes claims but no context supports them. Result is in [0, 1].
    """
    claim = _content_tokens(answer)
    if not claim:
        return 1.0
    support: set[str] = set()
    for context in contexts:
        support |= _content_tokens(context)
    return len(claim & support) / len(claim)
