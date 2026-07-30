"""Deterministic text materialization with calibrated token targeting.

The sampler and pool work in TOKENS; an endpoint accepts TEXT. This module
turns (doc_id, prefix_tokens, suffix_tokens) into real message text such
that:

  1. The same doc_id always yields byte-identical text (seeded by doc_id),
     so shared prefixes tokenize to identical leading tokens on ANY
     tokenizer. That property, not token counting, is what makes prefix
     caching engage.
  2. Token counts are targeted through a characters-per-token ratio (cpt).
     The default 4.0 is an approximation and is TREATED as one: the runner
     calibrates cpt against the endpoint's reported prompt_tokens during the
     warmup phase, and every report prints the residual token-targeting
     error. Endpoint-reported token counts are the source of truth in all
     tables.

Text is synthetic English-like prose (seeded word salad with sentence and
paragraph structure). It exercises tokenizers realistically without
containing anyone's data, so it is safe to share and to run before any
customer dataset lands.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

DEFAULT_CPT = 4.0

_WORDS = (
    "account update customer order status agent response ticket policy plan "
    "billing invoice refund shipping address device network error retry login "
    "password profile support issue resolved pending escalation priority queue "
    "message thread history context detail summary action item schedule change "
    "service request system record option setting balance payment method card "
    "subscription renewal cancel upgrade downgrade limit usage report metric "
    "latency throughput token model endpoint request response stream batch "
    "session window channel partner vendor region zone cluster node capacity "
    "the a an of to in for with on at by from about into over after before "
    "please verify confirm review check ensure provide describe explain list"
).split()


def _rng_for(tag: str, seed_root: int) -> np.random.Generator:
    h = hashlib.sha256(f"{seed_root}:{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _prose(rng: np.random.Generator, n_chars: int) -> str:
    """Sentence/paragraph structured pseudo-prose of ~n_chars characters."""
    out: list[str] = []
    total = 0
    sent_len = 0
    target_sent = int(rng.integers(8, 15))
    since_para = 0
    while total < n_chars:
        w = _WORDS[int(rng.integers(0, len(_WORDS)))]
        if sent_len == 0:
            w = w.capitalize()
        out.append(w)
        total += len(w) + 1
        sent_len += 1
        if sent_len >= target_sent:
            out[-1] = out[-1] + "."
            sent_len = 0
            target_sent = int(rng.integers(8, 15))
            since_para += 1
            if since_para >= 6:
                out[-1] = out[-1] + "\n\n"
                since_para = 0
    return " ".join(out)[:n_chars]


class TextMaterializer:
    """Turns token plans into concrete chat messages."""

    def __init__(self, cpt: float = DEFAULT_CPT, seed_root: int = 1337,
                 doc_cache_size: int = 64):
        self.cpt = float(cpt)
        self.seed_root = seed_root
        # doc text is deterministic given (doc_id, char length); cache the
        # longest cut per doc and slice from it.
        self._doc_full = lru_cache(maxsize=doc_cache_size)(self._doc_full_impl)

    # -- documents (shared prefixes) ------------------------------------
    def _doc_full_impl(self, doc_id: int, max_chars: int) -> str:
        rng = _rng_for(f"doc:{doc_id}", self.seed_root)
        return _prose(rng, max_chars)

    def prefix_text(self, doc_id: int, prefix_tokens: int,
                    doc_len_tokens: int) -> str:
        if doc_id < 0 or prefix_tokens <= 0:
            return ""
        max_chars = int(doc_len_tokens * self.cpt)
        want_chars = int(prefix_tokens * self.cpt)
        return self._doc_full(doc_id, max_chars)[:want_chars]

    # -- unique suffixes -------------------------------------------------
    def suffix_text(self, request_id: str, suffix_tokens: int) -> str:
        rng = _rng_for(f"req:{request_id}", self.seed_root)
        n_chars = max(int(suffix_tokens * self.cpt) - 64, 32)
        body = _prose(rng, n_chars)
        return (f"{body}\n\n[case {request_id}] Given the context above, "
                f"what is the correct next action for this customer?")

    # -- messages ---------------------------------------------------------
    def messages(self, request_id: str, doc_id: int, prefix_tokens: int,
                 doc_len_tokens: int, suffix_tokens: int) -> list[dict]:
        """Chat messages: shared prefix as system, unique tail as user.

        This mirrors the agent-workload pattern (stable system prompt plus
        retrieved context, short new user turn) and keeps the shared text
        leading, which is the position prefix caches match on.
        """
        msgs = []
        pre = self.prefix_text(doc_id, prefix_tokens, doc_len_tokens)
        if pre:
            msgs.append({"role": "system", "content": pre})
        msgs.append({"role": "user",
                     "content": self.suffix_text(request_id, suffix_tokens)})
        return msgs


def calibrate_cpt(cpt_used: float, chars_sent: int,
                  prompt_tokens_reported: int) -> float:
    """New cpt from endpoint-reported truth. Guarded against silly values."""
    if prompt_tokens_reported <= 0 or chars_sent <= 0:
        return cpt_used
    measured = chars_sent / prompt_tokens_reported
    return min(max(measured, 1.5), 12.0)
