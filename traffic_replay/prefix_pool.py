"""Prefix pool: constructs traffic that PRODUCES a target cache-hit ratio.

You cannot ask an endpoint for a 60% prompt-cache hit rate; you have to send
traffic whose structure produces one. Prompt caching keys on shared leading
tokens, so each request is assembled as:

    [shared prefix: leading slice of a pooled document] + [unique suffix]

Pool design:
  * Documents are bucketed by length so a request wanting an 8K-token prefix
    draws an 8K-class document, not a random one.
  * Popularity inside a bucket is Zipf-skewed (a few hot documents, a long
    tail), the way real knowledge-base content repeats.
  * A request wanting w tokens uses the leading w tokens of its document.
    Two requests cutting the same document at different lengths still share
    leading tokens, which is exactly how block-level prefix caches match.
  * First use of a document is a cold miss, later uses are warm. Whether a
    given request actually hits is the ENDPOINT'S business: the harness
    reports the endpoint's cached-token counts, never its own assumption
    (see metrics.py). The pool only guarantees the structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BUCKETS = (0, 2_000, 6_000, 12_000, 30_000, 200_000)
TOP_BUCKET_DOC_TOKENS = 40_000  # cap document size for memory sanity


@dataclass
class Assignment:
    doc_id: np.ndarray        # pooled document per request
    prefix_tokens: np.ndarray  # tokens actually taken from the document


class PrefixPool:
    """Assigns each request a (document, prefix length) pair."""

    def __init__(self, bucket_edges=DEFAULT_BUCKETS,
                 docs_per_bucket: int = 40, zipf_s: float = 1.1,
                 seed: int = 11):
        self.edges = tuple(bucket_edges)
        self.zipf_s = zipf_s
        self.rng = np.random.default_rng(seed)
        self.doc_len: dict[int, int] = {}
        self.buckets: dict[int, list[int]] = {}
        did = 0
        for b in range(len(self.edges) - 1):
            hi = min(self.edges[b + 1], TOP_BUCKET_DOC_TOKENS)
            ids = []
            for _ in range(docs_per_bucket):
                self.doc_len[did] = hi
                ids.append(did)
                did += 1
            self.buckets[b] = ids
        # Precompute Zipf weights once per bucket size.
        n = docs_per_bucket
        w = 1.0 / np.arange(1, n + 1) ** self.zipf_s
        self._weights = w / w.sum()

    def bucket_of(self, want: int) -> int:
        for b in range(len(self.edges) - 1):
            if self.edges[b] <= want < self.edges[b + 1]:
                return b
        return len(self.edges) - 2

    def assign(self, prefix_tokens: np.ndarray) -> Assignment:
        n = len(prefix_tokens)
        ids = np.empty(n, dtype=int)
        actual = np.empty(n, dtype=int)
        for i, want in enumerate(np.asarray(prefix_tokens, dtype=int)):
            if want <= 0:
                ids[i] = -1
                actual[i] = 0
                continue
            b = self.bucket_of(int(want))
            bucket = self.buckets[b]
            doc = int(self.rng.choice(bucket, p=self._weights))
            ids[i] = doc
            actual[i] = min(self.doc_len[doc], int(want))
        return Assignment(doc_id=ids, prefix_tokens=actual)

    def structure_report(self, a: Assignment, input_tokens: np.ndarray) -> dict:
        """Constructed (intended) cache structure of an assignment."""
        frac = np.where(np.asarray(input_tokens) > 0,
                        a.prefix_tokens / np.maximum(input_tokens, 1), 0.0)
        used, counts = np.unique(a.doc_id[a.doc_id >= 0], return_counts=True)
        return {
            "constructed_fraction_p50": float(np.percentile(frac, 50)),
            "constructed_fraction_p95": float(np.percentile(frac, 95)),
            "distinct_docs_used": int(len(used)),
            "hottest_doc_share": float(counts.max() / counts.sum())
            if len(counts) else 0.0,
            "cold_first_uses": int(len(used)),  # one cold miss per distinct doc
        }
