"""Traffic profile sampler.

Turns stated quantiles (P50/P95) into per-request draws of
(input_tokens, output_tokens, cache_target_fraction) using closed-form fits:

  token counts        -> lognormal fitted to (P50, P95)
  cache hit fraction  -> logit-normal fitted to (P50, P95), bounded in (0, 1)

Why closed form: two quantiles determine a two-parameter distribution
exactly, the fit is reproducible with no optimizer, and the sampled
population provably recovers the stated quantiles (see tests/test_profile.py).

Profiles are plain JSON files (see configs/), so a customer-supplied dataset
replaces a spoken estimate by dropping in a new config, nothing else changes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Z95 = 1.6448536269514722  # standard normal 95th percentile


def lognormal_from_quantiles(p50: float, p95: float) -> tuple[float, float]:
    """Return (mu, sigma) of the lognormal with the given median and p95."""
    if not (p95 > p50 > 0):
        raise ValueError(f"need p95 > p50 > 0, got p50={p50}, p95={p95}")
    mu = math.log(p50)
    sigma = math.log(p95 / p50) / Z95
    return mu, sigma


def logitnormal_from_quantiles(p50: float, p95: float) -> tuple[float, float]:
    """Return (mu, sigma) on the logit scale for the given quantiles."""
    if not (0.0 < p50 < p95 < 1.0):
        raise ValueError(f"need 0 < p50 < p95 < 1, got p50={p50}, p95={p95}")

    def logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    mu = logit(p50)
    sigma = (logit(p95) - mu) / Z95
    return mu, sigma


@dataclass
class Profile:
    """A traffic profile: quantile specs plus provenance."""

    name: str
    input_tokens: dict          # {"p50": .., "p95": ..}
    output_tokens: dict         # {"p50": .., "p95": ..}
    cache_fraction: dict        # {"p50": .., "p95": ..} in (0, 1)
    provenance: str = "unspecified"
    label: str = ""             # e.g. "ASSUMPTION: built to spoken figures"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "Profile":
        raw = json.loads(Path(path).read_text())
        known = {k: raw[k] for k in
                 ("name", "input_tokens", "output_tokens", "cache_fraction")
                 if k in raw}
        return cls(
            **known,
            provenance=raw.get("provenance", "unspecified"),
            label=raw.get("label", ""),
            extra={k: v for k, v in raw.items()
                   if k not in (*known, "provenance", "label")},
        )


def sample(profile: Profile, n: int, seed: int = 7,
           min_input: int = 64, max_input: int = 200_000,
           min_output: int = 1, max_output: int = 8_192) -> dict:
    """Draw n requests from the profile. Returns dict of numpy arrays.

    prefix_tokens is the per-request number of input tokens INTENDED to be
    served from prompt cache; suffix_tokens is the unique remainder.
    """
    rng = np.random.default_rng(seed)

    mu_i, sg_i = lognormal_from_quantiles(**profile.input_tokens)
    mu_o, sg_o = lognormal_from_quantiles(**profile.output_tokens)
    mu_c, sg_c = logitnormal_from_quantiles(**profile.cache_fraction)

    inp = np.clip(rng.lognormal(mu_i, sg_i, n).round(),
                  min_input, max_input).astype(int)
    out = np.clip(rng.lognormal(mu_o, sg_o, n).round(),
                  min_output, max_output).astype(int)
    cache_f = 1.0 / (1.0 + np.exp(-rng.normal(mu_c, sg_c, n)))

    prefix = np.round(inp * cache_f).astype(int)
    suffix = inp - prefix

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_target_fraction": cache_f,
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "params": {"input": (mu_i, sg_i), "output": (mu_o, sg_o),
                   "cache": (mu_c, sg_c)},
    }


def quantile_report(draw: dict) -> dict:
    """Recovered quantiles of a draw, for comparison against the spec."""
    def q(a, p):
        return float(np.percentile(a, p))

    return {
        "input_tokens": {"p50": q(draw["input_tokens"], 50),
                         "p95": q(draw["input_tokens"], 95)},
        "output_tokens": {"p50": q(draw["output_tokens"], 50),
                          "p95": q(draw["output_tokens"], 95)},
        "cache_fraction": {"p50": q(draw["cache_target_fraction"], 50),
                           "p95": q(draw["cache_target_fraction"], 95)},
    }
