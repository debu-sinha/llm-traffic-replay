"""Traffic profile sampler.

Turns stated quantiles (P50/P95) into per-request draws of
(input_tokens, output_tokens, cache_target_fraction) using closed-form fits:

  token counts        -> lognormal fitted to (P50, P95)
  cached prompt-token fraction -> logit-normal fitted to (P50, P95), bounded in (0, 1)

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
    if not (p95 >= p50 > 0):
        raise ValueError(f"need p95 >= p50 > 0, got p50={p50}, p95={p95}")
    mu = math.log(p50)
    sigma = math.log(p95 / p50) / Z95
    return mu, sigma


def logitnormal_from_quantiles(p50: float, p95: float) -> tuple[float, float]:
    """Return (mu, sigma) on the logit scale for the given quantiles."""
    if not (0.0 <= p50 <= p95 <= 1.0):
        raise ValueError(f"need 0 <= p50 <= p95 <= 1, got p50={p50}, p95={p95}")
    # A point mass is a legitimate distribution. In particular, real logs
    # commonly contain no cached tokens at all. Represent boundary point
    # masses with infinite logits; sample() handles sigma=0 without sending
    # those infinities through exp().
    if p50 == p95:
        if p50 == 0.0:
            return -math.inf, 0.0
        if p50 == 1.0:
            return math.inf, 0.0
    elif p50 == 0.0 or p95 == 1.0:
        raise ValueError(
            "a non-constant logit-normal needs 0 < p50 < p95 < 1; "
            f"got p50={p50}, p95={p95}")

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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be a non-empty string")
        for name in ("provenance", "label"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"profile {name} must be a string")
        if not isinstance(self.extra, dict):
            raise ValueError("profile extra fields must form an object")
        for field_name in ("input_tokens", "output_tokens", "cache_fraction"):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or set(value) != {"p50", "p95"}:
                raise ValueError(
                    f"{field_name} must contain exactly p50 and p95")
            if any(isinstance(value[q], bool)
                   or not isinstance(value[q], (int, float))
                   for q in ("p50", "p95")):
                raise ValueError(f"{field_name} quantiles must be numbers")
            p50, p95 = float(value["p50"]), float(value["p95"])
            if not (math.isfinite(p50) and math.isfinite(p95)):
                raise ValueError(f"{field_name} quantiles must be finite")
            # Normalize once. Downstream comparisons and **kwargs must never
            # see a numeric-looking string or provider-specific number type.
            setattr(self, field_name, {"p50": p50, "p95": p95})
        lognormal_from_quantiles(
            float(self.input_tokens["p50"]), float(self.input_tokens["p95"]))
        lognormal_from_quantiles(
            float(self.output_tokens["p50"]), float(self.output_tokens["p95"]))
        cp50 = float(self.cache_fraction["p50"])
        cp95 = float(self.cache_fraction["p95"])
        if not (0.0 <= cp50 <= cp95 <= 1.0):
            raise ValueError(
                f"need 0 <= p50 <= p95 <= 1, got p50={cp50}, p95={cp95}")
        if "acceptance_targets" in self.extra:
            from .config_validation import validate_acceptance_targets
            validate_acceptance_targets(
                self.extra["acceptance_targets"],
                "profile.acceptance_targets")

    @classmethod
    def from_json(cls, path: str | Path) -> "Profile":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("profile JSON must be an object")
        missing = [k for k in
                   ("name", "input_tokens", "output_tokens", "cache_fraction")
                   if k not in raw]
        if missing:
            raise ValueError("profile is missing required field(s): "
                             + ", ".join(missing))
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
           min_input: int = 1, max_input: int = 200_000,
           min_output: int = 1, max_output: int = 8_192) -> dict:
    """Draw n requests from the profile. Returns dict of numpy arrays.

    prefix_tokens is the per-request number of input tokens INTENDED to be
    served from prompt cache; suffix_tokens is the unique remainder.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) \
            or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool)
           for x in (min_input, max_input)) or not (0 < min_input <= max_input):
        raise ValueError("need 0 < min_input <= max_input")
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool)
           for x in (min_output, max_output)) \
            or not (0 < min_output <= max_output):
        raise ValueError("need 0 < min_output <= max_output")
    rng = np.random.default_rng(seed)

    mu_i, sg_i = lognormal_from_quantiles(**profile.input_tokens)
    mu_o, sg_o = lognormal_from_quantiles(**profile.output_tokens)
    if profile.input_tokens["p50"] < min_input \
            or profile.input_tokens["p95"] > max_input:
        raise ValueError(
            "input-token profile quantiles fall outside sampler bounds "
            f"{min_input}..{max_input}")
    if profile.output_tokens["p50"] < min_output \
            or profile.output_tokens["p95"] > max_output:
        raise ValueError(
            "output-token profile quantiles fall outside sampler bounds "
            f"{min_output}..{max_output}")

    cp50 = float(profile.cache_fraction["p50"])
    cp95 = float(profile.cache_fraction["p95"])
    boundary_cache = cp50 != cp95 and (cp50 == 0.0 or cp95 == 1.0)
    if boundary_cache:
        # A clipped normal supplies the required boundary point mass while
        # still recovering both stated quantiles. A pure logit-normal cannot
        # have an exact quantile at zero or one.
        mu_c, sg_c = cp50, (cp95 - cp50) / Z95
    else:
        mu_c, sg_c = logitnormal_from_quantiles(cp50, cp95)

    inp_raw = rng.lognormal(mu_i, sg_i, n).round()
    out_raw = rng.lognormal(mu_o, sg_o, n).round()
    inp = np.clip(inp_raw, min_input, max_input).astype(int)
    out = np.clip(out_raw, min_output, max_output).astype(int)
    if boundary_cache:
        cache_f = np.clip(rng.normal(mu_c, sg_c, n), 0.0, 1.0)
    elif sg_c == 0.0:
        if mu_c == -math.inf:
            cache_f = np.zeros(n, dtype=float)
        elif mu_c == math.inf:
            cache_f = np.ones(n, dtype=float)
        else:
            cache_f = np.full(n, 1.0 / (1.0 + math.exp(-mu_c)), dtype=float)
    else:
        latent = np.clip(rng.normal(mu_c, sg_c, n), -709.0, 709.0)
        cache_f = 1.0 / (1.0 + np.exp(-latent))

    prefix = np.round(inp * cache_f).astype(int)
    suffix = inp - prefix

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_target_fraction": cache_f,
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "params": {"input": (mu_i, sg_i), "output": (mu_o, sg_o),
                   "cache": (mu_c, sg_c),
                   "cache_family": ("clipped_normal" if boundary_cache
                                    else "logit_normal")},
        "clipping": {
            "input_below_min": int(np.sum(inp_raw < min_input)),
            "input_above_max": int(np.sum(inp_raw > max_input)),
            "output_below_min": int(np.sum(out_raw < min_output)),
            "output_above_max": int(np.sum(out_raw > max_output)),
            "input_bounds": (min_input, max_input),
            "output_bounds": (min_output, max_output),
        },
    }


def quantile_report(draw: dict) -> dict:
    """Recovered quantiles of a draw, for comparison against the spec."""
    def q(a, p):
        if len(a) == 0:
            raise ValueError("cannot report quantiles for an empty draw")
        return float(np.percentile(a, p))

    return {
        "input_tokens": {"p50": q(draw["input_tokens"], 50),
                         "p95": q(draw["input_tokens"], 95)},
        "output_tokens": {"p50": q(draw["output_tokens"], 50),
                          "p95": q(draw["output_tokens"], 95)},
        "cache_fraction": {"p50": q(draw["cache_target_fraction"], 50),
                           "p95": q(draw["cache_target_fraction"], 95)},
    }
