"""Traffic-profile validation and deterministic sampling.

Schema-v1 profiles retain the original P50/P95 closed-form sampler.  Schema
v2 adds two fidelity-preserving alternatives:

* ``quantile_cdf`` interpolates a complete, shared quantile ladder. Token
  counts interpolate in log space, cache fractions linearly, and the three
  marginal ranks are independent because a quantile ladder contains no
  evidence about their joint dependence.
* ``empirical_joint`` samples content-free observed triples in balanced,
  weighted cycles. It preserves the observed combinations and their
  correlation instead of inventing combinations from independent marginals.

Profiles are plain JSON files (see ``configs/``). All schema-v2 sampling
fields are closed schemas: unknown keys and lossy numeric coercions fail at
load time, before an endpoint can be called.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Z95 = 1.6448536269514722  # standard normal 95th percentile
_SCHEMA_VERSIONS = {1, 2}
_QUANTILE_CDF_KEYS = {
    "mode", "probabilities", "input_tokens", "output_tokens",
    "cache_fraction",
}
_EMPIRICAL_JOINT_KEYS = {"mode", "rows"}
_EMPIRICAL_ROW_KEYS = {
    "input_tokens", "output_tokens", "cache_fraction", "weight",
}
# A malformed profile must not be able to allocate an unbounded expanded
# cycle before the requested sample size is considered. Five million entries
# is already far larger than the normal benchmark workload while keeping the
# exact-cycle algorithm practical.
MAX_EMPIRICAL_CYCLE_WEIGHT = 5_000_000
_INT64_MAX = int(np.iinfo(np.int64).max)


def _number(value, where: str) -> float:
    """Return a finite JSON number without accepting booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _integer(value, where: str, *, positive: bool = False) -> int:
    """Return a strict integer; floats such as 1.0 are intentionally invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return result


def _unknown_keys(value: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{where} is missing key(s): {', '.join(missing)}")


def _weighted_inverted_cdf(rows: list[dict], field_name: str,
                           probability: float) -> float:
    """Exact empirical inverse CDF without expanding integer weights."""
    ordered = sorted((float(row[field_name]), int(row["weight"]))
                     for row in rows)
    total = sum(weight for _, weight in ordered)
    # ``inverted_cdf`` selects the first observation whose cumulative
    # probability reaches q. This always returns an actually observed value.
    rank = max(0, math.ceil(probability * total) - 1)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative > rank:
            return value
    raise AssertionError("validated empirical weights produced no quantile")


def _validate_quantile_cdf(profile: "Profile", sampling: dict) -> dict:
    _unknown_keys(sampling, _QUANTILE_CDF_KEYS,
                  "profile.sampling (quantile_cdf)")
    probabilities_raw = sampling["probabilities"]
    if not isinstance(probabilities_raw, list):
        raise ValueError("profile.sampling.probabilities must be an array")
    if len(probabilities_raw) < 2:
        raise ValueError(
            "profile.sampling.probabilities needs at least two knots")
    probabilities = [
        _number(value, f"profile.sampling.probabilities[{index}]")
        for index, value in enumerate(probabilities_raw)
    ]
    if any(not 0.0 < value < 1.0 for value in probabilities):
        raise ValueError(
            "profile.sampling.probabilities must be strictly between 0 and 1")
    if any(right <= left for left, right in
           zip(probabilities, probabilities[1:])):
        raise ValueError(
            "profile.sampling.probabilities must be strictly increasing")
    for required in (0.5, 0.95):
        if required not in probabilities:
            raise ValueError(
                "profile.sampling.probabilities must include exact 0.5 and "
                "0.95 legacy-anchor knots")

    normalized: dict[str, object] = {
        "mode": "quantile_cdf", "probabilities": probabilities,
    }
    for field_name in ("input_tokens", "output_tokens", "cache_fraction"):
        values_raw = sampling[field_name]
        if not isinstance(values_raw, list):
            raise ValueError(
                f"profile.sampling.{field_name} must be an array")
        if len(values_raw) != len(probabilities):
            raise ValueError(
                f"profile.sampling.{field_name} must have exactly "
                f"{len(probabilities)} values")
        values = [
            _number(value, f"profile.sampling.{field_name}[{index}]")
            for index, value in enumerate(values_raw)
        ]
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError(
                f"profile.sampling.{field_name} must be nondecreasing")
        if field_name == "cache_fraction":
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(
                    "profile.sampling.cache_fraction values must be between "
                    "0 and 1")
        elif any(value <= 0.0 for value in values):
            raise ValueError(
                f"profile.sampling.{field_name} values must be positive")

        p50_index = probabilities.index(0.5)
        p95_index = probabilities.index(0.95)
        anchors = getattr(profile, field_name)
        if values[p50_index] != anchors["p50"] \
                or values[p95_index] != anchors["p95"]:
            raise ValueError(
                f"profile.sampling.{field_name} must exactly match the "
                "legacy p50 and p95 anchors at probabilities 0.5 and 0.95")
        normalized[field_name] = values
    return normalized


def _validate_empirical_joint(profile: "Profile", sampling: dict) -> dict:
    _unknown_keys(sampling, _EMPIRICAL_JOINT_KEYS,
                  "profile.sampling (empirical_joint)")
    rows_raw = sampling["rows"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError(
            "profile.sampling.rows must be a non-empty array")

    rows = []
    seen: set[tuple[int, int, float]] = set()
    total_weight = 0
    for index, row in enumerate(rows_raw):
        where = f"profile.sampling.rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where} must be an object")
        _unknown_keys(row, _EMPIRICAL_ROW_KEYS, where)
        input_tokens = _integer(
            row["input_tokens"], f"{where}.input_tokens", positive=True)
        output_tokens = _integer(
            row["output_tokens"], f"{where}.output_tokens", positive=True)
        if input_tokens > _INT64_MAX or output_tokens > _INT64_MAX:
            raise ValueError(f"{where} token counts must fit signed 64-bit integers")
        cache_fraction = _number(
            row["cache_fraction"], f"{where}.cache_fraction")
        if not 0.0 <= cache_fraction <= 1.0:
            raise ValueError(
                f"{where}.cache_fraction must be between 0 and 1")
        weight = _integer(row["weight"], f"{where}.weight", positive=True)
        triple = (input_tokens, output_tokens, cache_fraction)
        if triple in seen:
            raise ValueError(
                f"{where} duplicates an earlier empirical triple; combine "
                "duplicates into its integer weight")
        seen.add(triple)
        total_weight += weight
        if total_weight > MAX_EMPIRICAL_CYCLE_WEIGHT:
            raise ValueError(
                "profile.sampling empirical cycle weight exceeds "
                f"{MAX_EMPIRICAL_CYCLE_WEIGHT}")
        rows.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_fraction": cache_fraction,
            "weight": weight,
        })

    # Canonical row ordering means semantically identical profiles yield the
    # same fixed-seed schedule regardless of source-log order.
    rows.sort(key=lambda row: (
        row["input_tokens"], row["output_tokens"], row["cache_fraction"]))
    for field_name in ("input_tokens", "output_tokens", "cache_fraction"):
        anchors = getattr(profile, field_name)
        expected_p50 = _weighted_inverted_cdf(rows, field_name, 0.5)
        expected_p95 = _weighted_inverted_cdf(rows, field_name, 0.95)
        if anchors["p50"] != expected_p50 or anchors["p95"] != expected_p95:
            raise ValueError(
                f"profile {field_name} p50/p95 must equal the empirical "
                "inverted-CDF anchors derived from sampling.rows")
    return {"mode": "empirical_joint", "rows": rows}


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
    schema_version: int = 1
    sampling: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be a non-empty string")
        for name in ("provenance", "label"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"profile {name} must be a string")
        if not isinstance(self.extra, dict):
            raise ValueError("profile extra fields must form an object")
        if isinstance(self.schema_version, bool) \
                or not isinstance(self.schema_version, (int, np.integer)):
            raise ValueError("profile schema_version must be an integer")
        self.schema_version = int(self.schema_version)
        if self.schema_version not in _SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported profile schema_version {self.schema_version}")
        for field_name in ("input_tokens", "output_tokens", "cache_fraction"):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or set(value) != {"p50", "p95"}:
                raise ValueError(
                    f"{field_name} must contain exactly p50 and p95")
            if any(isinstance(value[q], bool)
                   or not isinstance(value[q], (int, float))
                   for q in ("p50", "p95")):
                raise ValueError(f"{field_name} quantiles must be numbers")
            try:
                p50, p95 = float(value["p50"]), float(value["p95"])
            except OverflowError as exc:
                raise ValueError(
                    f"{field_name} quantiles must be finite") from exc
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
        if self.schema_version == 1:
            if self.sampling is not None:
                raise ValueError(
                    "profile sampling requires schema_version 2")
        else:
            if not isinstance(self.sampling, dict):
                raise ValueError(
                    "schema_version 2 profile sampling must be an object")
            mode = self.sampling.get("mode")
            if mode == "quantile_cdf":
                self.sampling = _validate_quantile_cdf(self, self.sampling)
            elif mode == "empirical_joint":
                self.sampling = _validate_empirical_joint(self, self.sampling)
            else:
                raise ValueError(
                    "profile.sampling.mode must be 'quantile_cdf' or "
                    "'empirical_joint'")

    @classmethod
    def from_json(cls, path: str | Path) -> "Profile":
        def object_without_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(
                        f"profile JSON contains duplicate key {key!r}")
                result[key] = value
            return result

        try:
            raw = json.loads(
                Path(path).read_text(encoding="utf-8-sig"),
                object_pairs_hook=object_without_duplicates)
        except json.JSONDecodeError as exc:
            raise ValueError(f"profile JSON is invalid: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("profile JSON must be an object")
        missing = [k for k in
                   ("name", "input_tokens", "output_tokens", "cache_fraction")
                   if k not in raw]
        if missing:
            raise ValueError("profile is missing required field(s): "
                             + ", ".join(missing))
        known = {k: raw[k] for k in
                 ("name", "input_tokens", "output_tokens", "cache_fraction",
                  "schema_version", "sampling")
                 if k in raw}
        return cls(
            **known,
            provenance=raw.get("provenance", "unspecified"),
            label=raw.get("label", ""),
            extra={k: v for k, v in raw.items()
                   if k not in (*known, "provenance", "label")},
        )


def _interpolate_quantile_cdf(probabilities: np.ndarray,
                              values: np.ndarray,
                              ranks: np.ndarray, *,
                              logarithmic: bool) -> np.ndarray:
    """Invert a piecewise CDF with explicit clamped tails.

    Exact knot ranks are restored from ``values`` after log interpolation so
    floating-point ``exp(log(x))`` cannot perturb an authoritative knot.
    """
    transformed = np.log(values) if logarithmic else values
    result = np.interp(
        ranks, probabilities, transformed,
        left=transformed[0], right=transformed[-1])
    if logarithmic:
        result = np.exp(result)
    result[ranks <= probabilities[0]] = values[0]
    result[ranks >= probabilities[-1]] = values[-1]
    for probability, value in zip(probabilities, values):
        result[ranks == probability] = value
    return result


def _validate_sample_controls(n: int, seed: int, min_input: int,
                              max_input: int, min_output: int,
                              max_output: int) -> None:
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
    if max_input > _INT64_MAX or max_output > _INT64_MAX:
        raise ValueError("sampler token bounds must fit signed 64-bit integers")


def _finish_draw(inp: np.ndarray, out: np.ndarray, cache_f: np.ndarray,
                 params: dict, clipping: dict) -> dict:
    prefix = np.rint(inp * cache_f).astype(np.int64)
    suffix = inp - prefix
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_target_fraction": cache_f,
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "params": params,
        "clipping": clipping,
    }


def _sample_quantile_cdf(profile: Profile, n: int, rng,
                         min_input: int, max_input: int,
                         min_output: int, max_output: int) -> dict:
    assert profile.sampling is not None
    sampling = profile.sampling
    probabilities = np.asarray(sampling["probabilities"], dtype=float)
    input_values = np.asarray(sampling["input_tokens"], dtype=float)
    output_values = np.asarray(sampling["output_tokens"], dtype=float)
    cache_values = np.asarray(sampling["cache_fraction"], dtype=float)
    if input_values[0] < min_input or input_values[-1] > max_input:
        raise ValueError(
            "input-token quantile ladder falls outside sampler bounds "
            f"{min_input}..{max_input}")
    if output_values[0] < min_output or output_values[-1] > max_output:
        raise ValueError(
            "output-token quantile ladder falls outside sampler bounds "
            f"{min_output}..{max_output}")

    # Stratification bounds finite-run quantile drift to one rank interval.
    # Each marginal independently shuffles the same evenly spaced rank set;
    # reusing one ordering would fabricate perfect cross-field correlation.
    base_ranks = ((np.arange(n, dtype=float) + 0.5) / n
                  if n else np.empty(0, dtype=float))
    input_raw = _interpolate_quantile_cdf(
        probabilities, input_values, rng.permutation(base_ranks),
        logarithmic=True)
    output_raw = _interpolate_quantile_cdf(
        probabilities, output_values, rng.permutation(base_ranks),
        logarithmic=True)
    cache_f = _interpolate_quantile_cdf(
        probabilities, cache_values, rng.permutation(base_ranks),
        logarithmic=False)
    inp = np.rint(input_raw).astype(np.int64)
    out = np.rint(output_raw).astype(np.int64)
    return _finish_draw(inp, out, cache_f, {
        "mode": "quantile_cdf",
        "dependence": "independent_marginals",
        "rank_sampling": "independently_shuffled_stratified",
        "tail_policy": "clamp_to_end_knots",
        "input_interpolation": "log",
        "output_interpolation": "log",
        "cache_interpolation": "linear",
        "probabilities": list(sampling["probabilities"]),
    }, {
        "input_below_min": 0, "input_above_max": 0,
        "output_below_min": 0, "output_above_max": 0,
        "input_bounds": (min_input, max_input),
        "output_bounds": (min_output, max_output),
    })


def _sample_empirical_joint(profile: Profile, n: int, rng,
                            min_input: int, max_input: int,
                            min_output: int, max_output: int) -> dict:
    assert profile.sampling is not None
    rows = profile.sampling["rows"]
    input_values = np.asarray(
        [row["input_tokens"] for row in rows], dtype=np.int64)
    output_values = np.asarray(
        [row["output_tokens"] for row in rows], dtype=np.int64)
    cache_values = np.asarray(
        [row["cache_fraction"] for row in rows], dtype=float)
    weights = np.asarray([row["weight"] for row in rows], dtype=np.int64)
    if input_values.min() < min_input or input_values.max() > max_input:
        raise ValueError(
            "empirical input-token rows fall outside sampler bounds "
            f"{min_input}..{max_input}")
    if output_values.min() < min_output or output_values.max() > max_output:
        raise ValueError(
            "empirical output-token rows fall outside sampler bounds "
            f"{min_output}..{max_output}")

    base_cycle = np.repeat(np.arange(len(rows), dtype=np.int64), weights)
    selected = np.empty(n, dtype=np.int64)
    offset = 0
    while offset < n:
        shuffled = rng.permutation(base_cycle)
        take = min(len(shuffled), n - offset)
        selected[offset:offset + take] = shuffled[:take]
        offset += take
    inp = input_values[selected]
    out = output_values[selected]
    cache_f = cache_values[selected]
    return _finish_draw(inp, out, cache_f, {
        "mode": "empirical_joint",
        "dependence": "observed_joint_triples",
        "sampling": "balanced_weighted_cycles",
        "quantile_method": "inverted_cdf",
        "unique_rows": len(rows),
        "cycle_weight": int(weights.sum()),
    }, {
        "input_below_min": 0, "input_above_max": 0,
        "output_below_min": 0, "output_above_max": 0,
        "input_bounds": (min_input, max_input),
        "output_bounds": (min_output, max_output),
    })


def sample(profile: Profile, n: int, seed: int = 7,
           min_input: int = 1, max_input: int = 200_000,
           min_output: int = 1, max_output: int = 8_192) -> dict:
    """Draw n requests from the profile. Returns dict of numpy arrays.

    prefix_tokens is the per-request number of input tokens INTENDED to be
    served from prompt cache; suffix_tokens is the unique remainder.
    """
    _validate_sample_controls(
        n, seed, min_input, max_input, min_output, max_output)
    rng = np.random.default_rng(seed)
    if profile.schema_version == 2:
        assert profile.sampling is not None
        if profile.sampling["mode"] == "quantile_cdf":
            return _sample_quantile_cdf(
                profile, n, rng, min_input, max_input,
                min_output, max_output)
        return _sample_empirical_joint(
            profile, n, rng, min_input, max_input, min_output, max_output)

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

    return _finish_draw(inp, out, cache_f, {
        "input": (mu_i, sg_i), "output": (mu_o, sg_o),
        "cache": (mu_c, sg_c),
        "cache_family": ("clipped_normal" if boundary_cache
                         else "logit_normal"),
    }, {
        "input_below_min": int(np.sum(inp_raw < min_input)),
        "input_above_max": int(np.sum(inp_raw > max_input)),
        "output_below_min": int(np.sum(out_raw < min_output)),
        "output_above_max": int(np.sum(out_raw > max_output)),
        "input_bounds": (min_input, max_input),
        "output_bounds": (min_output, max_output),
    })


def quantile_report(draw: dict) -> dict:
    """Recovered quantiles of a draw, for comparison against the spec."""
    params = draw.get("params", {})
    quantile_method = params.get("quantile_method")

    def q(a, p):
        if len(a) == 0:
            raise ValueError("cannot report quantiles for an empty draw")
        # Empirical-joint anchors are discrete inverse-CDF values.  Linear
        # interpolation can report a token count or cache fraction that was
        # never observed and can disagree with a profile that passed anchor
        # validation.  Other samplers retain NumPy's historical linear method.
        if quantile_method == "inverted_cdf":
            return float(np.percentile(a, p, method="inverted_cdf"))
        return float(np.percentile(a, p))

    probabilities = params.get("probabilities")
    if probabilities is None:
        probabilities = [0.5, 0.95]

    def report(values):
        result = {}
        for probability in probabilities:
            percentage = probability * 100.0
            label_number = (str(int(percentage)) if percentage.is_integer()
                            else format(percentage, ".12g"))
            result[f"p{label_number}"] = q(values, percentage)
        return result

    return {
        "input_tokens": report(draw["input_tokens"]),
        "output_tokens": report(draw["output_tokens"]),
        "cache_fraction": report(draw["cache_target_fraction"]),
    }
