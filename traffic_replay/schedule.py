"""Burst scheduler: spiky arrivals, not a flat rate.

Two-state modulated Poisson process:
  BASE state:  rate around qps_base
  BURST state: rate around qps_burst
State dwell times are exponential; within each second, arrivals are Poisson
at the state's rate and uniformly placed inside the second.

Emits absolute timestamps (seconds from run start). `rate_scale` thins the
schedule uniformly at random, preserving SHAPE while lowering volume, which
is how the same schedule serves both a laptop smoke test and a full run.
`shard i/n` deterministically splits a schedule across client processes.
"""
from __future__ import annotations

import math

import numpy as np


def make_schedule(duration_s: int = 300, qps_base: float = 25.0,
                  qps_burst: float = 350.0, qps_min: float = 10.0,
                  qps_max: float = 500.0, mean_base_dwell_s: float = 20.0,
                  mean_burst_dwell_s: float = 6.0, rate_scale: float = 1.0,
                  seed: int = 23) -> dict:
    if not isinstance(duration_s, int) or isinstance(duration_s, bool) \
            or duration_s <= 0:
        raise ValueError("duration_s must be a positive integer")
    numeric = {
        "qps_base": qps_base, "qps_burst": qps_burst,
        "qps_min": qps_min, "qps_max": qps_max,
        "mean_base_dwell_s": mean_base_dwell_s,
        "mean_burst_dwell_s": mean_burst_dwell_s,
        "rate_scale": rate_scale,
    }
    for name, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if qps_min > qps_max:
        raise ValueError("qps_min cannot exceed qps_max")
    if not qps_min <= qps_base <= qps_max:
        raise ValueError("qps_base must be between qps_min and qps_max")
    if not qps_min <= qps_burst <= qps_max:
        raise ValueError("qps_burst must be between qps_min and qps_max")
    if not (0 < rate_scale <= 1.0):
        raise ValueError("rate_scale must be in (0, 1]")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) \
            or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    rates = np.empty(duration_s)
    t, state = 0, "base"
    while t < duration_s:
        dwell = max(1, int(rng.exponential(
            mean_base_dwell_s if state == "base" else mean_burst_dwell_s)))
        end = min(duration_s, t + dwell)
        if state == "base":
            r = np.clip(rng.normal(qps_base, qps_base * 0.35), qps_min, qps_max)
        else:
            r = np.clip(rng.normal(qps_burst, qps_burst * 0.30), qps_min, qps_max)
        rates[t:end] = np.clip(r * rng.normal(1.0, 0.08, end - t),
                               qps_min, qps_max)
        t, state = end, ("burst" if state == "base" else "base")

    # Generate the full schedule first, then deterministically thin it. Runs
    # with the same seed at lower scales are exact subsets of the full run,
    # which makes smoke/full comparisons preserve individual arrival times.
    full_counts = rng.poisson(rates)
    if full_counts.sum() == 0:
        counts = np.zeros(duration_s, dtype=int)
        return {"rates": rates * rate_scale, "counts": counts,
                "timestamps": np.array([])}
    full_ts = np.concatenate([i + np.sort(rng.uniform(0, 1, c))
                              for i, c in enumerate(full_counts) if c > 0])
    keep = rng.random(len(full_ts)) < rate_scale
    ts = full_ts[keep]
    counts = np.bincount(ts.astype(int), minlength=duration_s)
    return {"rates": rates * rate_scale, "counts": counts,
            "timestamps": np.sort(ts)}


def load_trace(path, duration_cap_s: float | None = None) -> dict:
    """Replace the synthetic schedule with a real arrival trace.

    Accepts a file of arrival timestamps in seconds, one per line (plain
    text or JSONL with a `t` field). Timestamps are shifted to start at 0
    and sorted. This is the bring-your-own-trace path: the customer's
    production arrival log becomes the schedule, and every downstream
    stage (sizing, cache construction, measurement) is unchanged.
    """
    import json as _json
    from pathlib import Path as _Path

    if duration_cap_s is not None and (
            isinstance(duration_cap_s, bool)
            or not isinstance(duration_cap_s, (int, float))
            or not math.isfinite(float(duration_cap_s))
            or duration_cap_s < 0):
        raise ValueError("duration_cap_s must be non-negative and finite")
    ts = []
    for line_number, raw_line in enumerate(
            _Path(path).read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            if line.startswith("{"):
                value = _json.loads(line)
                if not isinstance(value, dict) or "t" not in value:
                    raise ValueError("JSON row must be an object with a t field")
                timestamp = float(value["t"])
            else:
                timestamp = float(line)
        except (KeyError, TypeError, ValueError, _json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid arrival timestamp at {path}:{line_number}: {exc}") \
                from exc
        if not math.isfinite(timestamp):
            raise ValueError(
                f"arrival timestamp at {path}:{line_number} must be finite")
        ts.append(timestamp)
    if not ts:
        raise ValueError(f"no timestamps in {path}")
    arr = np.sort(np.asarray(ts, dtype=float))
    arr = arr - arr[0]
    if duration_cap_s is not None:
        arr = arr[arr <= duration_cap_s]
    dur = int(np.ceil(arr[-1])) + 1 if len(arr) else 0
    counts = np.bincount(arr.astype(int), minlength=dur)
    return {"rates": counts.astype(float), "counts": counts,
            "timestamps": arr, "source": str(path)}


def shard(schedule: dict, index: int, total: int) -> dict:
    """Deterministic 1-of-n split, retaining global workload indices."""
    if not isinstance(total, int) or total <= 0 or not isinstance(index, int) \
            or not (0 <= index < total):
        raise ValueError("need 0 <= index < total")
    ts = schedule["timestamps"]
    existing = np.asarray(schedule.get("global_indices",
                                       np.arange(len(ts))), dtype=int)
    if len(existing) != len(ts):
        raise ValueError("global_indices must align with timestamps")
    # rates and counts describe the WHOLE run. passing them through unchanged
    # made a shard's own summary.json report the unsharded request count, so
    # anyone opening it read a shortfall that was not there.
    chosen = np.arange(index, len(ts), total)
    return {**schedule, "timestamps": ts[chosen],
            "global_indices": existing[chosen],
            "total_requests": int(schedule.get("total_requests", len(ts))),
            "shard": (index, total)}


def schedule_report(sched: dict) -> dict:
    r = np.asarray(sched["rates"])
    if r.size == 0:
        return {"seconds": 0, "requests": 0,
                "source": sched.get("source", "synthetic")}
    sh = sched.get("shard")
    n_req = (len(sched["timestamps"]) if sh
             else int(np.asarray(sched["counts"]).sum()))
    out_extra = {}
    if sh:
        out_extra = {
            "shard": f"{sh[0] + 1}/{sh[1]}",
            "rates_describe": ("the whole run, not this shard. this shard "
                               f"takes 1 arrival in {sh[1]}"),
        }
    return {
        **out_extra,
        "seconds": int(len(r)),
        "requests": n_req,
        "rate_min": float(r.min()),
        "rate_p50": float(np.percentile(r, 50)),
        "rate_p95": float(np.percentile(r, 95)),
        "rate_max": float(r.max()),
        "spiky": bool(r.max() / max(r.min(), 1e-9) >= 8.0),
        "source": sched.get("source", "synthetic"),
    }
