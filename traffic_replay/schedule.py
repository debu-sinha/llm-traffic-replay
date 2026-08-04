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

import numpy as np


def make_schedule(duration_s: int = 300, qps_base: float = 25.0,
                  qps_burst: float = 350.0, qps_min: float = 10.0,
                  qps_max: float = 500.0, mean_base_dwell_s: float = 20.0,
                  mean_burst_dwell_s: float = 6.0, rate_scale: float = 1.0,
                  seed: int = 23) -> dict:
    if not (0 < rate_scale <= 1.0):
        raise ValueError("rate_scale must be in (0, 1]")
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

    counts = rng.poisson(rates * rate_scale)
    if counts.sum() == 0:
        return {"rates": rates * rate_scale, "counts": counts,
                "timestamps": np.array([])}
    ts = np.concatenate([i + np.sort(rng.uniform(0, 1, c))
                         for i, c in enumerate(counts) if c > 0])
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

    ts = []
    for line in _Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            ts.append(float(_json.loads(line)["t"]))
        else:
            ts.append(float(line))
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
    """Deterministic 1-of-n split for multi-process clients."""
    if not (0 <= index < total):
        raise ValueError("need 0 <= index < total")
    ts = schedule["timestamps"]
    # rates and counts describe the WHOLE run. passing them through unchanged
    # made a shard's own summary.json report the unsharded request count, so
    # anyone opening it read a shortfall that was not there.
    return {**schedule, "timestamps": ts[index::total],
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
