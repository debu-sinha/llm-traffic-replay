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
import os
from pathlib import Path
import stat

import numpy as np

from .json_input import loads_strict


# The current scheduler and profile materializer are intentionally exact but
# O(N). Fail closed before they can allocate an unbounded schedule. Supporting
# larger runs requires a streaming scheduler/workload implementation, not an
# undocumented memory gamble.
MAX_SCHEDULE_REQUESTS = 1_000_000
MAX_SCHEDULE_SECONDS = 604_800

# Every current report/merge path computes exact percentiles from materialized
# request dictionaries.  Keep that distinct from the scheduler's looser
# numerical-array ceiling: a million timestamps are manageable, while a
# million decoded journal rows are not.  Raising this limit requires a
# bounded-memory/streaming statistics implementation and corresponding
# resource tests, not just a larger integer.
MAX_EXACT_ANALYSIS_REQUEST_ROWS = 50_000
MAX_TIMESTAMP_TRACE_BYTES = 16 * 1024 * 1024
MAX_TIMESTAMP_TRACE_LINE_BYTES = 64 * 1024

# Sizing uses a homogeneous Poisson ceiling, whose realized row count varies
# around duration * QPS.  Generated CLI configs reserve eight standard
# deviations below the remaining exact-analysis budget; the concrete seeded
# schedule is still counted and rejected before credentials/network if it
# exceeds the hard envelope.  This is headroom for a usable default, never a
# replacement for the exact gate.
SIZING_CEILING_POISSON_HEADROOM_STDDEVS = 8.0


def validate_schedule_capacity(duration_s: int, qps_max: float) -> None:
    if duration_s > MAX_SCHEDULE_SECONDS:
        raise ValueError(
            f"duration_s exceeds the {MAX_SCHEDULE_SECONDS}-second exact "
            "scheduler limit")
    projected = float(duration_s) * float(qps_max)
    if projected > MAX_SCHEDULE_REQUESTS:
        raise ValueError(
            f"schedule can project up to {projected:,.0f} arrivals, above "
            f"the exact scheduler limit of {MAX_SCHEDULE_REQUESTS:,}; lower "
            "duration/rate or implement a streaming schedule")


def validate_exact_analysis_capacity(*, replay_rows: int,
                                     calibration_rows: int = 0,
                                     sizing_rows: int = 0,
                                     setup_rows: int = 0,
                                     context: str = "run") -> int:
    """Fail closed before traffic can exceed exact-analysis memory bounds."""
    values = {
        "replay": replay_rows,
        "calibration": calibration_rows,
        "sizing": sizing_rows,
        "setup": setup_rows,
    }
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{context} {label} row count must be a non-negative integer")
    total = sum(values.values())
    if total > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
        breakdown = ", ".join(
            f"{label}={value:,}" for label, value in values.items() if value)
        raise ValueError(
            f"{context} requires {total:,} request rows ({breakdown or 'none'}), "
            f"above the exact-analysis resource envelope of "
            f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}; lower duration/rate, "
            "calibration/setup traffic, or sweep rungs, or implement bounded "
            "streaming statistics")
    return total


def exact_analysis_replay_budget(*, calibration_rows: int = 0,
                                 sizing_rows: int = 0,
                                 setup_rows: int = 0,
                                 context: str = "run") -> int:
    """Return rows left for replay after all other exact populations.

    Keeping this arithmetic next to the hard validator prevents CLI defaults
    from carrying a second, drifting idea of the resource limit.
    """
    used = validate_exact_analysis_capacity(
        replay_rows=0,
        calibration_rows=calibration_rows,
        sizing_rows=sizing_rows,
        setup_rows=setup_rows,
        context=context,
    )
    remaining = MAX_EXACT_ANALYSIS_REQUEST_ROWS - used
    if remaining <= 0:
        raise ValueError(
            f"{context} leaves no rows for measured replay inside the "
            f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}-row exact-analysis "
            "resource envelope")
    return remaining


def conservative_sizing_qps_ceiling(
        duration_s: int, *, calibration_rows: int = 0,
        sizing_rows: int = 0, setup_rows: int = 0,
        context: str = "sizing run") -> float:
    """Derive a usable Poisson QPS ceiling from the exact row envelope.

    If ``mu`` is the expected replay population, its standard deviation is
    ``sqrt(mu)``.  Solve ``mu + k*sqrt(mu) = available_rows`` for ``mu``, with
    the named eight-sigma headroom above.  Prevalidation subsequently counts
    the actual seeded schedule, so even an extreme draw fails before traffic.
    """
    if not isinstance(duration_s, int) or isinstance(duration_s, bool) \
            or duration_s <= 0:
        raise ValueError("duration_s must be a positive integer")
    available = exact_analysis_replay_budget(
        calibration_rows=calibration_rows,
        sizing_rows=sizing_rows,
        setup_rows=setup_rows,
        context=context,
    )
    k = SIZING_CEILING_POISSON_HEADROOM_STDDEVS
    root_mu = (math.sqrt(k * k + 4.0 * available) - k) / 2.0
    expected_rows = math.floor(root_mu * root_mu)
    if expected_rows < 1:
        raise ValueError(
            f"{context} exact-analysis budget is too small for one "
            "conservative sizing replay row")
    return expected_rows / duration_s


def make_schedule(duration_s: int = 300, qps_base: float = 25.0,
                  qps_burst: float = 350.0, qps_min: float = 10.0,
                  qps_max: float = 500.0, mean_base_dwell_s: float = 20.0,
                  mean_burst_dwell_s: float = 6.0, rate_scale: float = 1.0,
                  seed: int = 23, *,
                  request_limit: int = MAX_SCHEDULE_REQUESTS) -> dict:
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
    if not isinstance(request_limit, int) or isinstance(request_limit, bool) \
            or not 0 < request_limit <= MAX_SCHEDULE_REQUESTS:
        raise ValueError(
            f"request_limit must be an integer from 1 to "
            f"{MAX_SCHEDULE_REQUESTS:,}")
    validate_schedule_capacity(duration_s, qps_max)
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
    total = int(full_counts.sum())
    if total > request_limit:
        raise ValueError(
            f"sampled schedule contains {total:,} arrivals, above the exact "
            f"scheduler limit of {request_limit:,}")
    if total == 0:
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


def thin_schedule_ceiling(schedule: dict, fraction: float, *, seed: int) -> dict:
    """Derive an exact subset of a pretraffic sizing-rate ceiling schedule.

    A sizing pass learns its rate from endpoint latency and therefore cannot
    materialize that final schedule before paid traffic.  Prevalidation can,
    however, materialize the configured ``qps_max`` schedule.  Independent
    Bernoulli thinning produces the requested lower-rate Poisson schedule as
    an exact subset, so its row count can never exceed the already-approved
    ceiling.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) \
            or not math.isfinite(float(fraction)) \
            or not 0 < float(fraction) <= 1:
        raise ValueError("schedule ceiling fraction must be in (0, 1]")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) \
            or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    timestamps = np.asarray(schedule.get("timestamps"), dtype=float)
    rates = np.asarray(schedule.get("rates"), dtype=float)
    if timestamps.ndim != 1 or rates.ndim != 1:
        raise ValueError("schedule ceiling arrays must be one-dimensional")
    fraction = float(fraction)
    if fraction == 1.0:
        selected = timestamps.copy()
    else:
        keep = np.random.default_rng(seed).random(len(timestamps)) < fraction
        selected = timestamps[keep]
    counts = np.bincount(
        selected.astype(int), minlength=len(rates)).astype(int, copy=False)
    return {
        "rates": rates * fraction,
        "counts": counts,
        "timestamps": selected,
        "source": "sizing-derived subset of prevalidated qps_max ceiling",
    }


def _read_bounded_trace_lines(path, row_limit: int) -> list[str]:
    """Read a stable regular trace without blocking on special files."""
    source = Path(path)
    try:
        path_info = source.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect arrival trace {source}: {exc}") from exc
    if not stat.S_ISREG(path_info.st_mode):
        raise ValueError(f"arrival trace is not a regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"cannot read arrival trace {source}: {exc}") from exc
    lines: list[str] = []
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"arrival trace is not a regular file: {source}")
        if (path_info.st_dev, path_info.st_ino) != \
                (before.st_dev, before.st_ino):
            raise ValueError(f"arrival trace changed while opening: {source}")
        if before.st_size > MAX_TIMESTAMP_TRACE_BYTES:
            raise ValueError(
                f"arrival trace {source} declares {before.st_size:,} bytes, "
                f"above the {MAX_TIMESTAMP_TRACE_BYTES:,}-byte limit")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            size = 0
            while True:
                raw = handle.readline(MAX_TIMESTAMP_TRACE_LINE_BYTES + 1)
                if not raw:
                    break
                size += len(raw)
                if size > MAX_TIMESTAMP_TRACE_BYTES:
                    raise ValueError(
                        f"arrival trace exceeds the "
                        f"{MAX_TIMESTAMP_TRACE_BYTES:,}-byte limit: {source}")
                if len(raw) > MAX_TIMESTAMP_TRACE_LINE_BYTES:
                    raise ValueError(
                        f"arrival trace line {len(lines) + 1} exceeds the "
                        f"{MAX_TIMESTAMP_TRACE_LINE_BYTES:,}-byte limit: "
                        f"{source}")
                if len(lines) >= row_limit:
                    raise ValueError(
                        f"arrival trace exceeds the exact scheduler limit of "
                        f"{row_limit:,} physical lines")
                try:
                    lines.append(raw.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"arrival trace is not UTF-8 at line "
                        f"{len(lines) + 1}: {source}") from exc
            after = os.fstat(handle.fileno())
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise ValueError(
                f"arrival trace changed while it was being read: {source}")
        return lines
    finally:
        if fd >= 0:
            os.close(fd)


def load_trace(path, duration_cap_s: float | None = None, *,
               row_limit: int = MAX_SCHEDULE_REQUESTS) -> dict:
    """Replace the synthetic schedule with a real arrival trace.

    Accepts a file of arrival timestamps in seconds, one per line (plain
    text or JSONL with a `t` field). Timestamps are shifted to start at 0
    and sorted. This is the bring-your-own-trace path: the customer's
    production arrival log becomes the schedule, and every downstream
    stage (sizing, cache construction, measurement) is unchanged.
    """
    import json as _json

    if duration_cap_s is not None and (
            isinstance(duration_cap_s, bool)
            or not isinstance(duration_cap_s, (int, float))
            or not math.isfinite(float(duration_cap_s))
            or duration_cap_s < 0):
        raise ValueError("duration_cap_s must be non-negative and finite")
    if not isinstance(row_limit, int) or isinstance(row_limit, bool) \
            or not 0 < row_limit <= MAX_SCHEDULE_REQUESTS:
        raise ValueError(
            f"row_limit must be an integer from 1 to "
            f"{MAX_SCHEDULE_REQUESTS:,}")
    ts = []
    for line_number, raw_line in enumerate(
            _read_bounded_trace_lines(path, row_limit), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            if line.startswith("{"):
                value = loads_strict(line)
                if not isinstance(value, dict) or "t" not in value:
                    raise ValueError("JSON row must be an object with a t field")
                raw_timestamp = value["t"]
                if isinstance(raw_timestamp, bool) or not isinstance(
                        raw_timestamp, (int, float)):
                    raise ValueError("JSON t field must be a number")
                timestamp = float(raw_timestamp)
            else:
                timestamp = float(line)
        except (KeyError, TypeError, ValueError, OverflowError,
                _json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid arrival timestamp at {path}:{line_number}: {exc}") \
                from exc
        if not math.isfinite(timestamp):
            raise ValueError(
                f"arrival timestamp at {path}:{line_number} must be finite")
        ts.append(timestamp)
        if len(ts) > row_limit:
            raise ValueError(
                f"arrival trace exceeds the exact scheduler limit of "
                f"{row_limit:,} rows")
    if not ts:
        raise ValueError(f"no timestamps in {path}")
    arr = np.sort(np.asarray(ts, dtype=float))
    arr = arr - arr[0]
    if duration_cap_s is not None:
        arr = arr[arr <= duration_cap_s]
    # One bucket covers each interval [second, second + 1).  ``ceil(max) + 1``
    # creates a phantom trailing bucket whenever the last timestamp is not an
    # integer (for example, a trace ending at 1.2 seconds needs buckets 0 and
    # 1, not an empty bucket 2).  The integer part plus one is the exact bucket
    # count for non-negative, zero-based timestamps.
    dur = int(math.floor(arr[-1])) + 1 if len(arr) else 0
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
