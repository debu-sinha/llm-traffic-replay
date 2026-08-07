"""Schedule must be genuinely spiky, span the configured range, respect
rate_scale, and shard deterministically."""
import math

import numpy as np
import pytest

from traffic_replay.schedule import (MAX_SCHEDULE_REQUESTS, make_schedule,
                                     schedule_report, shard)


def test_shape_spans_range_and_is_spiky():
    s = make_schedule(duration_s=300, seed=23)
    r = schedule_report(s)
    assert r["spiky"] is True
    assert r["rate_min"] >= 10.0 - 1e-9
    assert r["rate_max"] <= 500.0 + 1e-9
    assert r["rate_max"] > 150  # bursts actually happen
    assert r["requests"] > 5_000


def test_timestamps_sorted_within_duration():
    s = make_schedule(duration_s=120, seed=5)
    ts = s["timestamps"]
    assert (np.diff(ts) >= 0).all()
    assert ts.min() >= 0 and ts.max() <= 120


def test_rate_scale_thins_volume_preserving_shape():
    full = make_schedule(duration_s=200, seed=7, rate_scale=1.0)
    thin = make_schedule(duration_s=200, seed=7, rate_scale=0.05)
    n_full = len(full["timestamps"])
    n_thin = len(thin["timestamps"])
    assert 0.02 < n_thin / n_full < 0.10  # ~5% with Poisson noise
    # shape preserved: same underlying rate curve up to the scale factor
    assert np.allclose(thin["rates"] * 20, full["rates"], rtol=1e-9)
    # It is actual thinning, not a fresh Poisson draw: every reduced-rate
    # arrival is one of the exact full-run arrivals.
    assert set(thin["timestamps"]).issubset(set(full["timestamps"]))


@pytest.mark.parametrize("kwargs", [
    {"duration_s": 0},
    {"duration_s": 1.5},
    {"qps_base": math.nan},
    {"qps_min": 20, "qps_max": 10},
    {"qps_base": 5, "qps_min": 10},
    {"qps_burst": 501, "qps_max": 500},
    {"mean_base_dwell_s": 0},
    {"rate_scale": True},
    {"seed": -1},
])
def test_invalid_schedule_parameters_fail_before_allocation(kwargs):
    with pytest.raises(ValueError):
        make_schedule(**kwargs)


def test_schedule_projection_is_bounded_before_large_arrays_are_allocated():
    with pytest.raises(ValueError, match="exact scheduler limit"):
        make_schedule(duration_s=300, qps_base=1_000_000,
                      qps_burst=1_000_000, qps_min=1_000_000,
                      qps_max=1_000_000)
    assert MAX_SCHEDULE_REQUESTS == 1_000_000


def test_shard_partitions_exactly():
    s = make_schedule(duration_s=60, seed=11)
    sharded = [shard(s, i, 3) for i in range(3)]
    parts = [part["timestamps"] for part in sharded]
    together = np.sort(np.concatenate(parts))
    assert np.array_equal(together, s["timestamps"])
    assert abs(len(parts[0]) - len(parts[1])) <= 1
    indices = np.concatenate([part["global_indices"] for part in sharded])
    assert np.array_equal(np.sort(indices), np.arange(len(s["timestamps"])))
    assert all(part["total_requests"] == len(s["timestamps"])
               for part in sharded)


def test_load_trace_replaces_synthetic(tmp_path_factory=None):
    import tempfile
    from pathlib import Path
    from traffic_replay.schedule import load_trace
    d = Path(tempfile.mkdtemp())
    # plain-text timestamps, unsorted, non-zero-based
    (d / "trace.txt").write_text("\n".join(
        str(t) for t in [100.5, 100.1, 103.0, 101.7, 102.2]))
    s = load_trace(d / "trace.txt")
    ts = s["timestamps"]
    assert ts[0] == 0.0                      # shifted to start at zero
    assert (np.diff(ts) >= 0).all()          # sorted
    assert len(ts) == 5
    # JSONL form with duration cap
    (d / "trace.jsonl").write_text("\n".join(
        f'{{"t": {t}}}' for t in [10.0, 11.0, 12.0, 40.0]))
    s2 = load_trace(d / "trace.jsonl", duration_cap_s=5.0)
    assert len(s2["timestamps"]) == 3        # the 40s arrival capped out


@pytest.mark.parametrize("content", [
    "nan\n", "inf\n", '{"missing": 1}\n', '{"t": "bad"}\n', "{bad}\n",
])
def test_invalid_trace_rows_have_context_and_never_reach_numpy(content,
                                                                tmp_path):
    from traffic_replay.schedule import load_trace

    path = tmp_path / "bad.trace"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"bad\.trace:1"):
        load_trace(path)


def test_trace_json_rejects_duplicate_timestamp_keys(tmp_path):
    from traffic_replay.schedule import load_trace

    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"t":1,"t":999}\n')
    with pytest.raises(ValueError, match="duplicate key 't'"):
        load_trace(path)


@pytest.mark.parametrize("cap", [-1, math.nan, math.inf, True])
def test_invalid_trace_duration_cap_is_rejected(cap, tmp_path):
    from traffic_replay.schedule import load_trace

    path = tmp_path / "trace.txt"
    path.write_text("1\n")
    with pytest.raises(ValueError, match="duration_cap_s"):
        load_trace(path, duration_cap_s=cap)
