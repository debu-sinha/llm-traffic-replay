"""Schedule must be genuinely spiky, span the configured range, respect
rate_scale, and shard deterministically."""
import numpy as np

from traffic_replay.schedule import make_schedule, schedule_report, shard


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


def test_shard_partitions_exactly():
    s = make_schedule(duration_s=60, seed=11)
    parts = [shard(s, i, 3)["timestamps"] for i in range(3)]
    together = np.sort(np.concatenate(parts))
    assert np.array_equal(together, s["timestamps"])
    assert abs(len(parts[0]) - len(parts[1])) <= 1


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
