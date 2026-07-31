"""compare tabulates several runs one column each and warns in bold when their
achieved cache p50 differ by more than 0.10 (the fake-comparison trap)."""
import json
import tempfile
import pytest
from pathlib import Path
from traffic_replay.aggregate import compare_runs


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="compare-"))


def _summary(title, cache_p50):
    def tab(p50):
        return {"p50": p50, "p90": p50 * 1.2, "p95": p50 * 1.3,
                "p99": p50 * 1.6, "n": 100}
    return {
        "run": {"title": title}, "error_rate": 0.0,
        "ttft_ms": tab(400), "e2e_ms": tab(800), "interchunk_max_ms": tab(6),
        "achieved_cache_fraction": {"p50": cache_p50, "p95": cache_p50 + 0.05},
        "throughput": {"input_tokens_per_min": 1_000_000,
                       "output_tokens_per_min": 5000},
        "arrivals": {"dispatch_lag_ms": {"p95": 8.0}},
    }


def _compare(caches):
    base = _tmp()
    dirs = []
    for i, c in enumerate(caches):
        d = base / f"r{i}"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(_summary(f"prov{i}", c)))
        dirs.append(d)
    out = compare_runs(base / "cmp", dirs)
    return (out / "comparison.md").read_text()


def test_table_shape_and_columns():
    md = _compare([0.60, 0.62, 0.64])
    assert "## TTFT (ms)" in md and "## TTFG / E2E (ms)" in md
    assert "## interchunk max (ms)" in md
    assert "prov0" in md and "prov1" in md and "prov2" in md
    for q in ("p50", "p90", "p95", "p99"):
        assert f"| {q} |" in md


def test_warns_only_when_cache_gap_exceeds_threshold():
    assert "WARNING" not in _compare([0.60, 0.62, 0.65])   # gap 0.05
    wide = _compare([0.60, 0.60, 0.85])                    # gap 0.25
    assert "**WARNING" in wide and "cache" in wide


def test_boundary_just_over_and_under():
    assert "WARNING" not in _compare([0.50, 0.60])   # gap exactly 0.10
    assert "WARNING" in _compare([0.50, 0.61])       # gap 0.11


def test_compare_missing_input_dir_gives_clean_error():
    base = _tmp()
    d = base / "r0"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(_summary("p0", 0.60)))
    from traffic_replay.aggregate import compare_runs
    with pytest.raises(ValueError):
        compare_runs(base / "cmp", [d, base / "missing"])
