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
        # a clean baseline for every comparability check except cache, so the
        # cache tests below isolate the thing they name
        "harness_version": "0.3.0",
        "sample": {"n": 400, "warning": None},
        "drift": {"drift_flag": False, "drift_kind": "stable"},
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
    assert "WARNING" in wide and "cache" in wide


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


def _compare_summaries(summaries):
    """Compare arbitrary summary dicts, not just cache values."""
    base = _tmp()
    dirs = []
    for i, sm in enumerate(summaries):
        d = base / f"r{i}"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(sm))
        dirs.append(d)
    out = compare_runs(base / "cmp", dirs)
    return (out / "comparison.md").read_text()


def test_a_provider_reporting_no_cache_at_all_is_warned_loudly():
    """The real case when putting Databricks next to a provider that does not
    report cached tokens. The old rule needed two cache values to compare, so
    a missing one silently produced a side-by-side of 57 percent cache against
    none, which is the most misleading table the tool can print."""
    a = _summary("databricks", 0.568)
    b = _summary("other-provider", 0.0)
    b["achieved_cache_fraction"] = {"p50": None, "p95": None, "n": 0,
                                    "source_fields": ["NOT REPORTED BY ENDPOINT"]}
    md = _compare_summaries([a, b])
    assert "WARNING" in md
    assert "did not report cached tokens" in md
    assert "may not be measuring the same work" in md
    assert "cache usage is unknown" in md          # not "they do not cache"
    # the disqualifier must appear before the first latency table
    assert md.index("did not report cached tokens") < md.index("## TTFT (ms)")
    # the cell itself must say why it is empty, not leave a bare dash
    assert "| achieved cache p50 | 0.568 | NOT REPORTED |" in md


def test_error_rate_is_warned_before_the_latency_tables():
    a = _summary("clean", 0.60)
    b = _summary("lossy", 0.60)
    b["error_rate"] = 0.104
    md = _compare_summaries([a, b])
    assert "failed requests" in md
    assert "10.4 percent" in md
    assert "survivorship" in md or "dropped its slowest" in md
    assert md.index("failed requests") < md.index("## TTFT (ms)")


def test_small_sample_and_drift_are_surfaced_in_a_comparison():
    a = _summary("steady", 0.60)
    a["sample"] = {"n": 400, "warning": None}
    a["drift"] = {"drift_flag": False, "drift_kind": "stable"}
    b = _summary("thin", 0.60)
    b["sample"] = {"n": 44, "warning": "small sample: p99 is unstable"}
    b["drift"] = {"drift_flag": True, "drift_kind": "warming"}
    md = _compare_summaries([a, b])
    assert "small samples" in md and "44 requests" in md
    assert "not in steady state" in md and "warming" in md


def test_mixed_harness_versions_are_refused_as_like_for_like():
    a = _summary("old", 0.60); a["harness_version"] = "0.2.0"
    b = _summary("new", 0.60); b["harness_version"] = "0.3.0"
    md = _compare_summaries([a, b])
    assert "different harness versions" in md
    assert "TCP/TLS" in md


def test_clean_matched_runs_produce_no_warnings():
    a = _summary("a", 0.60); b = _summary("b", 0.62)
    for sm in (a, b):
        sm["harness_version"] = "0.3.0"
        sm["sample"] = {"n": 400, "warning": None}
        sm["drift"] = {"drift_flag": False, "drift_kind": "stable"}
    md = _compare_summaries([a, b])
    assert "WARNING" not in md
    assert "Read this before the tables" not in md


def test_a_merged_run_reports_why_stability_was_never_established():
    """A merged run deliberately has no verdict. The compare warning must
    report that reason rather than claiming the run was too short."""
    a = _summary("single", 0.60)
    b = _summary("merged", 0.60)
    b["drift"] = {"windows": [], "note": "stability over time is not computed "
                                         "for a merged run."}
    md = _compare_summaries([a, b])
    assert "stability was never established" in md
    assert "not computed for a merged run" in md
    assert ".;" not in md


def test_no_run_reporting_cache_is_warned():
    """Two providers that both hide cached tokens is still an unverifiable
    comparison, and the old rule needed a reporting run to say anything."""
    a = _summary("prov-a", 0.0); b = _summary("prov-b", 0.0)
    for sm in (a, b):
        sm["achieved_cache_fraction"] = {"p50": None, "p95": None, "n": 0}
    md = _compare_summaries([a, b])
    assert "no run reported cached tokens" in md
    assert "biggest driver" in md
