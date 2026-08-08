"""compare tabulates several runs one column each and warns in bold when their
achieved cache p50 differ by more than 0.10 (the fake-comparison trap)."""
import hashlib
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from traffic_replay import aggregate
from traffic_replay.aggregate import compare_runs, verify_comparison_output


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="compare-"))


@pytest.fixture(autouse=True)
def _reconstructible_comparison_source(monkeypatch):
    monkeypatch.setattr(aggregate, "snapshot_source_state", lambda _path: {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source_tree_sha256": "f" * 64,
        "source_files": [],
    })


def _summary(title, cache_p50):
    def tab(p50):
        return {"p50": p50, "p90": p50 * 1.2, "p95": p50 * 1.3,
                "p99": p50 * 1.6, "n": 100}
    return {
        "run": {"title": title, "input_mode": "profile"}, "error_rate": 0.0,
        "requests_total": 1,
        "ttft_ms": tab(400), "e2e_ms": tab(800), "interchunk_max_ms": tab(6),
        "achieved_cache_fraction": {"p50": cache_p50, "p95": cache_p50 + 0.05},
        "throughput": {"input_tokens_per_min": 1_000_000,
                       "output_tokens_per_min": 5000},
        "token_targeting": {
            "status": "verified",
            "warning": None,
            "input_coverage": 1.0,
            "output_coverage": 1.0,
            "input_reported_over_intended": {"p50": 1.0, "p95": 1.0},
            "output_reported_over_intended": {"p50": 1.0, "p95": 1.0},
        },
        "cache_fidelity": {
            "status": "verified", "warning": None, "coverage": 1.0,
        },
        "latency_population": {
            "kind": "readable_answers", "n": 400, "warning": None,
        },
        "answers": {"answer_rate": 1.0},
        "arrivals": {"dispatch_lag_ms": {"p95": 8.0}},
        # a clean baseline for every comparability check except cache, so the
        # cache tests below isolate the thing they name
        "harness_version": "0.3.0",
        "latency_basis": "send-to-first-token; connection excluded",
        "schedule": {"seconds": 120, "requests": 1200,
                     "rate_min": 10.0, "rate_p50": 10.0,
                     "rate_p95": 10.0, "rate_max": 10.0,
                     "source": "synthetic"},
        "sample": {"n": 400, "warning": None},
        "drift": {"drift_flag": False, "drift_kind": "stable"},
    }


def _write_completion_marker(d: Path) -> None:
    manifest = json.loads((d / "manifest.json").read_text())
    manifest_raw = (d / "manifest.json").read_bytes()
    request_rows = manifest["artifacts"]["requests.jsonl"]["row_count"]
    (d / ".traffic-replay-complete").write_text(json.dumps({
        "artifact_id": manifest["artifact_id"],
        "status": "complete",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_bytes": len(manifest_raw),
        "request_rows": request_rows,
    }) + "\n")


def _replace_manifest(d: Path, manifest: dict) -> None:
    (d / "manifest.json").write_text(json.dumps(manifest))
    _write_completion_marker(d)


def _seal(d: Path, manifest: dict) -> None:
    summary_raw = (d / "summary.json").read_bytes()
    requests = d / "requests.jsonl"
    requests.write_text(json.dumps({
        "phase": "replay",
        "status": 200,
        "ok": True,
        "request_id": "request-0",
    }, sort_keys=True) + "\n")
    requests_raw = requests.read_bytes()
    manifest.update({
        "workload_id": manifest.get("workload_id", "workload-test"),
        "logical_run_id": manifest.get("logical_run_id", f"logical-{d.name}"),
        "run_id": manifest.get("logical_run_id", f"logical-{d.name}"),
        "execution_id": manifest.get("execution_id", f"execution-{d.name}"),
        "artifact_id": manifest.get("artifact_id", f"artifact-{d.name}"),
        "artifacts": {
            "summary.json": {
                "sha256": hashlib.sha256(summary_raw).hexdigest(),
                "bytes": len(summary_raw),
            },
            "requests.jsonl": {
                "sha256": hashlib.sha256(requests_raw).hexdigest(),
                "bytes": len(requests_raw),
                "row_count": 1,
            },
        },
    })
    (d / "manifest.json").write_text(json.dumps(manifest))
    _write_completion_marker(d)


def _replace_requests(d: Path, rows: list[dict]) -> None:
    raw = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    (d / "requests.jsonl").write_bytes(raw)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["artifacts"]["requests.jsonl"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": len(rows),
    }
    _replace_manifest(d, manifest)


def _add_sealed_source_report(d: Path) -> None:
    raw = b"<!doctype html><title>sealed source report</title>\n"
    (d / "report.html").write_bytes(raw)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["artifacts"]["report.html"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    _replace_manifest(d, manifest)


def _compare(caches):
    base = _tmp()
    dirs = []
    for i, c in enumerate(caches):
        d = base / f"r{i}"
        d.mkdir(parents=True, exist_ok=True)
        sm = _summary(f"prov{i}", c)
        (d / "summary.json").write_text(json.dumps(sm))
        _seal(d, _manifest(sm))
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


def test_self_contained_html_names_first_input_baseline_and_binds_both_reports():
    base = _tmp()
    dirs = _comparison_inputs(base)
    for source in dirs:
        _add_sealed_source_report(source)

    out = compare_runs(base / "comparison", dirs)
    report = (out / "comparison.html").read_text()
    manifest = json.loads((out / "manifest.json").read_text())

    assert report.startswith("<!doctype html>")
    assert report.index("VALID COMPARISON") < report.index("Compatibility matrix")
    assert "Baseline:&#x27;" not in report
    assert "<strong>Baseline:</strong> run-0 (first input)" in report
    assert "Baseline is explicitly the first input" in report
    assert "<table class='compat'>" in report
    assert "<caption>" in report
    assert "scope='col'" in report and "scope='row'" in report
    assert "lower preferred; untested" in report and "context only" in report
    assert "not statistically demonstrated improvements or regressions" in report
    assert "UNSEALED PRINT/PDF DERIVATIVE" in report
    assert "internal hashes are not a digital signature" in report
    assert "Baseline absolute" in report
    assert "Candidate absolute" in report
    assert "Absolute delta" in report and "Percent delta" in report
    assert "<dt>Artifact ID</dt><dd><code>artifact-input-0</code></dd>" \
        in report
    assert "<dt>UTC window</dt><dd>not recorded</dd>" in report
    assert "<dt>Deployment context</dt><dd>not recorded</dd>" in report
    assert "<dt>Sample count</dt><dd>400</dd>" in report
    assert "href='../input-0/report.html'" in report
    assert "href='../input-1/report.html'" in report
    assert report.index("How to read this report") < report.index(
        "Absolute values and deltas")
    assert report.count("tabindex='0' role='region'") == 2
    assert "aria-describedby='compatibility-scroll-hint'" in report
    assert "aria-describedby='metrics-scroll-hint'" in report
    assert "Scroll horizontally; the Dimension column stays visible." \
        in report
    assert "Scroll horizontally; the Metric column stays visible." in report
    assert ".table-wrap .sticky-col{position:sticky" in report
    assert ".table-wrap .sticky-col{position:static;box-shadow:none}" \
        in report
    assert "<script" not in report.lower()
    assert "<link" not in report.lower()
    assert "@import" not in report.lower()
    assert "url(" not in report.lower()
    assert "http://" not in report.lower()
    assert "https://" not in report.lower()

    assert set(manifest["artifacts"]) == {"comparison.md", "comparison.html"}
    for name in ("comparison.md", "comparison.html"):
        raw = (out / name).read_bytes()
        assert manifest["artifacts"][name] == {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    assert verify_comparison_output(out) == manifest


def test_comparison_source_cards_and_markdown_show_only_sealed_run_facts():
    base = _tmp()
    dirs = []
    digest = "9" * 64
    metadata = {
        "name": "glm-5-2-prod",
        "task": "llm/v1/chat",
        "route_optimized": False,
        "ready": "READY",
        "served_entities": [{
            "name": "glm-5-2",
            "entity_version": "7",
            "workload_type": "GPU_LARGE",
            "workload_size": "2x",
            "min_provisioned_throughput": 1200,
            "max_provisioned_throughput": 3600,
            "scale_to_zero_enabled": False,
        }],
    }
    for index, sample_n in enumerate((321, 654)):
        directory = base / f"source-{index}"
        directory.mkdir()
        summary = _summary(f"source-{index}", 0.60)
        summary["sample"]["n"] = sample_n
        summary["run"].update({
            "endpoint_base_url": "https://dbc.example.databricks.com",
            "endpoint_path": "/serving-endpoints/glm-5-2-prod/invocations",
            "endpoint_model": "glm-5-2-prod",
            "endpoint_metadata": metadata,
        })
        (directory / "summary.json").write_text(json.dumps(summary))
        _seal(directory, _manifest(
            summary,
            profile_sha256=digest,
            run_started_at_utc=(
                f"2027-01-15T08:0{index}:00Z"),
            run_ended_at_unix=1_800_000_060 + index * 60,
            endpoint_base_url="https://dbc.example.databricks.com",
            endpoint_path="/serving-endpoints/glm-5-2-prod/invocations",
            endpoint_model="glm-5-2-prod",
            endpoint_metadata=metadata,
        ))
        dirs.append(directory)

    out = compare_runs(base / "comparison", dirs)
    report = (out / "comparison.html").read_text()
    markdown = (out / "comparison.md").read_text()

    assert "2027-01-15T08:00:00Z → 2027-01-15T08:01:00Z" in report
    assert "2027-01-15T08:01:00Z → 2027-01-15T08:02:00Z" in report
    assert "route=https://dbc.example.databricks.com/serving-endpoints/" \
        "glm-5-2-prod/invocations; model=glm-5-2-prod" in report
    assert "endpoint=glm-5-2-prod; task=llm/v1/chat; route optimized=False; " \
        "ready=READY; served entity: name=glm-5-2, version=7, " \
        "workload=GPU_LARGE, size=2x, min throughput=1200, " \
        "max throughput=3600, scale to zero=False" in report
    assert f"<dt>Workload digest</dt><dd><code>{digest}</code></dd>" \
        in report
    assert "<dt>Sample count</dt><dd>321</dd>" in report
    assert "<dt>Sample count</dt><dd>654</dd>" in report
    assert "pay_per_token" not in report and "pay per token" not in report

    assert "## source runs" in markdown
    assert "- UTC window: 2027-01-15T08:00:00Z → " \
        "2027-01-15T08:01:00Z" in markdown
    assert f"- Workload digest: {digest}" in markdown
    assert "- Sample count: 321" in markdown
    assert "- Sample count: 654" in markdown


def test_valid_html_deltas_are_arithmetic_not_regression_verdicts():
    baseline = _summary("baseline", 0.60)
    candidate = _summary("candidate", 0.60)
    candidate["ttft_ms"]["p50"] = 300
    clean = [{"phase": "replay", "status": 200, "ok": True}]
    out, _md = _compare_with_rows([baseline, candidate], [clean, clean])

    report = (out / "comparison.html").read_text()

    assert "300.0 ms" in report
    assert "-100.0 ms" in report
    assert "-25.0%" in report
    assert "<td class='delta signal-change'>" in report
    assert "numerically preferred" in report
    assert "improved" not in report and "regressed" not in report
    assert "no repeat-run uncertainty or practical-effect threshold" in report
    assert "winner" not in report.lower()
    assert "fastest" not in report.lower()


def test_missing_endpoint_identity_and_request_evidence_qualify_comparison():
    base = _tmp()
    dirs = []
    for index in range(2):
        summary = _summary(f"unknown-{index}", 0.60)
        summary["requests_total"] = 0
        directory = base / f"input-{index}"
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(summary))
        manifest = _manifest(summary)
        manifest.pop("endpoint_model")
        _seal(directory, manifest)
        _replace_requests(directory, [])
        dirs.append(directory)

    out = compare_runs(base / "comparison", dirs)
    report = (out / "comparison.html").read_text()
    manifest = json.loads((out / "manifest.json").read_text())

    assert "QUALIFIED COMPARISON" in report
    assert "endpoint identity is not recorded" in report
    assert "no manifest-bound request rows are available" in report
    assert "direction withheld" in report
    assert manifest["comparison_state"] == "qualified"
    assert manifest["comparison_valid"] is False
    assert manifest["numeric_direction_labels_allowed"] is False


def test_warns_only_when_cache_gap_exceeds_threshold():
    assert "WARNING" not in _compare([0.60, 0.62, 0.65])   # gap 0.05
    wide = _compare([0.60, 0.60, 0.85])                    # gap 0.25
    assert "WARNING" in wide and "cache" in wide


def test_boundary_just_over_and_under():
    assert "WARNING" not in _compare([0.50, 0.60])   # gap exactly 0.10
    assert "WARNING" in _compare([0.50, 0.61])       # gap 0.11


def test_cache_mismatch_qualifies_and_neutralizes_comparison():
    base = _tmp()
    baseline = _summary("baseline", 0.50)
    candidate = _summary("candidate", 0.75)
    candidate["ttft_ms"]["p50"] = 300

    dirs = []
    for index, summary in enumerate((baseline, candidate)):
        d = base / f"input-{index}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps(summary))
        _seal(d, _manifest(summary))
        dirs.append(d)

    out = compare_runs(base / "comparison", dirs)
    md = (out / "comparison.md").read_text()
    report = (out / "comparison.html").read_text()
    manifest = json.loads((out / "manifest.json").read_text())

    reason = "cached prompt-token fraction p50 spans 0.500 to 0.750"
    assert "QUALIFIED COMPARISON" in md
    assert md.index(reason) < md.index("## TTFT (ms)")
    assert report.index("QUALIFIED COMPARISON") < report.index(reason)
    assert report.index(reason) < report.index("Absolute values and deltas")
    assert "All deltas are neutral diagnostic values" in report
    assert "<td class='delta signal-good'>" not in report
    assert "<td class='delta signal-bad'>" not in report
    assert "<span class='assessment'>improved</span>" not in report
    assert "<span class='assessment'>regressed</span>" not in report
    assert manifest["comparison_state"] == "qualified"
    assert manifest["comparison_valid"] is False
    assert manifest["directional_judgment_allowed"] is False
    assert manifest["numeric_direction_labels_allowed"] is False


def test_canonical_caution_qualifies_comparison():
    baseline = _summary("baseline", 0.60)
    candidate = _summary("candidate", 0.60)
    candidate["decision"] = {
        "measurement_validity": {
            "code": "CAUTION",
            "reason": "client delivery drift requires review",
        },
    }

    md = _compare_summaries([baseline, candidate])

    assert "QUALIFIED COMPARISON" in md
    assert "canonical measurement state is CAUTION" in md
    assert md.index("client delivery drift requires review") < md.index(
        "## TTFT (ms)")


def test_compare_missing_input_dir_gives_clean_error():
    base = _tmp()
    d = base / "r0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(_summary("p0", 0.60)))
    from traffic_replay.aggregate import compare_runs
    with pytest.raises(ValueError):
        compare_runs(base / "cmp", [d, base / "missing"])


def _manifest(summary, **overrides):
    count = int(summary["schedule"]["requests"])
    value = {
        "manifest_schema_version": 3,
        "git_commit": "a" * 40, "git_dirty": False,
        "harness_version": summary["harness_version"],
        "latency_basis": summary["latency_basis"],
        "input_mode": "profile", "profile_sha256": "b" * 64,
        "endpoint_model": "databricks-test-endpoint",
        "seed": 7, "request_params": {"temperature": 0.0,
                                       "max_output_tokens_cap": 512},
        "schedule": summary["schedule"],
        "shard": "1/1",
        "schedule_identity": {
            "encoding": "float64-le-seconds-from-run-start",
            "global_timestamps_sha256": "c" * 64,
            "global_count": count,
            "global_min_s": 0.0 if count else None,
            "global_max_s": float(count - 1) if count else None,
            "shard_timestamps_sha256": "c" * 64,
            "shard_count": count,
            "shard_min_s": 0.0 if count else None,
            "shard_max_s": float(count - 1) if count else None,
        },
        "index_identity": {
            "encoding": "int64-le",
            "global_indices_sha256": "d" * 64,
            "count": count,
            "min": 0 if count else None,
            "max": count - 1 if count else None,
            "global_count": count,
            "shard_index": 0,
            "shard_total": 1,
            "partition": "unsharded",
        },
    }
    value.update(overrides)
    return value


def _compare_summaries(summaries, manifest_overrides=None):
    """Compare arbitrary summary dicts, not just cache values."""
    base = _tmp()
    dirs = []
    for i, sm in enumerate(summaries):
        d = base / f"r{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(sm))
        override = (manifest_overrides or {}).get(i, {})
        _seal(d, _manifest(sm, **override))
        dirs.append(d)
    out = compare_runs(base / "cmp", dirs)
    return (out / "comparison.md").read_text()


def _compare_with_rows(summaries, rows_by_source):
    base = _tmp()
    dirs = []
    for i, (summary, rows) in enumerate(zip(summaries, rows_by_source)):
        d = base / f"r{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary))
        _seal(d, _manifest(summary))
        _replace_requests(d, rows)
        dirs.append(d)
    out = compare_runs(base / "cmp", dirs)
    return out, (out / "comparison.md").read_text()


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
    assert "| cached prompt-token fraction p50 | 0.568 | NOT REPORTED |" in md


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
    assert "p99 is indicative below 1000 requests" in md
    assert "not in steady state" in md and "warming" in md


def test_mixed_harness_versions_are_refused_as_like_for_like():
    a = _summary("old", 0.60)
    a["harness_version"] = "0.2.0"
    b = _summary("new", 0.60)
    b["harness_version"] = "0.3.0"
    md = _compare_summaries([a, b])
    assert "different harness versions" in md
    assert "TCP/TLS" in md


def test_clean_matched_runs_produce_no_warnings():
    a = _summary("a", 0.60)
    b = _summary("b", 0.62)
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
    a = _summary("prov-a", 0.0)
    b = _summary("prov-b", 0.0)
    for sm in (a, b):
        sm["achieved_cache_fraction"] = {"p50": None, "p95": None, "n": 0}
    md = _compare_summaries([a, b])
    assert "no run reported cached tokens" in md
    assert "biggest driver" in md


def test_a_failing_run_is_named_as_a_breaking_point_in_a_comparison():
    a = _summary("steady", 0.60)
    a["drift"] = {"drift_flag": False, "drift_kind": "stable"}
    b = _summary("broke", 0.60)
    b["drift"] = {"drift_flag": True, "drift_kind": "failing"}
    md = _compare_summaries([a, b])
    assert "broke was shedding requests" in md
    assert "is a breaking point" in md
    assert "its surviving percentiles" in md


def test_two_failing_runs_read_as_plural():
    a = _summary("broke-a", 0.60)
    b = _summary("broke-b", 0.60)
    for sm in (a, b):
        sm["drift"] = {"drift_flag": True, "drift_kind": "failing"}
    md = _compare_summaries([a, b])
    assert "were shedding requests" in md
    assert "are breaking points" in md
    assert "their surviving percentiles" in md


def test_different_workload_hashes_make_the_comparison_explicitly_invalid():
    a = _summary("a", 0.60)
    b = _summary("b", 0.60)
    md = _compare_summaries(
        [a, b], manifest_overrides={1: {"profile_sha256": "c" * 64}})
    assert "INVALID COMPARISON" in md
    assert "different profile or prompts SHA-256" in md
    assert md.index("INVALID COMPARISON") < md.index("## TTFT (ms)")


def test_dirty_source_or_different_request_params_invalidates_compare():
    a = _summary("a", 0.60)
    b = _summary("b", 0.60)
    md = _compare_summaries(
        [a, b], manifest_overrides={
            0: {"git_dirty": True},
            1: {"request_params": {"temperature": 1.0}}})
    assert "INVALID COMPARISON" in md
    assert "dirty or unknown Git state" in md
    assert "different request parameters" in md


def test_missing_manifest_is_rejected_as_untrusted_input():
    base = _tmp()
    dirs = []
    for i in range(2):
        d = base / f"r{i}"
        d.mkdir()
        sm = _summary(f"run-{i}", 0.60)
        (d / "summary.json").write_text(json.dumps(sm))
        if i == 0:
            _seal(d, _manifest(sm))
        else:
            (d / ".traffic-replay-complete").touch()
        dirs.append(d)
    with pytest.raises(ValueError, match="missing manifest.json"):
        compare_runs(base / "comparison", dirs)


def test_compare_never_treats_a_forced_invalid_aggregate_as_evidence():
    a = _summary("valid", 0.60)
    b = _summary("forced-merge", 0.60)
    b["run"]["aggregation_valid"] = False
    md = _compare_summaries([a, b])
    assert "INVALID COMPARISON" in md
    assert "explicitly INVALID aggregate" in md


def test_one_http_429_in_one_thousand_rows_invalidates_comparison():
    clean = _summary("clean", 0.60)
    limited = _summary("GLM 5.2 | customer", 0.60)
    for summary in (clean, limited):
        summary["schedule"]["requests"] = 1000
    rows = [
        {"phase": "replay", "status": 200, "ok": True,
         "request_id": f"request-{index}"}
        for index in range(1000)
    ]
    rows[731].update(status=429, ok=False)

    out, md = _compare_with_rows([clean, limited], [[], rows])

    assert "INVALID COMPARISON" in md
    assert "INCONCLUSIVE" in md
    assert "diagnostic-only" in md
    assert "GLM 5.2 &#124; customer" in md
    assert "1/1000 manifest-bound request rows returned HTTP 429" in md
    assert "phases: replay=1" in md
    assert "supports no endpoint-capacity conclusion" in md
    assert md.index("1/1000 manifest-bound") < md.index("## TTFT (ms)")
    assert "Comparability checks" not in md
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["comparison_valid"] is False


def test_invalid_html_is_diagnostic_only_escapes_input_and_neutralizes_deltas():
    payload = "candidate <img src='https://tracker.invalid/pixel'> | unsafe"
    baseline = _summary("baseline", 0.60)
    candidate = _summary(payload, 0.60)
    candidate["ttft_ms"]["p50"] = 300
    rows = [
        {"phase": "preflight", "status": 429, "ok": False},
        {"phase": "replay", "status": 200, "ok": True},
    ]

    out, _md = _compare_with_rows([baseline, candidate], [[], rows])
    report = (out / "comparison.html").read_text()

    assert report.index("INVALID COMPARISON") < report.index(
        "Compatibility matrix")
    assert "Diagnostic-only." in report
    assert "HTTP 429: <strong>1/2</strong>" in report
    assert "phases: preflight=1" in report
    assert "1/2 manifest-bound request rows returned HTTP 429" in report
    assert "candidate &lt;img src=&#x27;https://tracker.invalid/pixel&#x27;&gt;" \
        in report
    assert "<img" not in report.lower()
    assert "<script" not in report.lower()
    assert "<td class='delta signal-good'>" not in report
    assert "<td class='delta signal-bad'>" not in report
    assert "All deltas are neutral diagnostic values" in report
    assert "300.0 ms" in report and "-100.0 ms" in report
    assert "winner" not in report.lower()


def test_setup_phase_http_429s_cannot_hide_behind_clean_replay():
    a = _summary("baseline", 0.60)
    b = _summary("setup-limited", 0.60)
    rows = [
        {"phase": "preflight", "status": 429, "ok": False},
        {"phase": "calibration", "status": 429, "ok": False},
        {"phase": "replay", "status": 200, "ok": True},
        {"phase": "replay", "status": 200, "ok": True},
    ]

    _out, md = _compare_with_rows([a, b], [[], rows])

    assert "2/4 manifest-bound request rows returned HTTP 429" in md
    assert "phases: calibration=1, preflight=1" in md
    assert "INVALID COMPARISON" in md
    assert "Comparability checks" not in md


def test_authenticated_summary_429_is_invalid_even_if_journal_disagrees():
    a = _summary("baseline", 0.60)
    b = _summary("summary-limited", 0.60)
    b.update({
        "http_429_count": 1,
        "quota_limited": True,
        "http_429": {
            "count": 1,
            "request_rows_examined": 1000,
            "phases": {"probe": 1},
        },
    })

    md = _compare_summaries([a, b])

    assert "manifest-bound summary reports 1/1000 request rows" in md
    assert "phases: probe=1" in md
    assert "sealed journal contains no matching 429" in md
    assert "INVALID COMPARISON" in md
    assert "Comparability checks" not in md


@pytest.mark.parametrize("invalid_shape", [
    {"answers": {"invalid": "answer timing evidence is incomplete"}},
    {"measurement_valid": False},
    {"run": {"measurement_valid": False}},
    {"validity": {"valid": False, "status": "inconclusive"}},
])
def test_explicit_source_invalidity_is_diagnostic_only(invalid_shape):
    a = _summary("baseline", 0.60)
    b = _summary("invalid-source", 0.60)
    if "run" in invalid_shape:
        b["run"].update(invalid_shape["run"])
    else:
        b.update(invalid_shape)

    md = _compare_summaries([a, b])

    assert "INVALID COMPARISON" in md
    assert "diagnostic-only" in md
    assert "explicit" in md
    assert "Comparability checks" not in md
    assert md.index("INVALID COMPARISON") < md.index("## TTFT (ms)")


def test_customer_markdown_cannot_change_comparison_structure():
    payload = (
        "evil|column\n# injected <img src=https://tracker.invalid/pixel> "
        "`code` ![remote](https://tracker.invalid/image) "
        "[link](https://tracker.invalid/click)"
    )
    a = _summary("baseline", 0.60)
    b = _summary(payload, 0.60)
    b["drift"] = {
        "drift_flag": False,
        "drift_kind": None,
        "note": payload,
    }

    md = _compare_summaries(
        [a, b], manifest_overrides={
            1: {"request_params": {"customer_label": payload}},
        })

    header = next(
        line for line in md.splitlines()
        if line.startswith("| metric / quantile |"))
    assert header.count("|") == 4
    assert "evil&#124;column # injected" in header
    assert "<img" not in md
    assert not re.search(r"(?<!\\)!\[", md)
    assert not re.search(r"(?<!\\)\]\(", md)
    assert not any(line.startswith("# injected") for line in md.splitlines())
    assert "\\`code\\`" in md
    assert "&lt;img src=https://tracker.invalid/pixel&gt;" in md


def _comparison_inputs(base: Path) -> list[Path]:
    dirs = []
    for i in range(2):
        d = base / f"input-{i}"
        d.mkdir()
        sm = _summary(f"run-{i}", 0.60)
        (d / "summary.json").write_text(json.dumps(sm))
        _seal(d, _manifest(sm))
        dirs.append(d)
    return dirs


def test_compare_rejects_duplicate_input_directory_and_symlink_alias():
    base = _tmp()
    dirs = _comparison_inputs(base)
    with pytest.raises(ValueError, match="duplicate input run dir"):
        compare_runs(base / "same", [dirs[0], dirs[0]])
    alias = base / "alias"
    alias.symlink_to(dirs[0], target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate input run dir"):
        compare_runs(base / "alias-out", [dirs[0], alias])


@pytest.mark.parametrize("state", ["missing", "writing", "both"])
def test_compare_rejects_incomplete_or_writing_inputs(state):
    base = _tmp()
    dirs = _comparison_inputs(base)
    complete = dirs[1] / ".traffic-replay-complete"
    if state in ("missing", "writing"):
        complete.unlink()
    if state in ("writing", "both"):
        (dirs[1] / ".traffic-replay-writing").touch()
    match = "still being written" if state in ("writing", "both") \
        else "completion marker"
    with pytest.raises(ValueError, match=match):
        compare_runs(base / "out", dirs)


@pytest.mark.parametrize("field,value,match", [
    ("status", "writing", "status"),
    ("artifact_id", "artifact-copied", "artifact_id"),
    ("manifest_sha256", "0" * 64, "manifest SHA-256 mismatch"),
    ("manifest_bytes", 1, "manifest byte count mismatch"),
    ("request_rows", 2, "request_rows"),
])
def test_compare_rejects_an_unbound_completion_marker(field, value, match):
    base = _tmp()
    dirs = _comparison_inputs(base)
    marker_path = dirs[1] / ".traffic-replay-complete"
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match=match):
        compare_runs(base / "out", dirs)


def test_compare_rejects_an_empty_legacy_completion_marker():
    base = _tmp()
    dirs = _comparison_inputs(base)
    (dirs[1] / ".traffic-replay-complete").write_bytes(b"")
    with pytest.raises(ValueError, match="invalid completion marker"):
        compare_runs(base / "out", dirs)


@pytest.mark.parametrize("name,label", [
    ("manifest.json", "manifest.json"),
    (".traffic-replay-complete", "completion marker"),
])
def test_compare_rejects_duplicate_keys_in_evidence_envelopes(name, label):
    base = _tmp()
    dirs = _comparison_inputs(base)
    path = dirs[1] / name
    raw = path.read_text().rstrip()
    path.write_text(raw[:-1] + ',"artifact_id":"ambiguous"}\n')

    with pytest.raises(
            ValueError,
            match=rf"invalid {re.escape(label)} .*duplicate key 'artifact_id'"):
        compare_runs(base / "out", dirs)


def test_compare_rejects_duplicate_keys_in_authenticated_summary():
    base = _tmp()
    dirs = _comparison_inputs(base)
    path = dirs[1] / "summary.json"
    raw = path.read_text().rstrip()
    path.write_text(raw[:-1] + ',"run":{"title":"ambiguous"}}\n')
    changed = path.read_bytes()
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(changed).hexdigest(),
        "bytes": len(changed),
    }
    _replace_manifest(dirs[1], manifest)

    with pytest.raises(
            ValueError,
            match=r"invalid summary\.json .*duplicate key 'run'"):
        compare_runs(base / "out", dirs)


def test_compare_rejects_nonfinite_authenticated_summary_value():
    base = _tmp()
    dirs = _comparison_inputs(base)
    path = dirs[1] / "summary.json"
    summary = json.loads(path.read_text())
    summary["error_rate"] = float("nan")
    path.write_text(json.dumps(summary) + "\n")
    changed = path.read_bytes()
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(changed).hexdigest(),
        "bytes": len(changed),
    }
    _replace_manifest(dirs[1], manifest)

    with pytest.raises(
            ValueError,
            match=r"invalid summary\.json .*non-finite number"):
        compare_runs(base / "out", dirs)


def test_compare_rejects_unsupported_manifest_schema():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["manifest_schema_version"] = 999
    _replace_manifest(dirs[1], manifest)
    with pytest.raises(ValueError, match="unsupported manifest schema"):
        compare_runs(base / "out", dirs)


@pytest.mark.parametrize(
    "field", ["workload_id", "logical_run_id", "execution_id", "artifact_id"])
def test_compare_requires_v3_identity_fields(field):
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest[field] = None
    if field == "logical_run_id":
        manifest["run_id"] = None
    _replace_manifest(dirs[1], manifest)
    with pytest.raises(ValueError, match=field):
        compare_runs(base / "out", dirs)


def test_compare_requires_verified_summary_artifact_entry():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"].pop("summary.json")
    _replace_manifest(dirs[1], manifest)
    with pytest.raises(ValueError, match="summary.json"):
        compare_runs(base / "out", dirs)


def test_compare_verifies_artifact_hash_and_byte_metadata():
    base = _tmp()
    dirs = _comparison_inputs(base)
    path = dirs[1] / "summary.json"
    raw = path.read_bytes()
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    _replace_manifest(dirs[1], manifest)
    compare_runs(base / "valid", dirs)

    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compare_runs(base / "tampered", dirs)


def test_compare_rejects_a_copied_artifact_under_a_different_path():
    base = _tmp()
    dirs = _comparison_inputs(base)
    second_manifest = json.loads((dirs[1] / "manifest.json").read_text())
    first_manifest = json.loads((dirs[0] / "manifest.json").read_text())
    second_manifest["artifact_id"] = first_manifest["artifact_id"]
    _replace_manifest(dirs[1], second_manifest)
    with pytest.raises(ValueError, match="duplicate input artifact_id"):
        compare_runs(base / "out", dirs)


def test_compare_marks_different_exact_global_schedules_invalid():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["schedule_identity"]["global_timestamps_sha256"] = "e" * 64
    _replace_manifest(dirs[1], manifest)
    out = compare_runs(base / "out", dirs)
    md = (out / "comparison.md").read_text()
    assert "INVALID COMPARISON" in md
    assert "different arrival schedule" in md


@pytest.mark.parametrize("dirty", [True, None])
def test_dirty_or_unknown_generator_source_invalidates_comparison(
        monkeypatch, dirty):
    monkeypatch.setattr(aggregate, "snapshot_source_state", lambda _path: {
        "git_commit": "a" * 40,
        "git_dirty": dirty,
        "source_tree_sha256": "f" * 64,
        "source_files": [],
    })
    base = _tmp()
    out = compare_runs(base / "out", _comparison_inputs(base))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["comparison_valid"] is False
    assert manifest["generator_source_reconstructible"] is False
    report = (out / "comparison.md").read_text()
    assert "INVALID COMPARISON" in report
    assert "comparison generator has dirty or unknown Git state" in report
    assert "not reconstructible" in report


def test_compare_output_claim_is_repeated_and_concurrent_safe():
    base = _tmp()
    dirs = _comparison_inputs(base)
    requested = base / "comparison"
    first = compare_runs(requested, dirs)
    original = (first / "comparison.md").read_bytes()
    second = compare_runs(requested, dirs)
    assert first == requested
    assert second != first
    assert (first / "comparison.md").read_bytes() == original
    assert (first / ".traffic-replay-complete").is_file()
    assert not (first / ".traffic-replay-writing").exists()
    assert (second / ".traffic-replay-complete").is_file()

    manifest = verify_comparison_output(first)
    manifest_raw = (first / "manifest.json").read_bytes()
    completion = json.loads(
        (first / ".traffic-replay-complete").read_text())
    assert manifest["manifest_schema_version"] == 3
    assert manifest["artifact_type"] == "comparison"
    assert manifest["comparison_valid"] is True
    assert manifest["comparison_state"] == "valid"
    assert manifest["numeric_direction_labels_allowed"] is True
    assert manifest["directional_judgment_allowed"] is False
    assert manifest["generator_source_reconstructible"] is True
    assert manifest["artifact_id"] == completion["artifact_id"]
    assert completion["manifest_sha256"] == \
        hashlib.sha256(manifest_raw).hexdigest()
    assert completion["manifest_bytes"] == len(manifest_raw)
    report_raw = (first / "comparison.md").read_bytes()
    assert manifest["artifacts"]["comparison.md"] == {
        "sha256": hashlib.sha256(report_raw).hexdigest(),
        "bytes": len(report_raw),
    }
    html_raw = (first / "comparison.html").read_bytes()
    assert manifest["artifacts"]["comparison.html"] == {
        "sha256": hashlib.sha256(html_raw).hexdigest(),
        "bytes": len(html_raw),
    }
    assert [source["artifact_id"] for source in manifest["sources"]] == [
        json.loads((d / "manifest.json").read_text())["artifact_id"]
        for d in dirs
    ]
    for source, d in zip(manifest["sources"], dirs):
        source_manifest = (d / "manifest.json").read_bytes()
        source_summary = (d / "summary.json").read_bytes()
        assert source["manifest"] == {
            "sha256": hashlib.sha256(source_manifest).hexdigest(),
            "bytes": len(source_manifest),
        }
        assert source["summary"] == {
            "sha256": hashlib.sha256(source_summary).hexdigest(),
            "bytes": len(source_summary),
        }

    concurrent_target = base / "concurrent"
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(
            lambda _i: compare_runs(concurrent_target, dirs), range(4)))
    assert len(set(outputs)) == 4
    assert all((out / "comparison.md").is_file() for out in outputs)
    assert all((out / "comparison.html").is_file() for out in outputs)
    assert all((out / ".traffic-replay-complete").is_file()
               for out in outputs)
    assert all(verify_comparison_output(out) for out in outputs)


@pytest.mark.parametrize("name", ["comparison.md", "comparison.html"])
def test_comparison_verifier_detects_rendered_artifact_tampering(name):
    base = _tmp()
    out = compare_runs(base / "comparison", _comparison_inputs(base))
    report = out / name
    report.write_text(report.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_comparison_output(out)


def test_comparison_verifier_requires_html_as_a_first_class_artifact():
    base = _tmp()
    out = compare_runs(base / "comparison", _comparison_inputs(base))
    (out / "comparison.html").unlink()

    with pytest.raises(ValueError, match="missing comparison.html"):
        verify_comparison_output(out)


def test_comparison_verifier_requires_html_integrity_metadata():
    base = _tmp()
    out = compare_runs(base / "comparison", _comparison_inputs(base))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["artifacts"].pop("comparison.html")
    manifest_raw = (json.dumps(manifest) + "\n").encode()
    (out / "manifest.json").write_bytes(manifest_raw)
    completion = json.loads((out / ".traffic-replay-complete").read_text())
    completion["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    completion["manifest_bytes"] = len(manifest_raw)
    (out / ".traffic-replay-complete").write_text(json.dumps(completion) + "\n")

    with pytest.raises(ValueError, match="missing required artifact integrity"):
        verify_comparison_output(out)


@pytest.mark.parametrize("name,label", [
    ("manifest.json", "manifest.json"),
    (".traffic-replay-complete", "completion marker"),
])
def test_comparison_verifier_rejects_duplicate_envelope_keys(name, label):
    base = _tmp()
    out = compare_runs(base / "comparison", _comparison_inputs(base))
    path = out / name
    raw = path.read_text().rstrip()
    path.write_text(raw[:-1] + ',"artifact_type":"ambiguous"}\n')

    with pytest.raises(
            ValueError,
            match=rf"invalid {re.escape(label)} .*duplicate key 'artifact_type'"):
        verify_comparison_output(out)


def test_compare_cannot_claim_completion_before_manifest_is_durable(
        monkeypatch):
    base = _tmp()
    requested = base / "comparison"
    original = aggregate._atomic_compare_text

    def fail_manifest(dir_fd, name, value):
        if name == "manifest.json":
            raise OSError("injected manifest write failure")
        return original(dir_fd, name, value)

    monkeypatch.setattr(aggregate, "_atomic_compare_text", fail_manifest)
    with pytest.raises(OSError, match="injected manifest write failure"):
        compare_runs(requested, _comparison_inputs(base))
    assert (requested / ".traffic-replay-writing").is_file()
    assert not (requested / ".traffic-replay-complete").exists()
    assert not (requested / "manifest.json").exists()


def test_compare_never_follows_an_existing_output_symlink():
    base = _tmp()
    dirs = _comparison_inputs(base)
    victim = base / "victim"
    victim.mkdir()
    requested = base / "comparison"
    requested.symlink_to(victim, target_is_directory=True)
    out = compare_runs(requested, dirs)
    assert out != requested
    assert list(victim.iterdir()) == []
    assert (out / ".traffic-replay-complete").is_file()


def test_compare_supports_a_symlinked_parent_but_not_a_symlinked_leaf():
    base = _tmp()
    dirs = _comparison_inputs(base)
    real_parent = base / "real-parent"
    real_parent.mkdir()
    alias_parent = base / "parent-alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    requested = alias_parent / "comparison"
    out = compare_runs(requested, dirs)

    assert out == requested
    assert out.resolve().parent == real_parent.resolve()
    assert (out / ".traffic-replay-complete").is_file()
