"""compare tabulates several runs one column each and warns in bold when their
achieved cache p50 differ by more than 0.10 (the fake-comparison trap)."""
import json
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
        "run": {"title": title, "input_mode": "profile"}, "error_rate": 0.0,
        "ttft_ms": tab(400), "e2e_ms": tab(800), "interchunk_max_ms": tab(6),
        "achieved_cache_fraction": {"p50": cache_p50, "p95": cache_p50 + 0.05},
        "throughput": {"input_tokens_per_min": 1_000_000,
                       "output_tokens_per_min": 5000},
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


def _seal(d: Path, manifest: dict) -> None:
    raw = (d / "summary.json").read_bytes()
    manifest.update({
        "workload_id": manifest.get("workload_id", "workload-test"),
        "logical_run_id": manifest.get("logical_run_id", f"logical-{d.name}"),
        "run_id": manifest.get("logical_run_id", f"logical-{d.name}"),
        "execution_id": manifest.get("execution_id", f"execution-{d.name}"),
        "artifact_id": manifest.get("artifact_id", f"artifact-{d.name}"),
        "artifacts": {
            "summary.json": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        },
    })
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / ".traffic-replay-complete").touch()


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


def test_warns_only_when_cache_gap_exceeds_threshold():
    assert "WARNING" not in _compare([0.60, 0.62, 0.65])   # gap 0.05
    wide = _compare([0.60, 0.60, 0.85])                    # gap 0.25
    assert "WARNING" in wide and "cache" in wide


def test_boundary_just_over_and_under():
    assert "WARNING" not in _compare([0.50, 0.60])   # gap exactly 0.10
    assert "WARNING" in _compare([0.50, 0.61])       # gap 0.11


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


def test_compare_rejects_unsupported_manifest_schema():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["manifest_schema_version"] = 999
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
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
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        compare_runs(base / "out", dirs)


def test_compare_requires_verified_summary_artifact_entry():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"] = {}
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="summary.json"):
        compare_runs(base / "out", dirs)


def test_compare_verifies_artifact_hash_and_byte_metadata():
    base = _tmp()
    dirs = _comparison_inputs(base)
    path = dirs[1] / "summary.json"
    raw = path.read_bytes()
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["artifacts"] = {
        "summary.json": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    }
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
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
    (dirs[1] / "manifest.json").write_text(json.dumps(second_manifest))
    with pytest.raises(ValueError, match="duplicate input artifact_id"):
        compare_runs(base / "out", dirs)


def test_compare_marks_different_exact_global_schedules_invalid():
    base = _tmp()
    dirs = _comparison_inputs(base)
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["schedule_identity"]["global_timestamps_sha256"] = "e" * 64
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
    out = compare_runs(base / "out", dirs)
    md = (out / "comparison.md").read_text()
    assert "INVALID COMPARISON" in md
    assert "different arrival schedule" in md


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

    concurrent_target = base / "concurrent"
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(
            lambda _i: compare_runs(concurrent_target, dirs), range(4)))
    assert len(set(outputs)) == 4
    assert all((out / "comparison.md").is_file() for out in outputs)
    assert all((out / ".traffic-replay-complete").is_file()
               for out in outputs)


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
