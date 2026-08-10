"""merge pools replay rows from several run dirs and re-summarizes the union,
and refuses to merge different endpoints without force."""
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import pytest

from traffic_replay.aggregate import (
    _require_run_dir,
    _verified_comparison_request_evidence,
    merge_runs,
)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="merge-"))


def _row(i, ttft, e2e):
    return {"request_id": f"r{i}", "global_index": i,
            "phase": "replay", "ok": True,
            "ttft_ms": ttft, "ttfb_ms": ttft - 3, "e2e_ms": e2e,
            "interchunk_max_ms": 4.0, "dispatch_lag_ms": 1.0,
            "scheduled_s": float(i),
            "t_send_unix": 1000.0 + i, "prompt_tokens": 1000,
            "completion_tokens": 50, "cached_tokens": None,
            "cached_tokens_source": None, "intended_input_tokens": 1000,
            "intended_output_tokens": 50, "intended_cache_fraction": 0.6,
            "content_chunks": 50, "finish_reason": "stop", "status": 200,
            "error": None, "doc_id": 1, "chars_sent": 4000, "retries": 0}


def _rate_limits(**overrides):
    limits = {
        "input_tokens_per_minute": 10_000,
        "output_tokens_per_minute": 2_000,
        "queries_per_hour": 7_200,
        "warning_utilization": 0.8,
        "source": ("https://docs.databricks.com/aws/en/machine-learning/"
                   "foundation-model-apis/limits"),
        "as_of": "2026-08-03",
        "scope": "Enterprise workspace pay-per-token traffic",
        "provider": "databricks",
        "deployment_mode": "pay_per_token",
        "workspace_tier": "Enterprise",
        "model": "model",
        "accounting_model": "databricks_fmapi_pay_per_token",
    }
    limits.update(overrides)
    return limits


def _endpoint_metadata(name="model"):
    return {
        "name": name,
        "task": "llm/v1/chat",
        "route_optimized": False,
        "ready": "READY",
        "served_entities": [{
            "name": name,
            "foundation_model": {"name": f"system.ai.{name}"},
        }],
        "note": "captured test metadata",
    }


def _source_manifest(ep: str, *, input_mode="profile", profile_sha="b" * 64,
                     shard_index=0, shard_total=2, local_requests=5,
                     global_requests=None):
    global_requests = (local_requests * shard_total
                       if global_requests is None else global_requests)
    shard = f"{shard_index + 1}/{shard_total}"
    return {
        "manifest_schema_version": 3,
        "git_commit": "a" * 40, "git_dirty": False,
        "harness_version": "0.4.1",
        "latency_basis": "send-to-first-token; connection excluded",
        "input_mode": input_mode, "profile_sha256": profile_sha,
        "seed": 7,
        "request_params": {"temperature": 0.0,
                           "max_output_tokens_cap": 512},
        "config_identity": {
            "sla_definition": {"ttft_definition": "first_content"},
        },
        "schedule": {"seconds": 120, "requests": local_requests,
                     "total_requests": global_requests, "shard": shard,
                     "rate_min": 5.0, "rate_p50": 5.0,
                     "rate_p95": 5.0, "rate_max": 5.0,
                     "source": "synthetic"},
        "endpoint_base_url": "https://example.test",
        "endpoint_model": "model", "endpoint_path": ep,
        "workload_id": "workload-test",
        "logical_run_id": "logical-test-run", "run_id": "logical-test-run",
        "execution_id": f"execution-{shard_index}",
        "artifact_id": f"artifact-{shard_index}",
        "start_at_unix": 1_800_000_000.0, "shard": shard,
    }


def _seal_completion(d: Path) -> None:
    manifest = json.loads((d / "manifest.json").read_text())
    manifest_raw = (d / "manifest.json").read_bytes()
    (d / ".traffic-replay-complete").write_text(json.dumps({
        "artifact_id": manifest["artifact_id"],
        "status": "complete",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_bytes": len(manifest_raw),
        "request_rows": manifest["artifacts"]["requests.jsonl"]["row_count"],
    }) + "\n")


def _write_manifest(d: Path, manifest: dict) -> None:
    (d / "manifest.json").write_text(json.dumps(manifest))
    _seal_completion(d)


def _refresh_artifacts(d: Path) -> None:
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for name in ("summary.json", "requests.jsonl"):
        raw = (d / name).read_bytes()
        metadata = {"sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw)}
        if name == "requests.jsonl":
            metadata["row_count"] = len(raw.splitlines())
        artifacts[name] = metadata
    manifest["artifacts"] = artifacts
    rows = [json.loads(line)
            for line in (d / "requests.jsonl").read_text().splitlines()]
    replay = [row for row in rows if row.get("phase") == "replay"]
    replay.sort(key=lambda row: row["global_index"])
    indices = [row["global_index"] for row in replay]
    timestamps = [float(row["scheduled_s"]) for row in replay]
    shown, shard_total = (int(value) for value in manifest["shard"].split("/"))
    shard_index = shown - 1
    global_count = manifest["schedule"]["total_requests"]

    def packed_hash(values, fmt):
        digest = hashlib.sha256()
        for value in values:
            digest.update(struct.pack(fmt, value))
        return digest.hexdigest()

    global_timestamps = [float(index) for index in range(global_count)]
    manifest["schedule_identity"] = {
        "encoding": "float64-le-seconds-from-run-start",
        "global_timestamps_sha256": packed_hash(global_timestamps, "<d"),
        "global_count": global_count,
        "global_min_s": min(global_timestamps) if global_timestamps else None,
        "global_max_s": max(global_timestamps) if global_timestamps else None,
        "shard_timestamps_sha256": packed_hash(timestamps, "<d"),
        "shard_count": len(timestamps),
        "shard_min_s": min(timestamps) if timestamps else None,
        "shard_max_s": max(timestamps) if timestamps else None,
    }
    manifest["index_identity"] = {
        "encoding": "int64-le",
        "global_indices_sha256": packed_hash(indices, "<q"),
        "count": len(indices),
        "min": min(indices) if indices else None,
        "max": max(indices) if indices else None,
        "global_count": global_count,
        "shard_index": shard_index,
        "shard_total": shard_total,
        "partition": ("unsharded" if shard_total == 1
                      else "round_robin_modulo"),
    }
    manifest_path.write_text(json.dumps(manifest))
    if (d / ".traffic-replay-complete").exists():
        _seal_completion(d)


def _mkrun(d: Path, ep: str, ttfts, title="run", profile_sha="b" * 64,
           shard_index=None, shard_total=2, global_requests=None):
    if shard_index is None:
        shard_index = 0 if d.name == "a" else 1
    manifest = _source_manifest(
        ep, profile_sha=profile_sha, shard_index=shard_index,
        shard_total=shard_total, local_requests=len(ttfts),
        global_requests=global_requests)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "run": {"endpoint_path": ep, "title": title,
                "input_mode": "profile",
                "ttft_definition": "first_content"},
        "harness_version": "0.4.1",
        "latency_basis": "send-to-first-token; connection excluded",
        "schedule": manifest["schedule"],
    }))
    (d / "manifest.json").write_text(json.dumps(manifest))
    with (d / "requests.jsonl").open("w") as f:
        cal = dict(_row(0, 999.0, 999.0))
        cal["phase"] = "calibration"
        f.write(json.dumps(cal) + "\n")   # proves merge keeps only replay rows
        for local_index, t in enumerate(ttfts):
            global_index = shard_index + local_index * shard_total
            f.write(json.dumps(_row(global_index, float(t),
                                    float(t) + 200)) + "\n")
    _refresh_artifacts(d)
    _seal_completion(d)


def _set_quota_evidence(d: Path, *, limits=None, endpoint_metadata=None,
                        binding_complete=True):
    limits = _rate_limits() if limits is None else limits
    endpoint_metadata = (_endpoint_metadata() if endpoint_metadata is None
                         else endpoint_metadata)
    summary = json.loads((d / "summary.json").read_text())
    summary["run"]["endpoint_metadata"] = endpoint_metadata
    summary["rate_limits"] = {
        "configured": limits,
        "binding": {"binding_complete": binding_complete},
        "comparisons": {},
        "warning": None,
    }
    (d / "summary.json").write_text(json.dumps(summary))
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["endpoint_metadata"] = endpoint_metadata
    manifest["effective_config"] = {"rate_limits": limits}
    (d / "manifest.json").write_text(json.dumps(manifest))
    _refresh_artifacts(d)


def _set_quota_row_evidence(d: Path, prompt_tokens):
    path = d / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(prompt_tokens)
    for offset, (row, tokens) in enumerate(zip(rows, prompt_tokens)):
        stamp = 2_000.0 + offset / 10.0
        row.update({
            "first_send_unix": stamp,
            "finished_unix": stamp + 0.05,
            "prompt_tokens": tokens,
            "completion_tokens": 10,
            "max_tokens_requested": 20,
            "request_attempts": 1,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifacts(d)


def _set_first_visible_evidence(d: Path, visible_ms: float) -> None:
    summary_path = d / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run"]["ttft_definition"] = "first_visible"
    summary["sla"] = {"ttft_definition": "first_visible"}
    summary_path.write_text(json.dumps(summary))

    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_identity"] = {
        "sla_definition": {"ttft_definition": "first_visible"},
    }
    manifest_path.write_text(json.dumps(manifest))

    requests_path = d / "requests.jsonl"
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    for row in rows:
        if row.get("phase") != "replay":
            continue
        row.update({
            "ttfv_ms": visible_ms,
            "caller_ttfv_ms": visible_ms,
            "visible_content_seen": True,
            "stream_complete": True,
            "parse_errors": 0,
        })
    requests_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifacts(d)


def _set_acceptance_policy(d: Path, policy: dict) -> None:
    summary_path = d / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sla"] = {
        "ttft_definition": "first_content",
        "acceptance_config": {**policy, "targets_are": "source fixture"},
    }
    summary_path.write_text(json.dumps(summary))

    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_identity"]["sla_definition"][
        "acceptance_config"] = policy
    manifest_path.write_text(json.dumps(manifest))
    _refresh_artifacts(d)


def test_merge_pools_and_percentiles_from_union():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/pt/invocations", [100] * 5)
    _mkrun(base / "b", "/serving-endpoints/pt/invocations", [300] * 5)
    out = merge_runs(base / "out", [base / "a", base / "b"])
    summ = json.loads((out / "summary.json").read_text())
    assert summ["requests_total"] == 10           # calibration rows excluded
    assert summ["ttft_ms"]["n"] == 10
    assert 100 <= summ["ttft_ms"]["p50"] <= 300    # from the union
    sealed_rows = [json.loads(line) for line in
                   (out / "requests.jsonl").read_text().splitlines()]
    assert len(sealed_rows) == 12
    assert sum(row["phase"] == "replay" for row in sealed_rows) == 10
    assert sum(row["phase"] == "calibration" for row in sealed_rows) == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 3
    assert all(manifest[field] for field in (
        "workload_id", "logical_run_id", "execution_id", "artifact_id"))
    assert manifest["profile_sha256"] == "b" * 64
    assert manifest["execution_id"].startswith("execution-")
    assert manifest["execution_id"] != manifest["artifact_id"]
    assert manifest["index_identity"]["count"] == 10
    assert manifest["index_identity"]["global_count"] == 10
    assert manifest["schedule_identity"]["shard_count"] == 10
    assert set(("requests.jsonl", "summary.json")) <= set(
        manifest["artifacts"])
    assert (out / ".traffic-replay-complete").is_file()
    assert not (out / ".traffic-replay-writing").exists()


def test_merge_preserves_first_visible_scoring_and_global_schedule():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [10] * 5)
    _mkrun(base / "b", endpoint, [10] * 5)
    _set_first_visible_evidence(base / "a", 500.0)
    _set_first_visible_evidence(base / "b", 500.0)

    out = merge_runs(
        base / "out", [base / "a", base / "b"],
        acceptance={"ttft_ms": {"p95": 120.0}},
    )
    summary = json.loads((out / "summary.json").read_text())

    assert summary["run"]["ttft_definition"] == "first_visible"
    assert summary["sla"]["ttft_definition"] == "first_visible"
    assert summary["sla"]["ttft_vs_target"][0]["actual_ms"] == 500.0
    assert summary["sla"]["ttft_vs_target"][0]["met"] is False
    assert summary["schedule"]["requests"] == 10
    assert summary["schedule"]["total_requests"] == 10
    assert summary["schedule"]["seconds"] == 120
    assert summary["schedule"]["source"] == "synthetic"


def test_merge_rejects_conflicting_first_event_declarations():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    summary_path = base / "b" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["run"]["ttft_definition"] = "first_visible"
    summary_path.write_text(json.dumps(summary))
    _refresh_artifacts(base / "b")

    with pytest.raises(ValueError, match="conflicting TTFT definition"):
        merge_runs(base / "out", [base / "a", base / "b"])


def test_merge_propagates_source_acceptance_and_labels_post_hoc_override():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    source_policy = {"ttft_ms": {"p95": 200.0}, "success_rate": 0.95}
    for name in ("a", "b"):
        _mkrun(base / name, endpoint, [100] * 2)
        _set_acceptance_policy(base / name, source_policy)

    propagated = merge_runs(
        base / "propagated", [base / "a", base / "b"])
    propagated_summary = json.loads(
        (propagated / "summary.json").read_text())
    provenance = propagated_summary["run"]["aggregation"][
        "acceptance_policy_provenance"]
    assert propagated_summary["sla"]["acceptance_config"]["ttft_ms"] == {
        "p95": 200.0}
    assert provenance["mode"] == "source_policy_propagated"
    assert provenance["post_hoc"] is False

    override = {"ttft_ms": {"p95": 150.0}, "success_rate": 0.90}
    rescored = merge_runs(
        base / "rescored", [base / "a", base / "b"],
        acceptance=override)
    rescored_summary = json.loads((rescored / "summary.json").read_text())
    provenance = rescored_summary["run"]["aggregation"][
        "acceptance_policy_provenance"]
    assert rescored_summary["sla"]["acceptance_config"] == override
    assert provenance["mode"] == "post_hoc_override"
    assert provenance["post_hoc"] is True
    manifest = json.loads((rescored / "manifest.json").read_text())
    assert manifest["effective_config"]["acceptance_policy_provenance"][
        "post_hoc"] is True


def test_merge_rejects_different_source_acceptance_policies():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_acceptance_policy(base / "a", {"ttft_ms": {"p95": 200.0}})
    _set_acceptance_policy(base / "b", {"ttft_ms": {"p95": 300.0}})

    with pytest.raises(ValueError, match="different acceptance policies"):
        merge_runs(base / "out", [base / "a", base / "b"])


def test_merge_pools_every_sealed_traffic_phase_for_quota_only():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [300] * 2)
    # Each source contains calibration + two replay rows. All six sends overlap
    # in one rolling minute, so the union must be 700+100+100+500+100+100.
    _set_quota_row_evidence(base / "a", [700, 100, 100])
    _set_quota_row_evidence(base / "b", [500, 100, 100])
    _set_quota_evidence(base / "a")
    _set_quota_evidence(base / "b")

    out = merge_runs(base / "out", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())

    assert summary["requests_total"] == 4
    assert summary["ttft_ms"]["n"] == 4
    windows = summary["observed_rate_windows"]
    assert windows["input_tokens_by_first_send"]["max"] == 1_600
    assert windows["traffic_scope"]["rows"] == 6
    assert windows["traffic_scope"]["phases"]["calibration"]["rows"] == 2
    assert windows["traffic_scope"]["phases"]["replay"]["rows"] == 4
    assert summary["rate_limits"]["configured"] == _rate_limits()
    assert summary["rate_limits"]["binding"]["binding_complete"] is True
    assert summary["run"]["quota_merge"]["sla_population"] == "replay_only"
    assert summary["run"]["quota_merge"]["sealed_rows"] == 6
    assert summary["run"]["quota_merge"]["observed_phase_rows"] == {
        "calibration": 2, "replay": 4}
    sealed = [json.loads(line) for line in
              (out / "requests.jsonl").read_text().splitlines()]
    assert len(sealed) == 6
    assert {row["phase"] for row in sealed} == {"calibration", "replay"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["effective_config"]["rate_limits"] == _rate_limits()
    assert manifest["endpoint_metadata"] == _endpoint_metadata()


def test_merge_preserves_unknown_setup_outcome_as_incomplete_quota_evidence():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    for directory in (base / "a", base / "b"):
        _set_quota_row_evidence(directory, [100, 100, 100])
        _set_quota_evidence(directory)
    path = base / "b" / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["first_send_unix"] = None
    rows[0]["request_attempts"] = None
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifacts(base / "b")

    out = merge_runs(base / "out", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())

    scope = summary["observed_rate_windows"]["traffic_scope"]
    assert scope["unknown_outcome_rows"] == 1
    assert scope["phases"]["calibration"]["unknown_outcome_rows"] == 1
    assert all(comparison["status"] == "incomplete_run_evidence"
               for comparison in
               summary["rate_limits"]["comparisons"].values())
    assert "cannot establish headroom" in summary["rate_limits"]["warning"]


def test_merge_refuses_different_quota_snapshots_and_force_withholds_claim():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_quota_evidence(base / "a")
    _set_quota_evidence(
        base / "b", limits=_rate_limits(input_tokens_per_minute=9_000))

    with pytest.raises(ValueError, match="different rate-limit snapshots"):
        merge_runs(base / "refused", [base / "a", base / "b"])
    out = merge_runs(
        base / "diagnostic", [base / "a", base / "b"], force=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["aggregation_valid"] is False
    assert "rate_limits" not in summary
    assert summary["observed_rate_windows"]["withheld"] is True
    assert summary["run"]["quota_merge"][
        "configured_snapshot_status"] == "withheld_invalid_inputs"


def test_merge_refuses_partial_quota_snapshot_coverage():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_quota_evidence(base / "a")

    with pytest.raises(
            ValueError, match="not complete for every merge source"):
        merge_runs(base / "refused", [base / "a", base / "b"])


def test_merge_refuses_snapshot_disagreement_inside_one_source():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_quota_evidence(base / "a")
    _set_quota_evidence(base / "b")
    summary = json.loads((base / "b" / "summary.json").read_text())
    summary["rate_limits"]["configured"]["queries_per_hour"] = 7_199
    (base / "b" / "summary.json").write_text(json.dumps(summary))
    _refresh_artifacts(base / "b")

    with pytest.raises(ValueError, match="manifest and summary"):
        merge_runs(base / "refused", [base / "a", base / "b"])


def test_merge_refuses_unknown_phase_in_configured_quota_evidence():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_quota_evidence(base / "a")
    _set_quota_evidence(base / "b")
    path = base / "b" / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["phase"] = "warmup"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifacts(base / "b")

    with pytest.raises(ValueError, match="unsupported request phases: warmup"):
        merge_runs(base / "refused", [base / "a", base / "b"])


def test_merge_refuses_incomplete_or_different_quota_endpoint_binding():
    base = _tmp()
    endpoint = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", endpoint, [100] * 2)
    _mkrun(base / "b", endpoint, [100] * 2)
    _set_quota_evidence(base / "a")
    _set_quota_evidence(base / "b", binding_complete=False)

    with pytest.raises(ValueError, match="incomplete rate-limit endpoint binding"):
        merge_runs(base / "binding", [base / "a", base / "b"])

    changed_metadata = _endpoint_metadata()
    changed_metadata["ready"] = "NOT_READY"
    _set_quota_evidence(base / "b", endpoint_metadata=changed_metadata)
    with pytest.raises(ValueError, match="different rate-limit endpoint metadata"):
        merge_runs(base / "metadata", [base / "a", base / "b"])


def test_merge_refuses_mismatched_endpoints_without_force():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/AAA/invocations", [100] * 3)
    _mkrun(base / "b", "/serving-endpoints/BBB/invocations", [200] * 3)
    with pytest.raises(ValueError):
        merge_runs(base / "o1", [base / "a", base / "b"])
    out = merge_runs(base / "o2", [base / "a", base / "b"], force=True)
    assert json.loads((out / "summary.json").read_text())["requests_total"] == 6


def test_merge_missing_input_dir_gives_clean_error():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/pt/invocations", [100] * 3)
    with pytest.raises(ValueError):
        merge_runs(base / "out", [base / "a", base / "does_not_exist"])


def test_merge_refuses_a_source_without_a_manifest():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    (base / "b" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing manifest.json"):
        merge_runs(base / "out", [base / "a", base / "b"])


def test_merged_report_carries_concurrency_note():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/pt/invocations", [100] * 4)
    _mkrun(base / "b", "/serving-endpoints/pt/invocations", [200] * 4)
    out = merge_runs(base / "out", [base / "a", base / "b"])
    assert "union wall-clock window" in (out / "report.md").read_text()


def _mkprompts_run(d: Path, ep: str, n_rows: int, prompts_count: int):
    """A shard from prompts mode, carrying the fields summarize() needs to
    know the prompts were cycled."""
    shard_index = 0 if d.name == "a" else 1
    manifest = _source_manifest(
        ep, input_mode="prompts", shard_index=shard_index,
        shard_total=2, local_requests=n_rows, global_requests=n_rows * 2)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "run": {"endpoint_path": ep, "title": "shard",
                "input_mode": "prompts", "prompts_file": "p.jsonl",
                "prompts_count": prompts_count},
        "harness_version": "0.4.1",
        "latency_basis": "send-to-first-token; connection excluded",
        "schedule": manifest["schedule"],
    }))
    (d / "manifest.json").write_text(json.dumps(manifest))
    with (d / "requests.jsonl").open("w") as f:
        for local_index in range(n_rows):
            global_index = shard_index + local_index * 2
            f.write(json.dumps(_row(global_index, 100.0, 300.0)) + "\n")
    _refresh_artifacts(d)
    _seal_completion(d)


def test_merged_prompts_run_keeps_the_replay_caution():
    """Each shard cycled the same small prompt file, so the pooled cache
    fraction is still replay behavior. Losing the caution on merge would put
    the flattering number in the pooled report with nothing next to it."""
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkprompts_run(base / "a", ep, 60, 10)
    _mkprompts_run(base / "b", ep, 60, 10)
    out = merge_runs(base / "pooled", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["input_mode"] == "prompts"
    assert summary["replay"]["distinct_prompts"] == 10
    assert summary["replay"]["warning"] is not None
    assert "CAUTION (prompt replay)" in (out / "report.md").read_text()


def test_merged_run_reports_no_stability_verdict():
    """Pooled shards ran at different times, so a trend across them would
    describe the schedule rather than the endpoint."""
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5)
    _mkrun(base / "b", ep, [300] * 5)
    out = merge_runs(base / "pooled", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())
    assert "drift_kind" not in summary["drift"]
    assert "not computed for a merged run" in summary["drift"]["note"]


def test_profile_mode_merge_has_no_replay_block():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5)
    _mkrun(base / "b", ep, [120] * 5)
    out = merge_runs(base / "pooled", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())
    assert "replay" not in summary


def test_shards_disagreeing_on_prompt_count_do_not_claim_one():
    """Different prompts_count across shards means the pooled repeat factor is
    not well defined, so the carry-through must not invent one."""
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkprompts_run(base / "a", ep, 60, 10)
    _mkprompts_run(base / "b", ep, 60, 25)
    out = merge_runs(base / "pooled", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())
    assert "replay" not in summary


def test_merged_run_does_not_report_wire_lateness():
    """Shards start at different wall-clock times, so one schedule-vs-send
    offset across pooled rows reads the gap between shards as lateness. The
    real pooled artifact shows 3.3 s of exactly that."""
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5)
    _mkrun(base / "b", ep, [300] * 5)
    out = merge_runs(base / "pooled", [base / "a", base / "b"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arrivals"]["wire_lateness_ms"]["n"] == 0
    assert "client" not in summary
    note = summary["arrivals"]["wire_lateness_note"]
    assert "not computed for a merged run" in note
    assert note in (out / "report.md").read_text()


def test_merge_does_not_reconstruct_legacy_caller_latency():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5)
    _mkrun(base / "b", ep, [300] * 5)
    out = merge_runs(
        base / "pooled", [base / "a", base / "b"],
        acceptance={"targets_are": "test", "ttft_ms": {"p50": 250}})
    summary = json.loads((out / "summary.json").read_text())
    for key in ("ttft_corrected_ms", "ttfv_corrected_ms",
                "ttf_tool_call_corrected_ms", "e2e_corrected_ms"):
        assert key not in summary
    assert summary["sla"]["latency_basis"] == \
        "service_time_no_schedule_wait_available"
    assert "legacy schedule/send timestamps cannot be reconstructed" in \
        summary["latency_correction_note"]


def test_merge_pools_exact_caller_clocks_and_scores_them():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5)
    _mkrun(base / "b", ep, [300] * 5)
    for directory, caller_ttft, caller_e2e in (
            (base / "a", 150.0, 400.0),
            (base / "b", 350.0, 600.0)):
        for index in range(5):
            _edit_replay_row(
                directory, index, caller_ttft_ms=caller_ttft,
                caller_e2e_ms=caller_e2e)
    out = merge_runs(
        base / "pooled", [base / "a", base / "b"],
        acceptance={"targets_are": "test", "ttft_ms": {"p50": 300}})
    summary = json.loads((out / "summary.json").read_text())
    assert summary["ttft_corrected_ms"]["p50"] == 250.0
    assert summary["e2e_corrected_ms"]["p50"] == 500.0
    assert summary["sla"]["ttft_metric"] == "ttft_corrected_ms"
    assert summary["sla"]["latency_basis"] == "caller_experienced"
    assert summary["latency_correction_provenance"] == {
        "exact_values": 20, "legacy_reconstructed_values": 0}
    assert "pools only exact monotonic durations" in \
        summary["latency_correction_note"]


def test_merge_rejects_different_workload_hashes_and_force_marks_invalid():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 5, profile_sha="b" * 64)
    _mkrun(base / "b", ep, [100] * 5, profile_sha="c" * 64)
    with pytest.raises(ValueError, match="profile or prompts SHA-256"):
        merge_runs(base / "refused", [base / "a", base / "b"])
    out = merge_runs(base / "forced", [base / "a", base / "b"], force=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["aggregation_valid"] is False
    assert "different profile or prompts SHA-256" in \
        " ".join(summary["run"]["compatibility_issues"])
    assert "verdict: INVALID" in (out / "report.md").read_text()


def _edit_manifest(d: Path, **changes):
    manifest = json.loads((d / "manifest.json").read_text())
    manifest.update(changes)
    _write_manifest(d, manifest)


def _edit_replay_row(d: Path, replay_index: int, **changes):
    path = d / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    replay = [index for index, row in enumerate(rows)
              if row.get("phase") == "replay"]
    rows[replay[replay_index]].update(changes)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifacts(d)


def test_merge_rejects_duplicate_input_directory_and_alias():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    with pytest.raises(ValueError, match="duplicate input run dir"):
        merge_runs(base / "out", [base / "a", base / "a"])
    alias = base / "alias"
    alias.symlink_to(base / "a", target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate input run dir"):
        merge_runs(base / "alias-out", [base / "a", alias])


def test_merge_rejects_incomplete_writing_and_unsupported_inputs():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    (base / "b" / ".traffic-replay-writing").touch()
    with pytest.raises(ValueError, match="still being written"):
        merge_runs(base / "writing", [base / "a", base / "b"])
    (base / "b" / ".traffic-replay-writing").unlink()
    (base / "b" / ".traffic-replay-complete").unlink()
    with pytest.raises(ValueError, match="completion marker"):
        merge_runs(base / "incomplete", [base / "a", base / "b"])
    (base / "b" / ".traffic-replay-complete").touch()
    _edit_manifest(base / "b", manifest_schema_version=999)
    with pytest.raises(ValueError, match="unsupported manifest schema"):
        merge_runs(base / "schema", [base / "a", base / "b"])


def test_merge_rejects_tampered_hashed_artifact():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    requests = base / "b" / "requests.jsonl"
    raw = requests.read_bytes()
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["artifacts"]["requests.jsonl"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "row_count": len(raw.splitlines()),
    }
    _write_manifest(base / "b", manifest)
    requests.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        merge_runs(base / "out", [base / "a", base / "b"])
    changed = requests.read_bytes()
    manifest["artifacts"]["requests.jsonl"].update({
        "sha256": hashlib.sha256(changed).hexdigest(),
        "bytes": len(changed),
        "row_count": changed.count(b"\n"),
    })
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="blank JSONL record"):
        merge_runs(base / "blank-row", [base / "a", base / "b"])


def test_merge_rejects_duplicate_keys_in_authenticated_request_jsonl():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    path = base / "b" / "requests.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = lines[0][:-1] + ',"ok":false}'
    path.write_text("\n".join(lines) + "\n")
    changed = path.read_bytes()
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["artifacts"]["requests.jsonl"] = {
        "sha256": hashlib.sha256(changed).hexdigest(),
        "bytes": len(changed),
        "row_count": len(lines),
    }
    _write_manifest(base / "b", manifest)

    with pytest.raises(
            ValueError,
            match=r"invalid JSON .*requests\.jsonl line 1: .*duplicate key 'ok'"):
        merge_runs(base / "out", [base / "a", base / "b"])


def test_merge_rejects_nonfinite_authenticated_request_jsonl():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    path = base / "b" / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["ttft_ms"] = float("inf")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    changed = path.read_bytes()
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["artifacts"]["requests.jsonl"] = {
        "sha256": hashlib.sha256(changed).hexdigest(),
        "bytes": len(changed),
        "row_count": len(rows),
    }
    _write_manifest(base / "b", manifest)

    with pytest.raises(
            ValueError,
            match=r"invalid JSON .*requests\.jsonl line 1: .*non-finite"):
        merge_runs(base / "out", [base / "a", base / "b"])


def test_merge_requires_requests_hash_and_row_count_metadata():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    del manifest["artifacts"]["requests.jsonl"]["row_count"]
    (base / "b" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requests.jsonl row_count"):
        merge_runs(base / "row-count", [base / "a", base / "b"])
    manifest["artifacts"].pop("requests.jsonl")
    (base / "b" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requests.jsonl"):
        merge_runs(base / "hash", [base / "a", base / "b"])

    _refresh_artifacts(base / "b")
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    del manifest["artifacts"]["summary.json"]["bytes"]
    (base / "b" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact byte counts"):
        merge_runs(base / "bytes", [base / "a", base / "b"])


def test_merge_rejects_exact_index_and_schedule_identity_tampering():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["index_identity"]["global_indices_sha256"] = "e" * 64
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="index_identity SHA-256"):
        merge_runs(base / "index", [base / "a", base / "b"])

    _refresh_artifacts(base / "b")
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["schedule_identity"]["shard_timestamps_sha256"] = "e" * 64
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="schedule_identity shard SHA-256"):
        merge_runs(base / "shard-schedule", [base / "a", base / "b"])

    _refresh_artifacts(base / "b")
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["schedule_identity"]["global_timestamps_sha256"] = "e" * 64
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="global schedule disagrees"):
        merge_runs(base / "global-schedule", [base / "a", base / "b"])


@pytest.mark.parametrize("field", ["logical_run_id", "start_at_unix"])
def test_merge_rejects_null_or_inconsistent_shared_identity(field):
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    if field == "logical_run_id":
        _edit_manifest(base / "a", logical_run_id=None, run_id=None)
        with pytest.raises(ValueError, match="logical_run_id"):
            merge_runs(base / "null", [base / "a", base / "b"])
        _edit_manifest(base / "a", logical_run_id="one", run_id="one")
        _edit_manifest(base / "b", logical_run_id="two", run_id="two")
        match = "inconsistent logical_run_id"
    else:
        _edit_manifest(base / "a", start_at_unix=None)
        with pytest.raises(ValueError, match="start_at_unix"):
            merge_runs(base / "null", [base / "a", base / "b"])
        _edit_manifest(base / "a", start_at_unix=1_800_000_000.0)
        _edit_manifest(base / "b", start_at_unix=1_800_000_001.0)
        match = "inconsistent shared start_at_unix"
    with pytest.raises(ValueError, match=match):
        merge_runs(base / "different", [base / "a", base / "b"])


def test_merge_rejects_duplicate_or_inconsistent_shard_metadata():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    manifest = json.loads((base / "b" / "manifest.json").read_text())
    manifest["shard"] = "1/2"
    manifest["schedule"]["shard"] = "1/2"
    manifest["index_identity"]["shard_index"] = 0
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="duplicate shard indices"):
        merge_runs(base / "duplicate", [base / "a", base / "b"])

    manifest["shard"] = "2/3"
    manifest["schedule"]["shard"] = "2/3"
    manifest["index_identity"]["shard_index"] = 1
    manifest["index_identity"]["shard_total"] = 3
    _write_manifest(base / "b", manifest)
    with pytest.raises(ValueError, match="inconsistent shard totals"):
        merge_runs(base / "totals", [base / "a", base / "b"])


def test_merge_rejects_duplicate_request_ids_and_overlapping_indices():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    _edit_replay_row(base / "b", 0, request_id="r0")
    with pytest.raises(ValueError, match="duplicate replay request_id"):
        merge_runs(base / "requests", [base / "a", base / "b"])

    _edit_replay_row(base / "b", 0, request_id="unique", global_index=0)
    with pytest.raises(ValueError, match="overlapping replay global_index"):
        merge_runs(base / "indices", [base / "a", base / "b"])


def test_missing_index_coverage_is_never_marked_valid():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 3)
    _mkrun(base / "b", ep, [100] * 3)
    path = base / "b" / "requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    removed = False
    kept = []
    for row in rows:
        if row.get("phase") == "replay" and not removed:
            removed = True
            continue
        kept.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    _refresh_artifacts(base / "b")
    with pytest.raises(ValueError, match="not proven compatible"):
        merge_runs(base / "refused", [base / "a", base / "b"])
    out = merge_runs(base / "diagnostic", [base / "a", base / "b"],
                     force=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["aggregation_valid"] is False
    issues = " ".join(summary["run"]["compatibility_issues"])
    assert "global_index coverage" in issues
    manifest = _require_run_dir(out, "summary.json")
    assert manifest["schedule_identity"]["global_count"] == 5
    assert manifest["index_identity"]["global_count"] == 5
    assert manifest["index_identity"]["partition"] == \
        "diagnostic_observed_subset"
    assert manifest["schedule"]["source_expected_total_requests"] == 6
    evidence = _verified_comparison_request_evidence(out, manifest)
    assert evidence["phase_totals"]["replay"] == 5


def test_missing_expected_shard_is_never_marked_valid():
    base = _tmp()
    ep = "/serving-endpoints/pt/invocations"
    _mkrun(base / "a", ep, [100] * 2, shard_index=0, shard_total=3,
           global_requests=6)
    _mkrun(base / "b", ep, [100] * 2, shard_index=1, shard_total=3,
           global_requests=6)
    with pytest.raises(ValueError, match="missing expected shard indices"):
        merge_runs(base / "refused", [base / "a", base / "b"])
    out = merge_runs(base / "diagnostic", [base / "a", base / "b"],
                     force=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"]["aggregation_valid"] is False
    manifest = _require_run_dir(out, "summary.json")
    assert manifest["schedule_identity"]["global_count"] == 4
    assert manifest["schedule"]["source_expected_total_requests"] == 6
