"""merge pools replay rows from several run dirs and re-summarizes the union,
and refuses to merge different endpoints without force."""
import json
import tempfile
import pytest
from pathlib import Path
from traffic_replay.aggregate import merge_runs


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="merge-"))


def _row(i, ttft, e2e):
    return {"request_id": f"r{i}", "phase": "replay", "ok": True,
            "ttft_ms": ttft, "ttfb_ms": ttft - 3, "e2e_ms": e2e,
            "interchunk_max_ms": 4.0, "dispatch_lag_ms": 1.0,
            "t_send_unix": 1000.0 + i, "prompt_tokens": 1000,
            "completion_tokens": 50, "cached_tokens": None,
            "cached_tokens_source": None, "intended_input_tokens": 1000,
            "intended_output_tokens": 50, "intended_cache_fraction": 0.6,
            "content_chunks": 50, "finish_reason": "stop", "status": 200,
            "error": None, "doc_id": 1, "chars_sent": 4000, "retries": 0}


def _mkrun(d: Path, ep: str, ttfts, title="run"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {"run": {"endpoint_path": ep, "title": title}}))
    with (d / "requests.jsonl").open("w") as f:
        cal = dict(_row(0, 999.0, 999.0)); cal["phase"] = "calibration"
        f.write(json.dumps(cal) + "\n")   # proves merge keeps only replay rows
        for i, t in enumerate(ttfts):
            f.write(json.dumps(_row(i + 1, float(t), float(t) + 200)) + "\n")


def test_merge_pools_and_percentiles_from_union():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/pt/invocations", [100] * 5)
    _mkrun(base / "b", "/serving-endpoints/pt/invocations", [300] * 5)
    out = merge_runs(base / "out", [base / "a", base / "b"])
    summ = json.loads((out / "summary.json").read_text())
    assert summ["requests_total"] == 10           # calibration rows excluded
    assert summ["ttft_ms"]["n"] == 10
    assert 100 <= summ["ttft_ms"]["p50"] <= 300    # from the union
    assert len((out / "requests.jsonl").read_text().splitlines()) == 10


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


def test_merged_report_carries_concurrency_note():
    base = _tmp()
    _mkrun(base / "a", "/serving-endpoints/pt/invocations", [100] * 4)
    _mkrun(base / "b", "/serving-endpoints/pt/invocations", [200] * 4)
    out = merge_runs(base / "out", [base / "a", base / "b"])
    assert "union wall-clock window" in (out / "report.md").read_text()


def _mkprompts_run(d: Path, ep: str, n_rows: int, prompts_count: int):
    """A shard from prompts mode, carrying the fields summarize() needs to
    know the prompts were cycled."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {"run": {"endpoint_path": ep, "title": "shard",
                 "input_mode": "prompts", "prompts_file": "p.jsonl",
                 "prompts_count": prompts_count}}))
    with (d / "requests.jsonl").open("w") as f:
        for i in range(n_rows):
            f.write(json.dumps(_row(i + 1, 100.0, 300.0)) + "\n")


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
