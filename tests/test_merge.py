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
