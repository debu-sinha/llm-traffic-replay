"""Request-parameter passthrough (extra_body) and reasoning-token reporting.

extra_body lets a user steer model behavior (top_p, stop, response_format,
and provider thinking control) without the harness losing control of the
keys it must own. Reasoning-token counts are read from usage the same way
cached tokens are, so thinking cost shows up in the report.
"""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from traffic_replay.client import EndpointClient, EndpointConfig
from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run
from traffic_replay.sse import extract_usage


def test_extra_body_merges_but_core_keys_win():
    cfg = EndpointConfig(
        base_url="http://x", path="/p",
        extra_body={"top_p": 0.9,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 999, "stream": False, "messages": ["nope"],
                    "model": "evil", "stream_options": {"include_usage": False},
                    "temperature": 5})
    client = EndpointClient(cfg, None)
    body = json.loads(client._body([{"role": "user", "content": "hi"}], 128,
                                   True))
    # passthrough survives
    assert body["top_p"] == 0.9
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # harness-owned keys always win over anything in extra_body
    assert body["max_tokens"] == 128
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream_options"] == {"include_usage": True}
    assert "model" not in body                       # no cfg.model, none injected
    # the include_usage=False fallback retry must not let a user's
    # stream_options resurrect and re-trigger the 400 loop
    retry = json.loads(client._body([{"role": "user", "content": "hi"}], 128,
                                    False))
    assert "stream_options" not in retry
    assert retry["top_p"] == 0.9


def test_no_extra_body_is_unchanged():
    body = json.loads(EndpointClient(
        EndpointConfig(base_url="http://x", path="/p"), None)._body(
        [{"role": "user", "content": "hi"}], 64, False))
    assert set(body) == {"messages", "max_tokens", "temperature", "stream"}


@pytest.mark.parametrize("extra", [
    {"api_key": "sensitive-value"},
    {"metadata": {"authorization": "sensitive-value"}},
    {"metadata": "Bearer sensitive-value"},
])
def test_extra_body_rejects_credentials_because_it_is_persisted(extra):
    with pytest.raises(ValueError, match="persisted as evidence"):
        EndpointConfig(base_url="http://x", path="/p", extra_body=extra)


def test_reasoning_tokens_extracted_from_usage():
    u = extract_usage({"prompt_tokens": 100, "completion_tokens": 80,
                       "completion_tokens_details": {"reasoning_tokens": 55}})
    assert u["reasoning_tokens"] == 55
    assert u["reasoning_tokens_source"] == \
        "completion_tokens_details.reasoning_tokens"
    assert extract_usage({"prompt_tokens": 5})["reasoning_tokens"] is None


def test_reasoning_tokens_reported_end_to_end():
    d = tempfile.mkdtemp()
    pf = os.path.join(d, "p.jsonl")
    open(pf, "w").write(json.dumps({"prompt": "think about this"}) + "\n")
    truth = Path(d) / "truth.jsonl"
    srv = serve(0, truth, reasoning_tokens=4)  # mock emits reasoning
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.3)
    try:
        rc = RunConfig(
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "TRAFFIC_REPLAY_NO_TOKEN",
                      "extra_body": {"reasoning_effort": "low"}},
            prompts_file=pf, duration_s=5, qps_base=2.0, qps_burst=3.0,
            qps_min=1.0, qps_max=4.0, max_concurrency=4, calibrate_n=1,
            out_dir=os.path.join(d, "results"),
            title="reasoning + extra_body e2e", max_output_tokens_cap=16)
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()

    s = out["summary"]
    assert s["reasoning_tokens_total"] > 0
    assert s["reasoning_tokens_source"] == \
        "completion_tokens_details.reasoning_tokens"
    assert s["run"]["request_params"]["extra_body"] == \
        {"reasoning_effort": "low"}
    rows = [json.loads(line) for line in
            Path(out["out_dir"], "requests.jsonl").read_text().splitlines()
            if line.strip()]
    replay = [row for row in rows if row.get("phase") == "replay"]
    assert replay
    assert all(row["completion_tokens"] <= 16 for row in replay)
    assert all(row["reasoning_tokens"] <= row["completion_tokens"]
               for row in replay)
    truth_rows = [json.loads(line) for line in truth.read_text().splitlines()
                  if line.strip()]
    assert truth_rows
    assert all(row["completion_tokens"] <= 16 for row in truth_rows)
    report = Path(out["out_dir"], "report.md").read_text()
    assert "reasoning tokens:" in report
    assert "reasoning_effort" in report  # provenance line echoes extra_body


def test_compare_table_has_reasoning_tokens_row():
    from traffic_replay.aggregate import compare_runs

    def run_dir(title, reasoning_total):
        d = Path(tempfile.mkdtemp())
        schedule = {"seconds": 1, "requests": 1, "rate_min": 1.0,
                    "rate_p50": 1.0, "rate_p95": 1.0, "rate_max": 1.0,
                    "source": "test"}
        summ = {"run": {"title": title, "endpoint_path": "/p",
                         "input_mode": "profile"},
                "reasoning_tokens_total": reasoning_total,
                "harness_version": "0.4.1",
                "latency_basis": "send-to-first-token; connection excluded",
                "schedule": schedule,
                "throughput": {"input_tokens_per_min": 100,
                               "output_tokens_per_min": 50}}
        raw = json.dumps(summ).encode()
        (d / "summary.json").write_bytes(raw)
        requests_raw = b""
        (d / "requests.jsonl").write_bytes(requests_raw)
        manifest = {
            "manifest_schema_version": 3,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "harness_version": "0.4.1",
            "latency_basis": summ["latency_basis"],
            "input_mode": "profile",
            "profile_sha256": "b" * 64,
            "seed": 7,
            "request_params": {"temperature": 0.0},
            "schedule": schedule,
            "shard": "1/1",
            "workload_id": "workload-test",
            "logical_run_id": "logical-test",
            "run_id": "logical-test",
            "execution_id": f"execution-{title}",
            "artifact_id": f"artifact-{title}",
            "schedule_identity": {
                "encoding": "float64-le-seconds-from-run-start",
                "global_timestamps_sha256": "c" * 64,
                "global_count": 1,
                "global_min_s": 0.0,
                "global_max_s": 0.0,
                "shard_timestamps_sha256": "c" * 64,
                "shard_count": 1,
                "shard_min_s": 0.0,
                "shard_max_s": 0.0,
            },
            "index_identity": {
                "encoding": "int64-le",
                "global_indices_sha256": "d" * 64,
                "count": 1,
                "min": 0,
                "max": 0,
                "global_count": 1,
                "shard_index": 0,
                "shard_total": 1,
                "partition": "unsharded",
            },
            "artifacts": {
                "summary.json": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                },
                "requests.jsonl": {
                    "sha256": hashlib.sha256(requests_raw).hexdigest(),
                    "bytes": len(requests_raw),
                    "row_count": 0,
                },
            },
        }
        (d / "manifest.json").write_text(json.dumps(manifest))
        manifest_raw = (d / "manifest.json").read_bytes()
        (d / ".traffic-replay-complete").write_text(json.dumps({
            "artifact_id": manifest["artifact_id"],
            "status": "complete",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_bytes": len(manifest_raw),
            "request_rows": 0,
        }) + "\n")
        return str(d)

    out = compare_runs(
        Path(tempfile.mkdtemp()) / "comparison",
        [run_dir("thinking-on", 1200), run_dir("thinking-off", 0)])
    md = (out / "comparison.md").read_text()
    assert "reasoning tokens (total)" in md
    assert "1,200" in md
