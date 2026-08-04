"""Request-parameter passthrough (extra_body) and reasoning-token reporting.

extra_body lets a user steer model behavior (top_p, stop, response_format,
and provider thinking control) without the harness losing control of the
keys it must own. Reasoning-token counts are read from usage the same way
cached tokens are, so thinking cost shows up in the report.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

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
    report = Path(out["out_dir"], "report.md").read_text()
    assert "reasoning tokens:" in report
    assert "reasoning_effort" in report  # provenance line echoes extra_body


def test_compare_table_has_reasoning_tokens_row():
    from traffic_replay.aggregate import compare_runs

    def run_dir(title, reasoning_total):
        d = Path(tempfile.mkdtemp())
        summ = {"run": {"title": title, "endpoint_path": "/p"},
                "reasoning_tokens_total": reasoning_total,
                "throughput": {"input_tokens_per_min": 100,
                               "output_tokens_per_min": 50}}
        (d / "summary.json").write_text(json.dumps(summ))
        return str(d)

    out = Path(tempfile.mkdtemp())
    compare_runs(str(out), [run_dir("thinking-on", 1200),
                            run_dir("thinking-off", 0)])
    md = (out / "comparison.md").read_text()
    assert "reasoning tokens (total)" in md
    assert "1,200" in md
