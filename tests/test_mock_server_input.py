"""The validation oracle must fail closed on ambiguous request JSON."""
from __future__ import annotations

import http.client
import threading
import time

import pytest

from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run


@pytest.fixture
def mock_endpoint(tmp_path):
    truth = tmp_path / "truth.jsonl"
    server = serve(0, truth, ttft_base_ms=0, ms_per_1k_uncached=0,
                   per_token_ms=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], truth
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(port: int, body: bytes) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(
            "POST", "/serving-endpoints/mock/invocations", body=body,
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("body", [
    b'{"messages":[],"messages":[]}',
    b'{"messages":[],"temperature":NaN}',
    b'[{"messages":[]}]',
    b'{"messages":[{"role":"user","content":7}]}',
    b'{"messages":[{"role":"user","content":"hello"}],'
    b'"max_tokens":true}',
    b'{"messages":[{"role":"user","content":"\xff"}]}',
])
def test_mock_rejects_ambiguous_or_wrong_typed_json(mock_endpoint, body):
    port, truth = mock_endpoint

    status, _response = _post(port, body)

    assert status == 400
    assert truth.read_text() == ""


def test_mock_remains_usable_after_bad_json(mock_endpoint):
    port, truth = mock_endpoint
    assert _post(port, b'{"messages":[],"messages":[]}')[0] == 400

    status, response = _post(
        port,
        b'{"messages":[{"role":"user","content":"hello"}],'
        b'"max_tokens":1,"stream":true}',
    )

    assert status == 200
    assert b"data: [DONE]" in response
    assert len(truth.read_text().splitlines()) == 1


def test_runner_rejects_legacy_unbound_prior_cli_traffic_before_network(
        tmp_path):
    port = 9
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"hello"}\n')
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n")
    stamp = time.time() - 0.1
    prior = {
        "phase": "preflight", "request_id": "preflight-one",
        "first_attempt_unix": stamp - 0.001,
        "first_send_unix": stamp, "t_send_unix": stamp,
        "finished_unix": stamp + 0.01,
        "status": 200, "ok": True, "prompt_tokens": 5,
        "completion_tokens": 1, "max_tokens_requested": 2,
        "request_attempts": 1, "connection_attempts": 1, "retries": 0,
        "retry_reasons": [], "stream_complete": True, "parse_errors": 0,
        "endpoint_adapter": "openai.chat_completions.sse/v1",
        "response_mode": "streaming",
    }
    config = RunConfig(
        prompts_file=str(prompts), timestamps_file=str(trace), duration_s=1,
        endpoint={
            "base_url": f"http://127.0.0.1:{port}",
            "path": "/serving-endpoints/mock/invocations",
            "auth_token_env": "TRAFFIC_REPLAY_NO_TOKEN",
        },
        qps_base=1, qps_burst=1, qps_min=1, qps_max=1,
        max_concurrency=2, max_pending_requests=2, calibrate_n=0,
        max_output_tokens_cap=1, capture_endpoint_metadata=False,
        measure_network_path=False, out_dir=str(tmp_path / "results"),
    )

    with pytest.raises(ValueError, match="legacy carried rows fail closed"):
        run(config, quiet=True, prior_request_rows=[prior])
    assert not (tmp_path / "results").exists()
