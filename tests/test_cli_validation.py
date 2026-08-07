"""The instrument oracle and machine-readable CLI fail closed."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from traffic_replay.cli import (_answer_is_complete, cmd_validate,
                                _validation_error_stats,
                                _validation_passes)


def test_instrument_profile_is_a_packaged_resource_and_matches_example():
    packaged = files("traffic_replay").joinpath(
        "data/profile_validation_small.json").read_bytes()
    example = Path("configs/profile_validation_small.json").read_bytes()
    assert json.loads(packaged) == json.loads(example)


def test_large_negative_clock_error_cannot_pass_a_signed_percentile_check():
    report = {
        "ttft_error_ms": _validation_error_stats(np.full(100, -100.0)),
        "e2e_error_ms": _validation_error_stats(np.zeros(100)),
    }
    assert report["ttft_error_ms"]["p95"] == -100.0
    assert report["ttft_error_ms"]["absolute_p95"] == 100.0
    assert not _validation_passes(report, 60.0)


def test_tool_call_only_response_is_a_valid_completed_agent_answer():
    result = SimpleNamespace(
        stream_complete=True, parse_errors=0, visible_content_seen=False,
        valid_tool_calls=1)
    assert _answer_is_complete(result)
    result.valid_tool_calls = 0
    assert not _answer_is_complete(result)


def test_validate_port_zero_uses_assigned_port_and_emits_one_json_document(
        tmp_path, monkeypatch):
    assigned = 43127
    truth = tmp_path / "mock_truth.jsonl"
    requests = tmp_path / "run" / "requests.jsonl"
    requests.parent.mkdir()

    class FakeServer:
        server_address = ("127.0.0.1", assigned)

        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    def fake_serve(port, truth_path):
        assert port == 0
        assert Path(truth_path) == truth
        truth.write_text(json.dumps({
            "request_id": "r1", "ttft_true_ms": 10.0,
            "e2e_true_ms": 20.0}) + "\n")
        return FakeServer()

    def fake_run(rc, quiet=False):
        assert rc.endpoint["base_url"] == f"http://127.0.0.1:{assigned}"
        assert quiet is True
        requests.write_text(json.dumps({
            "phase": "replay", "ok": True, "request_id": "r1",
            "ttft_ms": 11.0, "e2e_ms": 22.0}) + "\n")
        return {"out_dir": str(requests.parent), "summary": {}}

    monkeypatch.setattr("traffic_replay.mock_server.serve", fake_serve)
    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr("traffic_replay.cli.time.sleep", lambda _: None)
    args = argparse.Namespace(
        port=0, workdir=str(tmp_path), duration=1, quiet=False,
        format="json", tolerance_ms=60.0)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cmd_validate(args) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["passed"] is True
    assert payload["ttft_error_ms"]["absolute_p95"] == 1.0
