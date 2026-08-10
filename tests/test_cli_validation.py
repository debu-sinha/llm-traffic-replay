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
import pytest

from traffic_replay.cli import (_answer_is_complete, cmd_validate, main,
                                _validation_error_stats,
                                _validation_passes)


def test_cli_reports_the_installed_package_version(capsys):
    from traffic_replay import __version__

    with pytest.raises(SystemExit) as stopped:
        main(["--version"])

    assert stopped.value.code == 0
    assert capsys.readouterr().out == f"traffic_replay {__version__}\n"


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
        valid_tool_calls=1, refusal_seen=False)
    assert _answer_is_complete(result)
    result.valid_tool_calls = 0
    assert not _answer_is_complete(result)


@pytest.mark.parametrize("visible,valid_tool_calls", [(True, 0), (False, 1)])
def test_refusal_is_never_a_completed_answer_even_with_usable_output(
        visible, valid_tool_calls):
    result = SimpleNamespace(
        stream_complete=True, parse_errors=0,
        visible_content_seen=visible,
        valid_tool_calls=valid_tool_calls,
        refusal_seen=True,
    )

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


def test_validate_defaults_to_a_collision_free_ephemeral_port(monkeypatch):
    seen = {}

    def fake_validate(args):
        seen["port"] = args.port
        return 0

    monkeypatch.setattr("traffic_replay.cli.cmd_validate", fake_validate)
    assert main(["validate"]) == 0
    assert seen["port"] == 0


@pytest.mark.parametrize("corrupt_file,corrupt_bytes,match", [
    (
        "truth",
        b'{"request_id":"r1","request_id":"r2",'
        b'"ttft_true_ms":10,"e2e_true_ms":20}\n',
        r"mock_truth\.jsonl:1.*duplicate key 'request_id'",
    ),
    (
        "requests",
        b'{"phase":"replay","ok":true,"request_id":"r1",'
        b'"ttft_ms":NaN,"e2e_ms":20}\n',
        r"requests\.jsonl:1.*non-finite",
    ),
    (
        "requests",
        b'{"phase":"replay","ok":true,"request_id":"r1",'
        b'"ttft_ms":"\xff","e2e_ms":20}\n',
        r"requests\.jsonl:1.*not UTF-8",
    ),
])
def test_validate_strictly_parses_both_oracle_and_result_jsonl(
        tmp_path, monkeypatch, corrupt_file, corrupt_bytes, match):
    """Even the local oracle is evidence input once it crosses a file edge."""
    requests = tmp_path / "run" / "requests.jsonl"
    requests.parent.mkdir()
    valid_truth = (
        b'{"request_id":"r1","ttft_true_ms":10,"e2e_true_ms":20}\n')
    valid_requests = (
        b'{"phase":"replay","ok":true,"request_id":"r1",'
        b'"ttft_ms":11,"e2e_ms":20}\n')

    class FakeServer:
        server_address = ("127.0.0.1", 43127)

        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    def fake_serve(_port, truth_path):
        Path(truth_path).write_bytes(
            corrupt_bytes if corrupt_file == "truth" else valid_truth)
        return FakeServer()

    def fake_run(_rc, quiet=False):
        assert quiet is True
        requests.write_bytes(
            corrupt_bytes if corrupt_file == "requests" else valid_requests)
        return {"out_dir": str(requests.parent), "summary": {}}

    monkeypatch.setattr("traffic_replay.mock_server.serve", fake_serve)
    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr("traffic_replay.cli.time.sleep", lambda _: None)
    args = argparse.Namespace(
        port=0, workdir=str(tmp_path), duration=1, quiet=True,
        format="json", tolerance_ms=60.0)
    with pytest.raises(ValueError, match=match):
        cmd_validate(args)
