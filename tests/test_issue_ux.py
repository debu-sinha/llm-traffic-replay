from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_replay.cli import main
from traffic_replay.runner import RunConfig
from traffic_replay.schedule import load_trace


def _benchmark_args(tmp_path: Path) -> list[str]:
    return [
        "benchmark", "--host", "https://example.invalid",
        "--endpoint", "ep", "--requests", "1000", "--duration", "100",
        "--calibrate-requests", "0", "--skip-preflight",
        "--out-dir", str(tmp_path / "results"),
    ]


def test_benchmark_requests_materializes_and_saves_exact_count(
        tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run(rc, quiet=False, **kwargs):
        captured["rc"] = rc
        return {"out_dir": str(tmp_path / "sealed"), "summary": {}}

    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr("traffic_replay.cli._finish", lambda *args: 0)

    assert main(_benchmark_args(tmp_path)) == 0
    saved = json.loads(next(
        (tmp_path / ".traffic-replay-configs" / "runs").glob(
            "*/run-config.json")).read_text())
    trace = load_trace(
        saved["timestamps_file"], duration_cap_s=saved["duration_s"])
    assert len(trace["timestamps"]) == 1000
    assert trace["timestamps"][0] == 0
    assert trace["timestamps"][-1] < 100
    assert saved["timestamps_file"] != captured["rc"].timestamps_file
    assert "measured replay=1,000" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [
    ["--requests", "10", "--fixed-rate", "1"],
    ["--requests", "10", "--sizing-concurrency", "1"],
])
def test_exact_count_rejects_ambiguous_rate_controls(tmp_path, flags):
    with pytest.raises(SystemExit, match="use --requests"):
        main([
            "benchmark", "--host", "https://example.invalid",
            "--endpoint", "ep", "--duration", "10", "--skip-preflight",
            "--out-dir", str(tmp_path / "results"), *flags,
        ])


def test_init_databricks_reads_host_discovers_endpoint_and_writes_starter(
        tmp_path, monkeypatch, capsys):
    config = tmp_path / "databrickscfg"
    config.write_text(
        "[work]\nhost = https://workspace.cloud.databricks.com\n"
        "auth_type = databricks-cli\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))
    payload = [{
        "name": "chat-ready",
        "state": {"ready": "READY"},
        "task": "llm/v1/chat",
        "route_optimized": False,
        "config": {"served_entities": [{"entity_name": "system.ai/model"}]},
    }, {
        "name": "not-ready",
        "state": {"ready": "NOT_READY"},
        "task": "llm/v1/chat",
    }]
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(payload), stderr="", returncode=0))
    out = tmp_path / "starter.json"
    assert main([
        "init-databricks", "--auth-profile", "work",
        "--endpoint", "chat-ready", "--out", str(out),
        "--out-dir", str(tmp_path / "results"),
    ]) == 0
    cfg = json.loads(out.read_text())
    RunConfig(**cfg)
    assert cfg["endpoint"]["base_url"] == \
        "https://workspace.cloud.databricks.com"
    assert cfg["endpoint"]["auth_profile"] == "work"
    assert len(load_trace(cfg["timestamps_file"])["timestamps"]) == 3
    printed = capsys.readouterr().out
    assert "planned measured requests: 3" in printed
    assert "--verify-after-run" in printed


def test_init_databricks_requires_explicit_choice_noninteractively(
        tmp_path, monkeypatch):
    config = tmp_path / "databrickscfg"
    config.write_text("[work]\nhost = https://workspace.cloud.databricks.com\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))
    payload = [{"name": name, "state": {"ready": "READY"},
                "task": "llm/v1/chat"} for name in ("a", "b")]
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(payload), stderr="", returncode=0))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="multiple READY endpoints"):
        main(["init-databricks", "--auth-profile", "work"])


def test_verify_after_run_prints_authoritative_paths_and_decisions(
        tmp_path, monkeypatch, capsys):
    sealed = tmp_path / "sealed-run"
    receipt = tmp_path / "sealed-run-verification"
    monkeypatch.setattr(
        "traffic_replay.runner.run",
        lambda rc, quiet=False, **kwargs: {
            "out_dir": str(sealed), "summary": {}})
    monkeypatch.setattr("traffic_replay.cli._finish", lambda *args: 0)
    monkeypatch.setattr(
        "traffic_replay.run_verification.create_run_verification_receipt",
        lambda source, out: receipt)
    monkeypatch.setattr(
        "traffic_replay.run_verification.verify_run_receipt",
        lambda *args, **kwargs: {
            "decision": {
                "evidence_integrity": {"code": "VERIFIED"},
                "measurement_validity": {"code": "CAUTION"},
                "customer_sla": {"code": "NOT_EVALUATED"},
                "quota_state": {"code": "NOT_OBSERVED"},
                "endpoint_capacity": {"code": "INCONCLUSIVE"},
                "tested_load": {
                    "measured_replay_requests": 3,
                    "captured_quota_request_rows": 5,
                },
            },
        })
    argv = _benchmark_args(tmp_path)
    argv[argv.index("1000")] = "3"
    argv.append("--verify-after-run")
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "AUTHORITATIVE VERIFIED RESULT" in printed
    assert str(receipt / "verified-report.html") in printed
    assert "integrity=VERIFIED" in printed
    assert "captured_setup_and_calibration_requests: 2" in printed
