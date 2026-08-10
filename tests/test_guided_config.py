"""Customer-guided workload and SLA setup contracts."""
from __future__ import annotations

import json
import io
import sys
from pathlib import Path

import pytest

from traffic_replay.cli import (
    _load_run_config_with_sla,
    main,
)


def _telemetry(path: Path) -> None:
    path.write_text(
        "prompt_tokens,completion_tokens,cached_tokens\n"
        "100,20,0\n200,40,50\n300,60,150\n400,80,200\n",
        encoding="utf-8")


def _init_args(source: Path, output: Path) -> list[str]:
    return [
        "init-config",
        "--telemetry", str(source),
        "--provider", "openai",
        "--name", "customer_workload",
        "--response-start-ms", "500",
        "--response-finish-ms", "3000",
        "--success-percent", "99",
        "--abandon-start-ms", "2000",
        "--abandon-finish-ms", "10000",
        "--sla-source", "Customer contract dated 2026-08-10",
        "--host", "https://workspace.cloud.databricks.com",
        "--endpoint", "customer-endpoint",
        "--requests", "4",
        "--duration", "60",
        "--calibrate-requests", "1",
        "--out-dir", str(output),
    ]


def test_noninteractive_init_writes_separate_owned_inputs_and_plain_preview(
        tmp_path, capsys):
    source = tmp_path / "telemetry.csv"
    output = tmp_path / "generated"
    _telemetry(source)

    assert main(_init_args(source, output)) == 0
    shown = capsys.readouterr().out
    profile = json.loads((output / "workload-profile.json").read_text())
    sla = json.loads((output / "customer-sla.json").read_text())
    run = json.loads((output / "run-config.json").read_text())

    assert profile["extraction"] == {
        "total_records": 4,
        "usable_input_records": 4,
        "dropped_input_records": 0,
        "usable_output_records": 4,
        "dropped_output_records": 0,
        "usable_cache_records": 4,
        "dropped_cache_records": 0,
        "complete_joint_records": 4,
        "dropped_incomplete_joint_records": 0,
    }
    assert sla["targets_are"] == "Customer contract dated 2026-08-10"
    assert sla["ttft_ms"] == {"p95": 500.0}
    assert sla["ttfg_ms"] == {"p95": 3000.0}
    assert sla["success_rate"] == 0.99
    assert sla["hard_timeouts"]["ttft_s"] == 2.0
    assert sla["hard_timeouts"]["ttfg_s"] == 10.0
    assert run["customer_sla_path"] == str(output / "customer-sla.json")
    assert "acceptance_targets" not in run
    assert run["ttft_definition"] == "first_visible"
    assert "CONFIG VALID - ZERO ENDPOINT TRAFFIC SENT" in shown
    assert "prompt tokens p50" in shown and "p90" in shown
    assert "cache meaning: reusable prompt-token share" in shown
    assert "recovered rows: 4; dropped input/output/cache: 0/0/0" in shown
    assert "95% must show visible answer content within 500 ms" in shown
    assert "4 measured replay + 1 calibration + 0 preflight" in shown
    assert "COST NOT CALCULATED" in shown


def test_check_config_is_repeatable_and_sends_no_endpoint_traffic(
        tmp_path, capsys, monkeypatch):
    source = tmp_path / "telemetry.csv"
    output = tmp_path / "generated"
    _telemetry(source)
    main(_init_args(source, output))
    capsys.readouterr()

    def forbidden(*args, **kwargs):
        raise AssertionError("check-config must not open an endpoint client")

    monkeypatch.setattr("traffic_replay.client.EndpointClient", forbidden)
    assert main([
        "check-config", "--config", str(output / "run-config.json")]) == 0
    shown = capsys.readouterr().out
    assert shown.startswith("CONFIG VALID - ZERO ENDPOINT TRAFFIC SENT")
    assert "estimated replay tokens:" in shown


def test_inline_sla_override_has_explicit_precedence(tmp_path):
    sla_path = tmp_path / "customer-sla.json"
    sla_path.write_text(json.dumps({
        "targets_are": "Customer contract dated 2026-08-10",
        "ttft_ms": {"p95": 500},
    }), encoding="utf-8")
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({
        "customer_sla_path": "customer-sla.json",
        "acceptance_targets": {
            "targets_are": "expert inline override",
            "ttft_ms": {"p95": 250},
        },
    }), encoding="utf-8")

    cfg, precedence = _load_run_config_with_sla(config_path)

    assert cfg["acceptance_targets"]["ttft_ms"]["p95"] == 250
    assert precedence == (
        "SLA precedence: inline acceptance_targets override "
        f"customer_sla_path {sla_path}")


def test_noninteractive_init_names_every_missing_plain_language_input(
        tmp_path):
    with pytest.raises(SystemExit, match="requires --telemetry"):
        main(["init-config", "--out-dir", str(tmp_path)])


def test_guided_init_rejects_ambiguous_one_hundred_percent_target(
        tmp_path):
    source = tmp_path / "telemetry.csv"
    _telemetry(source)
    args = _init_args(source, tmp_path / "generated")
    args[args.index("99")] = "100"

    with pytest.raises(SystemExit, match="less than 100"):
        main(args)


def test_guided_init_refuses_to_overwrite_owned_inputs(tmp_path):
    source = tmp_path / "telemetry.csv"
    output = tmp_path / "generated"
    _telemetry(source)
    assert main(_init_args(source, output)) == 0

    with pytest.raises(SystemExit, match="refused to overwrite"):
        main(_init_args(source, output))


def test_interactive_init_asks_plain_language_questions(
        tmp_path, monkeypatch, capsys):
    source = tmp_path / "telemetry.csv"
    output = tmp_path / "interactive"
    _telemetry(source)

    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

    answers = TerminalInput("\n".join([
        str(source),
        "openai",
        "interactive_workload",
        "500",
        "3000",
        "99",
        "2000",
        "10000",
        "Customer contract dated 2026-08-10",
        "https://workspace.cloud.databricks.com",
        "customer-endpoint",
        "4",
        "60",
    ]) + "\n")
    monkeypatch.setattr(sys, "stdin", answers)

    assert main(["init-config", "--out-dir", str(output)]) == 0
    shown = capsys.readouterr().out
    assert "95% of responses must start" in shown
    assert "What percentage of requests must succeed" in shown
    assert (output / "customer-sla.json").is_file()
