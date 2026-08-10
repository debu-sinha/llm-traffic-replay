"""High-level calibration controls are explicit, bounded, and non-drifting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import traffic_replay.cli as cli


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profile_validation_small.json"


def _argv(command: str, *extra: str) -> list[str]:
    common = [
        command,
        "--host", "https://example.invalid",
        "--endpoint", "candidate",
        "--profile", str(PROFILE),
    ]
    if command == "benchmark":
        common.extend([
            "--sizing-concurrency", "1", "--duration", "1",
            "--skip-preflight",
        ])
    elif command == "quickstart":
        common.extend(["--sizing-concurrency", "1"])
    elif command == "sweep":
        common.extend([
            "--rate", "1,2", "--duration", "1", "--diagnostic-only",
            "--skip-preflight",
        ])
    else:  # pragma: no cover - helper guard
        raise AssertionError(command)
    common.extend(extra)
    return common


@pytest.mark.parametrize("value", ["0", "7", "10000"])
def test_benchmark_maps_bounded_calibration_count(monkeypatch, value):
    captured: dict = {}

    def capture(args) -> int:
        captured.update(cli._benchmark_config(args))
        return 0

    monkeypatch.setattr(cli, "cmd_benchmark", capture)
    assert cli.main(_argv("benchmark", "--calibrate-requests", value)) == 0
    assert captured["calibrate_n"] == int(value)


def test_benchmark_default_preserves_runconfig_calibration(monkeypatch):
    captured: dict = {}

    def capture(args) -> int:
        captured.update(cli._benchmark_config(args))
        return 0

    monkeypatch.setattr(cli, "cmd_benchmark", capture)
    assert cli.main(_argv("benchmark")) == 0
    assert captured["calibrate_n"] == 12


def test_quickstart_maps_calibration_count(tmp_path):
    out = tmp_path / "quickstart.json"
    assert cli.main(_argv(
        "quickstart", "--calibrate-requests", "3", "--out", str(out))) == 0
    assert json.loads(out.read_text())["calibrate_n"] == 3


@pytest.mark.parametrize("command", ["benchmark", "quickstart"])
@pytest.mark.parametrize("value", ["-1", "10001", "1.5", "nope"])
def test_high_level_commands_reject_invalid_calibration_count(command, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv(command, "--calibrate-requests", value))
    assert exc.value.code == 2


def test_sweep_has_no_calibration_override_and_forces_zero(monkeypatch):
    captured: dict = {}

    def capture(args) -> int:
        captured.update(cli._benchmark_config(args))
        return 0

    monkeypatch.setattr(cli, "cmd_sweep", capture)
    assert cli.main(_argv("sweep")) == 0
    # The command config remains backwards-compatible; cmd_sweep seals its
    # per-rung base with calibrate_n=0 before execution.
    assert "calibrate_n" not in captured
    with pytest.raises(SystemExit) as exc:
        cli.main(_argv("sweep", "--calibrate-requests", "1"))
    assert exc.value.code == 2
