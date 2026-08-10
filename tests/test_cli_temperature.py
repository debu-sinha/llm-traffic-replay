"""High-level commands preserve numeric temperature versus field omission."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import traffic_replay.cli as cli
from traffic_replay.client import EndpointConfig, serialize_request_body


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profile_validation_small.json"


def _command_argv(command: str, *extra: str) -> list[str]:
    argv = [
        command,
        "--host", "https://example.invalid",
        "--endpoint", "candidate",
    ]
    if command == "benchmark":
        argv.extend([
            "--profile", str(PROFILE),
            "--sizing-concurrency", "1",
            "--duration", "1",
            "--skip-preflight",
        ])
    elif command == "sweep":
        argv.extend([
            "--profile", str(PROFILE),
            "--rate", "1,2",
            "--duration", "1",
            "--diagnostic-only",
            "--skip-preflight",
        ])
    elif command == "quickstart":
        argv.extend([
            "--profile", str(PROFILE),
            "--sizing-concurrency", "1",
        ])
    else:  # pragma: no cover - test helper guard
        raise AssertionError(command)
    argv.extend(extra)
    return argv


def _benchmark_or_sweep_config(
        monkeypatch: pytest.MonkeyPatch, command: str,
        *extra: str) -> dict:
    captured: dict = {}

    def capture(args) -> int:
        captured.update(cli._benchmark_config(args))
        return 0

    monkeypatch.setattr(cli, f"cmd_{command}", capture)
    assert cli.main(_command_argv(command, *extra)) == 0
    return captured


def _payload(endpoint: dict) -> dict:
    cfg = EndpointConfig(**endpoint)
    raw = serialize_request_body(
        cfg, [{"role": "user", "content": "hello"}], 8,
        include_usage=True)
    return json.loads(raw)


@pytest.mark.parametrize("command", ["benchmark", "sweep"])
@pytest.mark.parametrize(
    "flags,expected,present",
    [
        ((), 0.0, True),
        (("--temperature", "0.375"), 0.375, True),
        (("--omit-temperature",), None, False),
    ],
)
def test_benchmark_and_sweep_config_and_wire_temperature(
        monkeypatch, command, flags, expected, present):
    cfg = _benchmark_or_sweep_config(monkeypatch, command, *flags)

    assert "temperature" in cfg["endpoint"]
    assert cfg["endpoint"]["temperature"] == expected
    payload = _payload(cfg["endpoint"])
    assert ("temperature" in payload) is present
    if present:
        assert payload["temperature"] == expected


@pytest.mark.parametrize(
    "flags,expected,present",
    [
        ((), 0.0, True),
        (("--temperature", "0.375"), 0.375, True),
        (("--omit-temperature",), None, False),
    ],
)
def test_quickstart_config_and_wire_temperature(
        tmp_path, flags, expected, present):
    output = tmp_path / "quickstart.json"
    argv = _command_argv("quickstart", "--out", str(output), *flags)

    assert cli.main(argv) == 0
    cfg = json.loads(output.read_text())
    assert "temperature" in cfg["endpoint"]
    assert cfg["endpoint"]["temperature"] == expected
    payload = _payload(cfg["endpoint"])
    assert ("temperature" in payload) is present
    if present:
        assert payload["temperature"] == expected


@pytest.mark.parametrize("command", ["benchmark", "sweep", "quickstart"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_high_level_commands_reject_nonfinite_temperature_before_execution(
        command, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(_command_argv(command, "--temperature", value))
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["benchmark", "sweep", "quickstart"])
def test_temperature_and_omission_are_mutually_exclusive(command):
    with pytest.raises(SystemExit) as exc:
        cli.main(_command_argv(
            command, "--temperature", "0.0", "--omit-temperature"))
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["benchmark", "sweep", "quickstart"])
def test_temperature_help_explains_default_and_omission(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--temperature FINITE_FLOAT" in help_text
    assert "default: 0.0" in help_text
    assert "--omit-temperature" in help_text
    assert "distinct from sending numeric 0.0" in " ".join(help_text.split())
