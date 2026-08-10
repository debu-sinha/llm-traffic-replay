"""Adversarial resource-boundary tests for production artifact handling."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import time

import pytest

from traffic_replay import aggregate, cli, runner
from traffic_replay.aggregate import _read_regular_bytes, _scan_request_journal
from traffic_replay.runner import RunConfig, _read_stable_bytes
from traffic_replay.schedule import (
    MAX_EXACT_ANALYSIS_REQUEST_ROWS,
    SIZING_CEILING_POISSON_HEADROOM_STDDEVS,
    conservative_sizing_qps_ceiling,
    exact_analysis_replay_budget,
    validate_exact_analysis_capacity,
)


PROFILE = Path(__file__).parents[1] / "configs/profile_validation_small.json"


def test_workload_fifo_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "prompts.jsonl"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(ValueError, match="not a regular file"):
        _read_stable_bytes(str(fifo), input_kind="prompts")
    assert time.monotonic() - started < 0.5


def test_workload_symlink_and_sparse_file_fail_before_read(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "profile.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="not a regular file"):
        _read_stable_bytes(str(link), input_kind="profile")

    sparse = tmp_path / "sparse-prompts.jsonl"
    with sparse.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="snapshot limit"):
        _read_stable_bytes(str(sparse), input_kind="prompts")


def test_workload_long_record_is_bounded(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="record limit"):
        _read_stable_bytes(str(prompts), input_kind="prompts")


@pytest.mark.parametrize("sizing", [False, True])
def test_oversize_fixed_and_sizing_schedules_fail_before_credentials(
        tmp_path, monkeypatch, sizing):
    touched = False

    def forbidden(_cfg):
        nonlocal touched
        touched = True
        raise AssertionError("credentials must not be read")

    monkeypatch.setattr(runner, "_token", forbidden)
    kwargs = {
        "endpoint": {
            "base_url": "https://workspace.example",
            "path": "/serving-endpoints/test/invocations",
        },
        "profile_path": str(PROFILE),
        "duration_s": 300,
        "qps_base": 200.0,
        "qps_burst": 200.0,
        "qps_min": 200.0,
        "qps_max": 200.0,
        "calibrate_n": 4,
        "capture_endpoint_metadata": False,
        "measure_network_path": False,
        "out_dir": str(tmp_path / "out"),
    }
    if sizing:
        kwargs["sizing_concurrency"] = 4
    with pytest.raises(ValueError, match="50,000"):
        runner.run(RunConfig(**kwargs), quiet=True)
    assert touched is False


def test_metadata_sparse_file_and_journal_long_line_are_bounded(tmp_path):
    sparse = tmp_path / "manifest.json"
    with sparse.open("wb") as handle:
        handle.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="metadata limit"):
        _read_regular_bytes(sparse)

    journal = tmp_path / "requests.jsonl"
    raw = b'{"padding":"' + b"x" * (256 * 1024) + b'"}\n'
    journal.write_bytes(raw)
    expected = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": 1,
    }
    with pytest.raises(ValueError, match="line 1 exceeds"):
        _scan_request_journal(journal, expected, lambda _row, _line: None)


def test_merge_combined_row_cap_precedes_materialization(
        tmp_path, monkeypatch):
    dirs = [tmp_path / "a", tmp_path / "b"]
    manifests = []
    for position in range(2):
        manifests.append({
            "artifacts": {
                "requests.jsonl": {
                    "sha256": f"{position + 1:064x}",
                    "bytes": 1,
                    "row_count": 30_000,
                },
            },
        })
    monkeypatch.setattr(
        aggregate, "_validated_input_dirs",
        lambda *_args, **_kwargs: (dirs, manifests))
    monkeypatch.setattr(
        aggregate, "_request_rows",
        lambda *_args, **_kwargs: pytest.fail(
            "request rows must not be materialized"))
    with pytest.raises(ValueError, match="60,000 total request rows"):
        aggregate.merge_runs(tmp_path / "out", dirs)


def test_u2m_subprocess_capture_stops_at_stdout_limit():
    command = [
        sys.executable,
        "-c",
        ("import os; os.write(2, b'e' * 1048576); "
         "os.write(1, b'x' * 1048576)"),
    ]
    with pytest.raises(runner._CLIOutputLimitError) as err:
        runner._run_cli_bounded(
            command, env=dict(os.environ), timeout_s=5.0,
            max_stdout_bytes=1024)
    assert len(err.value.captured) == 1025


def test_exact_analysis_limit_is_a_named_conservative_contract():
    assert MAX_EXACT_ANALYSIS_REQUEST_ROWS == 50_000


def test_exact_analysis_boundary_accepts_cap_and_refuses_one_over():
    assert validate_exact_analysis_capacity(
        replay_rows=49_978, calibration_rows=12,
        sizing_rows=8, setup_rows=2) == 50_000
    with pytest.raises(ValueError, match="50,001 request rows"):
        validate_exact_analysis_capacity(
            replay_rows=49_979, calibration_rows=12,
            sizing_rows=8, setup_rows=2)
    assert exact_analysis_replay_budget(
        calibration_rows=12, sizing_rows=8, setup_rows=2) == 49_978


def test_generated_sizing_ceiling_arithmetic_and_actual_schedule_fit():
    budget = exact_analysis_replay_budget(
        calibration_rows=12, sizing_rows=8, setup_rows=2)
    qps = conservative_sizing_qps_ceiling(
        300, calibration_rows=12, sizing_rows=8, setup_rows=2)
    expected = qps * 300
    assert (expected
            + SIZING_CEILING_POISSON_HEADROOM_STDDEVS * expected ** 0.5
            <= budget)

    cfg = {
        "endpoint": {
            "base_url": "https://workspace.example",
            "path": "/serving-endpoints/test/invocations",
        },
        "profile_path": str(PROFILE),
        "duration_s": 300,
        "sizing_concurrency": 10,
    }
    cli._apply_cli_sizing_resource_ceiling(cfg, setup_rows=2)
    prevalidated = runner.prevalidate_run_inputs(RunConfig(**cfg))
    counts = runner.exact_analysis_row_counts(prevalidated)
    assert validate_exact_analysis_capacity(
        **counts, setup_rows=2) <= MAX_EXACT_ANALYSIS_REQUEST_ROWS


def test_architecture_documents_the_enforced_resource_contract():
    architecture = " ".join((
        Path(__file__).parents[1] / "docs/ARCHITECTURE.md").read_text().split())
    for required in (
            "50,000 logical rows total", "16 MiB", "64 MiB", "4 MiB",
            "64 KiB", "256 MiB", "256 KiB per JSONL row",
            "eight Poisson standard deviations"):
        assert required in architecture
