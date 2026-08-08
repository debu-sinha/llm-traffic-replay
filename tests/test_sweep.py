"""The rate ladder.

The axis is arrival rate, not concurrency, and that is a correctness choice
rather than a convenience. An open-loop generator cannot hold a concurrency:
in-flight is arrival rate times service time, and service time rises under
load, so fixing the rate moves the concurrency. Offering concurrency as an
input would mean either lying about it or going closed loop, and closed loop
is what bakes coordinated omission into every other sweep in the category.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from traffic_replay.cli import _rungs


def test_a_range_becomes_a_geometric_ladder():
    """Geometric because the interesting region is multiplicative: 1 to 2
    matters as much as 16 to 32, and a linear ladder spends most of its
    rungs past the knee."""
    assert _rungs("1:32") == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert _rungs("1:16:5") == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_an_explicit_list_is_taken_as_given_and_sorted():
    assert _rungs("10,2,5") == [2.0, 5.0, 10.0]


def test_distinct_rates_cannot_collide_in_directory_or_report_labels():
    from traffic_replay.sweep_artifacts import rate_label

    rates = _rungs("1.0000001,1.0000002")
    labels = [rate_label(rate) for rate in rates]
    assert len(set(labels)) == len(rates)
    assert labels == ["1.0000001", "1.0000002"]


def test_nonsense_is_refused_rather_than_producing_a_silent_ladder():
    # a loop rather than parametrize, because the stdlib runner has no marks
    for bad in ("32:1", "0:10", "-5:10", "abc", "", "1:2:3:4", "0", "-3",
                "1,,2", ",1", "1,"):
        try:
            _rungs(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} should have been refused")


def _rung(rate, kind, held=None, err=0.0):
    return {"rate": rate, "kind": kind, "text": f"{kind} at {rate}",
            "dir": f"/tmp/r{rate}", "held": held, "achieved_rps": rate,
            "err": err, "ttft_p50": 100.0, "ttft_p95": 200.0,
            "e2e_p50": 300.0, "source_position": 0,
            "request_rows": 0, "replay_rows": 0,
            "calibration_rows": 0, "sizing_rows": 0,
            "preflight_rows": 0, "probe_rows": 0, "other_rows": 0,
            "unknown_attempt_rows": 0}


class _Args:
    endpoint = "my-endpoint"
    cooldown = 0
    _cooldown_events = 0
    _preflight_evidence = {
        "skipped": True, "attempted": 0, "reachable": 0, "readable": 0,
        "reasoning_probe_requests": 0,
    }


class _ReportSink:
    """Keep prose tests focused on rendering; sealing has adversarial tests."""

    def __init__(self, path):
        self.path = path

    def seal(self, body, rungs, **_result):
        (self.path / "sweep.md").write_text(body)


def _report(rungs, path):
    from traffic_replay.cli import _sweep_report
    return _sweep_report(rungs, _ReportSink(path), _Args())


def _base_config(tmp_path: Path, out_dir: Path | None = None) -> dict:
    source = (Path(__file__).parents[1] / "traffic_replay" / "data" /
              "profile_validation_small.json")
    profile = tmp_path / "source-profile.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    if not profile.exists():
        profile.write_bytes(source.read_bytes())
    return {
        "endpoint": {
            "base_url": "https://example.invalid",
            "path": "/serving-endpoints/test-endpoint/invocations",
            "auth_token_env": "TEST_TOKEN",
            "model": "test-model",
        },
        "profile_path": str(profile),
        "duration_s": 10,
        "out_dir": str(out_dir or (tmp_path / "sweep")),
        "title": "test-endpoint rate sweep",
        "max_output_tokens_cap": 24,
        "max_concurrency": 8,
        "calibrate_n": 0,
        "cpt": 4.0,
        "capture_endpoint_metadata": False,
        "measure_network_path": False,
    }


def _rung_config(base: dict, rate: float, out_dir: Path) -> dict:
    from traffic_replay.sweep_artifacts import rate_label

    cfg = json.loads(json.dumps(base))
    cfg.update(
        qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, out_dir=str(out_dir),
        title=f"{base['title']} @ {rate_label(rate)} requests/second")
    return cfg


def _sealed_run(path: Path, summary: dict, identity: str, *,
                run_config: dict, request_rows: list[dict] | None = None) -> Path:
    """Create the smallest valid v3 run accepted by the production verifier."""
    from traffic_replay.artifacts import canonical_sha256
    from traffic_replay.runner import (
        RunConfig, _effective_config, _resolved_workload_id)

    path.mkdir(parents=True)
    summary_raw = (json.dumps(summary, indent=2) + "\n").encode()
    rows = request_rows or []
    requests_raw = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode()
        for row in rows)
    (path / "summary.json").write_bytes(summary_raw)
    (path / "requests.jsonl").write_bytes(requests_raw)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    rc = RunConfig(**run_config)
    input_mode = "prompts" if rc.prompts_file else "profile"
    input_path = rc.prompts_file or rc.profile_path
    input_raw = Path(input_path).read_bytes()
    inputs = {input_mode: {
        "name": Path(input_path).name,
        "sha256": hashlib.sha256(input_raw).hexdigest(),
        "bytes": len(input_raw),
    }}
    effective = _effective_config(rc, rc)
    replay_rows = sum(row.get("phase") == "replay" for row in rows)
    manifest = {
        "manifest_schema_version": 3,
        "workload_id": _resolved_workload_id(rc, inputs),
        "logical_run_id": f"logical-{identity}",
        "run_id": f"logical-{identity}",
        "execution_id": f"execution-{identity}",
        "artifact_id": f"artifact-{identity}",
        "endpoint_base_url": rc.endpoint["base_url"],
        "endpoint_path": rc.endpoint["path"],
        "endpoint_model": rc.endpoint.get("model"),
        "input_mode": input_mode,
        "profile_sha256": inputs[input_mode]["sha256"],
        "inputs": inputs,
        "effective_config": effective,
        "effective_config_sha256": canonical_sha256(effective),
        "schedule": {"requests": replay_rows, "shard": "1/1"},
        "schedule_identity": {
            "encoding": "float64-le-seconds-from-run-start",
            "global_timestamps_sha256": digest,
            "shard_timestamps_sha256": digest,
            "global_count": replay_rows, "shard_count": replay_rows,
            "global_min_s": None, "global_max_s": None,
            "shard_min_s": None, "shard_max_s": None,
        },
        "index_identity": {
            "encoding": "int64-le", "global_indices_sha256": digest,
            "count": replay_rows, "global_count": replay_rows,
            "shard_index": 0,
            "shard_total": 1, "partition": "unsharded",
            "min": None, "max": None,
        },
        "artifacts": {
            "summary.json": {
                "sha256": hashlib.sha256(summary_raw).hexdigest(),
                "bytes": len(summary_raw),
            },
            "requests.jsonl": {
                "sha256": hashlib.sha256(requests_raw).hexdigest(),
                "bytes": len(requests_raw), "row_count": len(rows),
            },
        },
    }
    manifest_raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (path / "manifest.json").write_bytes(manifest_raw)
    completion = {
        "status": "complete", "artifact_id": manifest["artifact_id"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_bytes": len(manifest_raw), "request_rows": len(rows),
    }
    (path / ".traffic-replay-complete").write_text(
        json.dumps(completion) + "\n")
    return path


def _summary(rate: float) -> dict:
    return {
        "arrivals": {"achieved_qps_overall": rate},
        "requests_total": 1000,
        "requests_failed": 0,
        "error_rate": 0.0,
        "ttft_ms": {"p50": 100.0, "p95": 200.0},
        "e2e_ms": {"p50": 300.0},
        "concurrency": {"in_flight_p50": 2.0},
        "answers": {"answer_rate": 1.0, "judged": 1000, "answered": 1000},
        "drift": {"drift_kind": "stable", "drift_flag": False},
        "sample": {"n": 1000, "indicative_only": []},
        "sla": {"success_rate": {
            "target": 0.99, "met": True,
            "statistically_demonstrated": True,
        }},
    }


def _record(rate: float, source_position: int, run_dir: Path) -> dict:
    return {
        **_rung(rate, "ok", held=2),
        "text": "meets every acceptance target",
        "dir": str(run_dir),
        "source_position": source_position,
        "wall_s": 1.0,
        "request_rows": 0, "replay_rows": 0,
        "calibration_rows": 0, "sizing_rows": 0,
        "preflight_rows": 0, "probe_rows": 0, "other_rows": 0,
        "unknown_attempt_rows": 0,
    }


def _claim_with_rungs(tmp_path: Path, rates=(1.0,)):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    base = _base_config(tmp_path)
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    records = []
    dirs = []
    for i, rate in enumerate(rates):
        rung_root = artifact.path / f"rate_{rate:g}"
        cfg = _rung_config(base, rate, rung_root)
        d = _sealed_run(
            rung_root / "run", _summary(rate), f"rung-{i}",
            run_config=cfg)
        _verified, position = artifact.add_rung(rate, d, _summary(rate))
        records.append(_record(rate, position, d.relative_to(artifact.path)))
        dirs.append(d)
    return artifact, records, dirs


def _report_context(artifact, *, skipped=True, attempted=0, reachable=0,
                    readable=0, probes=0, cooldown=0.0, events=0):
    return {
        "endpoint": artifact._base_config["endpoint"]["path"],
        "sweep_wall_s": 1.5,
        "cooldown_s": cooldown,
        "cooldown_events": events,
        "preflight": {
            "skipped": skipped,
            "attempted": attempted,
            "reachable": reachable,
            "readable": readable,
            "reasoning_probe_requests": probes,
        },
    }


def _seal(artifact, records, *, context=None):
    from traffic_replay.sweep_artifacts import (
        render_sweep_report, sweep_outcome)

    context = context or _report_context(artifact)
    outcome = sweep_outcome(records)
    return artifact.seal(
        render_sweep_report(records, context), records,
        exit_code=outcome["exit_code"],
        highest_held_rate=outcome["highest_held_rate"],
        report_context=context)


def _rewrite_sweep_manifest(out: Path, manifest: dict) -> None:
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (out / "manifest.json").write_bytes(raw)
    completion = json.loads(
        (out / ".traffic-replay-complete").read_text())
    completion["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    completion["manifest_bytes"] = len(raw)
    (out / ".traffic-replay-complete").write_text(
        json.dumps(completion) + "\n")


def _rewrite_run_manifest(out: Path, manifest: dict) -> None:
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (out / "manifest.json").write_bytes(raw)
    completion = json.loads(
        (out / ".traffic-replay-complete").read_text())
    completion["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    completion["manifest_bytes"] = len(raw)
    (out / ".traffic-replay-complete").write_text(
        json.dumps(completion) + "\n")


def _unverified_record(rate: float) -> dict:
    return {
        "rate": rate,
        "kind": "invalid",
        "text": "rung failed before a verified report",
        "dir": f"rate_{rate:g}",
        "source_position": None,
        "held": None,
        "achieved_rps": None,
        "err": None,
        "ttft_p50": None,
        "ttft_p95": None,
        "e2e_p50": None,
        "wall_s": 0.5,
        "request_rows": None,
        "replay_rows": None,
        "calibration_rows": None,
        "sizing_rows": None,
        "preflight_rows": None,
        "probe_rows": None,
        "other_rows": None,
        "unknown_attempt_rows": None,
    }


def test_the_ceiling_is_the_highest_rung_that_HELD():
    """Every sweep in this category anchors its ceiling on the highest rung
    it managed to submit, then reports a top rung its own error rate
    disqualifies. The ceiling here is the last one that stayed valid."""
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rungs = [_rung(1, "ok", held=2), _rung(2, "ok", held=5),
             _rung(4, "miss", held=9, err=0.4)]
    code = _report(rungs, tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "Highest rate that held: 2 requests/second" in body
    assert "carried about 5 concurrent" in body
    assert "The next rung, 4 rps, missed" in body
    assert code == 0


def test_a_caution_is_not_claimed_as_a_proven_held_rung():
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rungs = [_rung(1, "ok", held=2), _rung(2, "caution", held=5)]
    _report(rungs, tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "Highest rate that held: 1 requests/second" in body
    assert "next rung, 2 rps, cautioned" in body


def test_topping_out_says_the_ceiling_may_be_higher():
    """Reporting the top rung as the ceiling when nothing failed would
    understate the endpoint."""
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    _report([_rung(1, "ok", held=2), _rung(2, "ok", held=4)], tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "top of the ladder" in body
    assert "Raise --rate" in body


def test_no_rung_holding_is_reported_and_exits_nonzero():
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    code = _report([_rung(1, "miss", err=0.5)], tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "No rung held" in body
    assert "lowest rate tested (1 rps)" in body
    assert code == 1


def test_missing_error_rate_is_not_printed_as_zero():
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rung = _rung(1, "invalid")
    rung["err"] = None
    _report([rung], tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "| 1 rps | 1.0 | - | - |" in body


def test_concurrency_is_reported_as_measured_not_as_asked():
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    _report([_rung(1, "ok", held=3)], tmp_path)
    body = (tmp_path / "sweep.md").read_text()
    assert "as measured, not as asked for" in body
    assert "| held |" in body


def test_the_config_the_sweep_builds_is_actually_a_valid_run_config():
    """The preflight adds a key RunConfig does not accept, and the single-run
    path pops it. The ladder did not, so every sweep died on rung 1 with a
    TypeError after the first run had already been paid for."""
    import copy
    import tempfile
    from pathlib import Path
    from traffic_replay.cli import _benchmark_config
    from traffic_replay.runner import RunConfig

    class A:
        host = "https://example.invalid"
        endpoint = "ep"
        auth_profile = None
        token_env = "T"
        model = None
        extra_body = None
        sizing_concurrency = None
        legacy_concurrency = None
        duration = 10
        out_dir = tempfile.mkdtemp()
        title = label = None
        input_tokens = "1000"
        output_tokens = "50"
        cache_hit_rate = "0.2,0.6"
        prompts = profile = None
        ttft_p50 = ttft_p90 = ttft_p95 = ttft_p99 = None
        ttfg_p50 = ttfg_p90 = ttfg_p95 = ttfg_p99 = None
        success_rate = 0.99

    base = _benchmark_config(A())
    base.pop("sizing_concurrency", None)
    cfg = copy.deepcopy(base)
    cfg.update(qps_base=4.0, qps_burst=4.0, qps_min=4.0, qps_max=4.0,
               rate_scale=1.0, duration_s=10,
               out_dir=str(Path(A.out_dir) / "rate_4"),
               max_concurrency=120)
    rc = RunConfig(**cfg)              # must not raise
    assert rc.qps_base == 4.0
    assert rc.sizing_concurrency is None, "the ladder sets a fixed rate"


def test_sweep_reuses_the_exact_workload_and_runs_one_preflight(monkeypatch):
    import json
    from traffic_replay.cli import main

    root = Path(tempfile.mkdtemp(prefix="sweep-exact-"))
    prompts = root / "prompts.jsonl"
    prompts.write_text('{"prompt":"real one"}\n{"prompt":"real two"}\n')
    preflight = []
    runs = []
    prior_seen = []
    sleeps = []

    representative_sets = []

    def fake_preflight(cfg, args, *, representative_plans=None):
        preflight.append(json.loads(json.dumps(cfg)))
        representative_sets.append(representative_plans)
        cfg["ttft_definition"] = "first_visible"
        args._preflight_request_rows = [
            {"phase": "preflight", "request_id": "pf-1",
             "request_attempts": 1, "first_send_unix": 1.0},
            {"phase": "preflight", "request_id": "pf-2",
             "request_attempts": 1, "first_send_unix": 2.0},
        ]
        args._preflight_evidence = {
            "skipped": False, "attempted": 2, "reachable": 2,
            "readable": 2, "reasoning_probe_requests": 0,
        }
        return None

    def fake_run(rc, quiet=False, prior_request_rows=None):
        runs.append(rc)
        prior_seen.append(list(prior_request_rows or []))
        d = Path(rc.out_dir) / "fake"
        summary = {
            "arrivals": {"achieved_qps_overall": rc.qps_base},
            "error_rate": 0.0,
            "ttft_ms": {"p50": 10.0, "p95": 20.0},
            "e2e_ms": {"p50": 30.0},
            "concurrency": {"in_flight_p50": 2.0}}
        _sealed_run(
            d, summary, f"rate-{rc.qps_base:g}",
            run_config=vars(rc).copy(),
            request_rows=list(prior_request_rows or []))
        return {"out_dir": str(d), "summary": summary}

    monkeypatch.setattr("traffic_replay.cli._check_preflight", fake_preflight)
    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr("traffic_replay.metrics._verdict",
                        lambda summary: ("ok", "held"))
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    code = main([
        "sweep", "--host", "https://ws.example", "--endpoint", "ep",
        "--rate", "1,2", "--duration", "7", "--cooldown", "3",
        "--cpt", "3.5",
        "--prompts", str(prompts), "--output-tokens", "40,90",
        "--auth-profile", "workspace-test",
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--max-concurrency", "17", "--max-pending-requests", "23",
        "--out-dir", str(root / "out")])

    assert code == 0
    assert len(preflight) == 1
    assert len(representative_sets[0]) == 2
    assert len(runs) == 2
    assert [len(rows) for rows in prior_seen] == [2, 0]
    assert sleeps == [3, 3]
    assert [r.qps_base for r in runs] == [1.0, 2.0]
    assert len({r.prompts_file for r in runs}) == 1
    for rc in runs:
        assert Path(rc.prompts_file).name == prompts.name
        assert rc.prompts_file != str(prompts)
        assert rc.input_expectations == {
            "prompts": {
                "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "bytes": len(prompts.read_bytes()),
            }}
        assert rc.max_output_tokens_cap == 135
        assert rc.endpoint["auth_profile"] == "workspace-test"
        assert rc.endpoint["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}}
        assert rc.max_concurrency == 17
        assert rc.max_pending_requests == 23
        assert rc.sizing_concurrency is None
        assert rc.calibrate_n == 0
        assert rc.cpt == 3.5
        assert rc.ttft_definition == "first_visible"
    sealed_base = json.loads((root / "out" / "sweep-base-config.json").read_text())
    assert sealed_base["calibrate_n"] == 0
    assert sealed_base["cpt"] == 3.5
    assert sealed_base["ttft_definition"] == "first_visible"
    for rate in (1, 2):
        saved = json.loads((root / "out" / f"rate_{rate}" /
                            "run-config.json").read_text())
        assert saved["prompts_file"] == str(prompts)
        assert saved["input_expectations"] == {
            "prompts": {
                "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "bytes": len(prompts.read_bytes()),
            }}
        assert saved["endpoint"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}}


def test_sweep_aggregate_seals_report_config_and_exact_rung_identities(tmp_path):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, dirs = _claim_with_rungs(tmp_path, (1.0, 2.0))
    out = _seal(artifact, records)
    manifest = verify_sweep_output(out)
    completion = json.loads(
        (out / ".traffic-replay-complete").read_text())
    manifest_raw = (out / "manifest.json").read_bytes()

    assert manifest["manifest_schema_version"] == 3
    assert manifest["artifact_type"] == "sweep"
    assert manifest["input_count"] == 2
    assert manifest["rung_count"] == 2
    assert manifest["highest_held_rate_requests_per_second"] == 2.0
    assert completion["artifact_id"] == manifest["artifact_id"]
    assert completion["manifest_sha256"] == hashlib.sha256(
        manifest_raw).hexdigest()
    assert completion["manifest_bytes"] == len(manifest_raw)
    assert set(manifest["artifacts"]) == {
        "sweep-base-config.json", "sweep.md"}
    assert all(not Path(record["dir"]).is_absolute()
               for record in manifest["rungs"])
    for source, d in zip(manifest["sources"], dirs):
        run_manifest = (d / "manifest.json").read_bytes()
        run_summary = (d / "summary.json").read_bytes()
        assert source["artifact_id"] == json.loads(
            run_manifest)["artifact_id"]
        assert source["manifest"] == {
            "sha256": hashlib.sha256(run_manifest).hexdigest(),
            "bytes": len(run_manifest),
        }
        assert source["summary"] == {
            "sha256": hashlib.sha256(run_summary).hexdigest(),
            "bytes": len(run_summary),
        }

    copied = tmp_path / "copied-sweep"
    shutil.copytree(out, copied)
    assert verify_sweep_output(copied)["artifact_id"] == manifest["artifact_id"]


@pytest.mark.parametrize("name", ["sweep.md", "sweep-base-config.json"])
def test_sweep_verifier_rejects_tampered_headline_or_config(tmp_path, name):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    with (out / name).open("ab") as handle:
        handle.write(b"tampered\n")
    with pytest.raises(ValueError, match="artifact (SHA-256|byte count) mismatch"):
        verify_sweep_output(out)


def test_sweep_verifier_rejects_a_rung_changed_after_sealing(tmp_path):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    (dirs[0] / "summary.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="artifact (SHA-256|byte count) mismatch"):
        verify_sweep_output(out)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("held", 999.0, "held disagrees"),
        ("kind", "miss", "kind disagrees"),
        ("text", "invented conclusion", "text disagrees"),
    ],
)
def test_sweep_verifier_rejects_resealed_false_headline_rows(
        tmp_path, field, value, message):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["rungs"][0][field] = value
    _rewrite_sweep_manifest(out, manifest)
    with pytest.raises(ValueError, match=message):
        verify_sweep_output(out)


def test_sweep_verifier_rejects_resealed_false_ceiling(tmp_path):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["highest_held_rate_requests_per_second"] = 8.0
    _rewrite_sweep_manifest(out, manifest)
    with pytest.raises(ValueError, match="highest held rate disagrees"):
        verify_sweep_output(out)


def test_sweep_rejects_unsealed_truncated_and_actively_written_rungs(tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    for case in ("unsealed", "truncated", "writing"):
        base = _base_config(tmp_path / case, tmp_path / f"sweep-{case}")
        artifact = SweepArtifacts.claim(tmp_path / f"sweep-{case}", base)
        cfg = _rung_config(base, 1.0, artifact.path / "rate_1")
        d = _sealed_run(
            artifact.path / "rate_1" / "run", _summary(1.0), case,
            run_config=cfg)
        if case == "unsealed":
            (d / ".traffic-replay-complete").unlink()
            expected = "missing completion marker"
        elif case == "truncated":
            (d / "summary.json").write_bytes(b'{"incomplete":')
            expected = "artifact SHA-256 mismatch"
        else:
            (d / ".traffic-replay-writing").write_text("still writing\n")
            expected = "still being written"
        with pytest.raises(ValueError, match=expected):
            artifact.add_rung(1.0, d)
        artifact.close()


def test_sweep_rejects_duplicate_directory_artifact_and_rate(tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    base = _base_config(tmp_path / "duplicate-dir-base",
                        tmp_path / "duplicate-dir")
    duplicate_dir = SweepArtifacts.claim(tmp_path / "duplicate-dir", base)
    one_cfg = _rung_config(base, 1.0, duplicate_dir.path / "rate_1")
    first = _sealed_run(
        duplicate_dir.path / "rate_1" / "run", _summary(1.0), "same-dir",
        run_config=one_cfg)
    duplicate_dir.add_rung(1.0, first)
    with pytest.raises(ValueError, match="effective config does not match"):
        duplicate_dir.add_rung(2.0, first)
    duplicate_dir.close()

    base = _base_config(tmp_path / "duplicate-artifact-base",
                        tmp_path / "duplicate-artifact")
    duplicate_artifact = SweepArtifacts.claim(
        tmp_path / "duplicate-artifact", base)
    one = _sealed_run(
        duplicate_artifact.path / "rate_1" / "run", _summary(1.0), "same-id",
        run_config=_rung_config(
            base, 1.0, duplicate_artifact.path / "rate_1"))
    two = _sealed_run(
        duplicate_artifact.path / "rate_2" / "run", _summary(2.0), "same-id",
        run_config=_rung_config(
            base, 2.0, duplicate_artifact.path / "rate_2"))
    duplicate_artifact.add_rung(1.0, one)
    with pytest.raises(ValueError, match="duplicate input artifact_id"):
        duplicate_artifact.add_rung(2.0, two)
    duplicate_artifact.close()

    base = _base_config(tmp_path / "duplicate-rate-base",
                        tmp_path / "duplicate-rate")
    duplicate_rate = SweepArtifacts.claim(tmp_path / "duplicate-rate", base)
    first_root = duplicate_rate.path / "first" / "rate_1"
    second_root = duplicate_rate.path / "second" / "rate_1"
    one = _sealed_run(
        first_root / "run", _summary(1.0), "rate-one",
        run_config=_rung_config(base, 1.0, first_root))
    two = _sealed_run(
        second_root / "run", _summary(1.0), "rate-two",
        run_config=_rung_config(base, 1.0, second_root))
    duplicate_rate.add_rung(1.0, one)
    with pytest.raises(ValueError, match="duplicate sweep rung rate"):
        duplicate_rate.add_rung(1.0, two)
    duplicate_rate.close()


def test_sweep_detects_source_or_base_config_mutation_before_seal(tmp_path):
    source_mutated, records, dirs = _claim_with_rungs(
        tmp_path / "source-mutated")
    (dirs[0] / "summary.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        source_mutated.seal(
            "# no publication\n", records, exit_code=0,
            highest_held_rate=1.0,
            report_context=_report_context(source_mutated))
    assert not (source_mutated.path / ".traffic-replay-complete").exists()
    source_mutated.close()

    config_mutated, records, _dirs = _claim_with_rungs(
        tmp_path / "config-mutated")
    (config_mutated.path / "sweep-base-config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="base config changed"):
        config_mutated.seal(
            "# no publication\n", records, exit_code=0,
            highest_held_rate=1.0,
            report_context=_report_context(config_mutated))
    assert not (config_mutated.path / ".traffic-replay-complete").exists()
    config_mutated.close()


def test_sweep_claims_are_exclusive_even_for_an_existing_empty_path(tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    requested = tmp_path / "same-output"
    requested.mkdir()
    base = _base_config(tmp_path / "claim-base", requested)
    first = SweepArtifacts.claim(requested, base)
    second = SweepArtifacts.claim(requested, base)
    try:
        assert first.path != requested
        assert second.path not in {requested, first.path}
        assert (first.path / ".traffic-replay-writing").is_file()
        assert (second.path / ".traffic-replay-writing").is_file()
    finally:
        first.close()
        second.close()


def test_sweep_cannot_claim_completion_before_manifest_is_durable(
        tmp_path, monkeypatch):
    import traffic_replay.sweep_artifacts as sweep_artifacts

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    original = sweep_artifacts._atomic_text

    def fail_manifest(dir_fd, name, value):
        if name == "manifest.json":
            raise OSError("injected manifest write failure")
        return original(dir_fd, name, value)

    monkeypatch.setattr(sweep_artifacts, "_atomic_text", fail_manifest)
    with pytest.raises(OSError, match="injected manifest write failure"):
        _seal(artifact, records)
    assert (artifact.path / ".traffic-replay-writing").exists()
    assert not (artifact.path / ".traffic-replay-complete").exists()
    assert not (artifact.path / "manifest.json").exists()
    artifact.close()


def test_sweep_refuses_arbitrary_prose_even_when_the_caller_supplies_matching_numbers(
        tmp_path):
    from traffic_replay.sweep_artifacts import sweep_outcome

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    outcome = sweep_outcome(records)
    with pytest.raises(ValueError, match="not the canonical report"):
        artifact.seal(
            "# Highest rate that held: 999999 requests/second\n",
            records, exit_code=outcome["exit_code"],
            highest_held_rate=outcome["highest_held_rate"],
            report_context=_report_context(artifact))
    assert not (artifact.path / "sweep.md").exists()
    artifact.close()


def test_unverified_attempt_after_an_ok_rung_is_invalid_and_has_no_ceiling(
        tmp_path):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    records.append(_unverified_record(2.0))
    out = _seal(artifact, records)
    manifest = verify_sweep_output(out)
    report = (out / "sweep.md").read_text()

    assert manifest["exit_code"] == 2
    assert manifest["sweep_valid"] is False
    assert manifest["highest_held_rate_requests_per_second"] is None
    assert "INVALID SWEEP" in report
    assert "makes no capacity conclusion" in report
    assert "Highest rate that held" not in report


def test_sweep_verifier_rejects_an_intermediate_symlink_escape(tmp_path):
    from traffic_replay.sweep_artifacts import verify_sweep_output

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    rate_dir = out / "rate_1"
    outside = tmp_path / "outside-rate-1"
    rate_dir.rename(outside)
    rate_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="not a regular directory"):
        verify_sweep_output(out)


@pytest.mark.parametrize("mutation", ["model", "rate", "workload"])
def test_sweep_rejects_a_sealed_run_from_another_experiment(
        tmp_path, mutation):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    base = _base_config(tmp_path, tmp_path / "sweep")
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    actual = json.loads(json.dumps(base))
    if mutation == "model":
        actual["endpoint"]["model"] = "another-model"
    elif mutation == "rate":
        actual.update(qps_base=2.0, qps_burst=2.0, qps_min=2.0, qps_max=2.0,
                      rate_scale=1.0)
    else:
        other = tmp_path / "other" / "source-profile.json"
        other.parent.mkdir()
        profile = json.loads(Path(base["profile_path"]).read_text())
        profile["name"] = "different-workload"
        other.write_text(json.dumps(profile) + "\n")
        actual["profile_path"] = str(other)
    actual.update(
        out_dir=str(artifact.path / "rate_1"),
        title=f"{base['title']} @ 1 requests/second")
    if mutation != "rate":
        actual.update(qps_base=1.0, qps_burst=1.0, qps_min=1.0, qps_max=1.0,
                      rate_scale=1.0)
    run_dir = _sealed_run(
        artifact.path / "rate_1" / "run", _summary(1.0), mutation,
        run_config=actual)

    with pytest.raises(ValueError, match=(
            "effective config does not match|workload inputs do not match|"
            "profile bytes do not match|workload_id does not match")):
        artifact.add_rung(1.0, run_dir)
    artifact.close()


def test_sweep_rejects_duplicate_json_keys_in_a_manifest_bound_summary(tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts

    base = _base_config(tmp_path, tmp_path / "sweep")
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    run_dir = _sealed_run(
        artifact.path / "rate_1" / "run", _summary(1.0), "duplicate-json",
        run_config=_rung_config(base, 1.0, artifact.path / "rate_1"))
    raw = b'{"error_rate":0,"error_rate":1}\n'
    (run_dir / "summary.json").write_bytes(raw)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["artifacts"]["summary.json"] = {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="duplicate (object )?key"):
        artifact.add_rung(1.0, run_dir)
    artifact.close()


def test_per_rung_calibration_invalidates_the_capacity_conclusion(tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts, verify_sweep_output

    base = _base_config(tmp_path, tmp_path / "sweep")
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    run_dir = _sealed_run(
        artifact.path / "rate_1" / "run", _summary(1.0), "calibrated",
        run_config=_rung_config(base, 1.0, artifact.path / "rate_1"),
        request_rows=[{
            "phase": "calibration", "request_id": "cal-1",
            "request_attempts": 1, "first_send_unix": 1.0}])
    _summary_value, position = artifact.add_rung(1.0, run_dir)
    record = _record(1.0, position, run_dir.relative_to(artifact.path))
    record.update(artifact.rung_accounting(position))
    out = _seal(artifact, [record])
    manifest = verify_sweep_output(out)

    assert manifest["exit_code"] == 2
    assert manifest["sweep_valid"] is False
    assert manifest["highest_held_rate_requests_per_second"] is None
    assert "calibration request" in manifest["invalid_reasons"][0]


@pytest.mark.parametrize("sent_at", [None, -1.0])
def test_unknown_provider_attempt_from_setup_traffic_invalidates_sweep(
        tmp_path, sent_at):
    from traffic_replay.sweep_artifacts import SweepArtifacts, verify_sweep_output

    base = _base_config(tmp_path, tmp_path / "sweep")
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    run_dir = _sealed_run(
        artifact.path / "rate_1" / "run", _summary(1.0), "unknown-probe",
        run_config=_rung_config(base, 1.0, artifact.path / "rate_1"),
        request_rows=[{
            "phase": "probe", "request_id": "probe-unknown",
            "request_attempts": (None if sent_at is None else 1),
            "first_send_unix": sent_at}])
    _summary_value, position = artifact.add_rung(1.0, run_dir)
    record = _record(1.0, position, run_dir.relative_to(artifact.path))
    record.update(artifact.rung_accounting(position))
    context = _report_context(
        artifact, skipped=False, attempted=0, reachable=0, readable=0,
        probes=1)
    out = _seal(artifact, [record], context=context)
    manifest = verify_sweep_output(out)

    assert manifest["sweep_valid"] is False
    assert manifest["exit_code"] == 2
    assert any("unknown provider-attempt" in reason
               for reason in manifest["invalid_reasons"])


def test_preflight_and_probe_rows_are_manifest_bound_once_on_the_first_rung(
        tmp_path):
    from traffic_replay.sweep_artifacts import SweepArtifacts, verify_sweep_output

    base = _base_config(tmp_path, tmp_path / "sweep")
    artifact = SweepArtifacts.claim(tmp_path / "sweep", base)
    records = []
    phase_rows = [
        {"phase": "preflight", "request_id": "pf-1",
         "request_attempts": 1, "first_send_unix": 1.0},
        {"phase": "preflight", "request_id": "pf-2",
         "request_attempts": 1, "first_send_unix": 2.0},
        {"phase": "probe", "request_id": "probe-1",
         "request_attempts": 1, "first_send_unix": 3.0},
    ]
    for index, rate in enumerate((1.0, 2.0)):
        root = artifact.path / f"rate_{rate:g}"
        run_dir = _sealed_run(
            root / "run", _summary(rate), f"traffic-{index}",
            run_config=_rung_config(base, rate, root),
            request_rows=phase_rows if index == 0 else [])
        _summary_value, position = artifact.add_rung(rate, run_dir)
        record = _record(rate, position, run_dir.relative_to(artifact.path))
        record.update(artifact.rung_accounting(position))
        records.append(record)
    context = _report_context(
        artifact, skipped=False, attempted=2, reachable=2, readable=2,
        probes=1)
    out = _seal(artifact, records, context=context)
    manifest = verify_sweep_output(out)
    report = (out / "sweep.md").read_text()

    assert manifest["sources"][0]["preflight_rows"] == 2
    assert manifest["sources"][0]["probe_rows"] == 1
    assert manifest["sources"][1]["preflight_rows"] == 0
    assert manifest["sources"][1]["probe_rows"] == 0
    assert "2 preflight, 1 probe" in report
    assert "sequential and stateful" in report
    assert "proves neither QPH recovery" in report


def test_higher_pass_after_lower_failure_is_invalid_not_a_ceiling():
    from traffic_replay.sweep_artifacts import sweep_outcome

    low = _rung(1.0, "miss")
    high = _rung(2.0, "ok")
    outcome = sweep_outcome([low, high])
    assert outcome["exit_code"] == 2
    assert outcome["highest_held_rate"] is None
    assert outcome["non_monotonic"] is True


def test_a_manifest_bound_invalid_rung_removes_an_earlier_capacity_conclusion():
    from traffic_replay.sweep_artifacts import sweep_outcome

    outcome = sweep_outcome([_rung(1.0, "ok"), _rung(2.0, "invalid")])
    assert outcome["exit_code"] == 2
    assert outcome["highest_held_rate"] is None
    assert outcome["invalid_reports"]


def test_verify_sweep_command_rederives_a_sealed_conclusion(tmp_path, capsys):
    from traffic_replay.cli import main

    artifact, records, _dirs = _claim_with_rungs(tmp_path)
    out = _seal(artifact, records)
    assert main(["verify-sweep", str(out), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["sweep_valid"] is True
    assert payload["highest_held_rate_requests_per_second"] == 1.0


def test_preflight_preserves_metadata_rows_and_marks_exception_attempt_unknown(
        tmp_path, monkeypatch):
    import time
    from traffic_replay.client import RequestResult
    from traffic_replay.cli import _preflight

    calls = 0

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, _messages, max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("provider outcome unknown")
            now = time.time()
            return RequestResult(
                request_id=request_id, scheduled_s=scheduled_s,
                dispatch_lag_ms=dispatch_lag_ms, t_send_unix=now,
                ttfb_ms=1.0, ttft_ms=1.0, ttfr_ms=None, ttfv_ms=1.0,
                e2e_ms=2.0, status=200, ok=True, error=None,
                content_chunks=1, interchunk_max_ms=None,
                finish_reason="stop", prompt_tokens=max(1, intended[0]),
                completion_tokens=1, cached_tokens=0,
                cached_tokens_source="test",
                intended_input_tokens=intended[0],
                intended_output_tokens=intended[1],
                intended_cache_fraction=intended[2], doc_id=intended[3],
                chars_sent=chars_sent, stream_complete=True,
                visible_content_seen=True, first_send_unix=now,
                max_tokens_requested=max_tokens, request_attempts=1,
                connection_attempts=1)

    monkeypatch.setattr("traffic_replay.client.EndpointClient", Client)
    result = _preflight(_base_config(tmp_path))
    rows = result["_request_rows"]

    assert result["attempted"] == 2
    assert result["reachable"] == 1
    assert len(rows) == 2
    assert [row["phase"] for row in rows] == ["preflight", "preflight"]
    assert rows[0]["request_attempts"] == 1
    assert rows[1]["request_attempts"] is None
    assert rows[1]["connection_attempts"] is None
    assert all("messages" not in row and "content" not in row for row in rows)
    assert all(len(row["request_body_sha256"]) == 64 for row in rows)
