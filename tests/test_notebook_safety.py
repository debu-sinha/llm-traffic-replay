from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from scripts import pack_notebook
from traffic_replay._build_provenance import (
    PROVENANCE_FILENAME,
    source_inventory,
    validate_embedded_provenance,
)
from traffic_replay.cli import (
    _benchmark_config,
    _freeze_and_prevalidate_cli_config,
    _quota_setup_plans,
)
from traffic_replay.profile import Profile
from traffic_replay.quota_planner import plan_run_quota
from traffic_replay.runner import RunConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/smoke_test_e2e_demo.ipynb"


def _sources() -> tuple[str, list[str]]:
    source = (NOTEBOOK if NOTEBOOK.is_file()
              else ROOT / pack_notebook.NOTEBOOK_CONTRACT)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "markdown")
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    return markdown, code


def test_notebook_is_an_explicit_quota_guarded_diagnostic_canary():
    markdown, cells = _sources()
    source = "\n".join(cells)

    assert "diagnostic endpoint-contract canary" in markdown
    assert "must never be quoted for" in markdown
    for excluded_claim in (
            "latency, SLA attainment, throughput, endpoint capacity",
            "customer workload demand"):
        assert excluded_claim in markdown
    assert "configs/profile_glm52_canary_illustrative.json" in source
    assert '"--fixed-rate", "0.1", "--duration", "12"' in source
    assert '"--ttft-definition", "first_visible"' in source
    assert "rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json" \
        in source
    assert '"--max-concurrency", "1"' in source
    assert '"--max-pending-requests", "1"' in source
    assert '"--fail-on", "none"' in source
    assert "confirm_paid_canary" in source
    assert "validate_bearer_transport(HOST)" in source
    assert "rate_limit_endpoint_binding" in source
    assert "setup-traffic" in source
    assert "measured_verification = verify_run_output(run_dir)" in source
    assert "runtime_quota_admission" in source
    assert "endpoint_metadata_stability" in source
    assert "response_identity" in source
    assert 'summary.get("ttfv_ms")' in source
    assert "DIAGNOSTIC ENDPOINT-CONTRACT CANARY PASSED" in source

    assert "profile_validation_small.json" not in source
    assert "from traffic_replay.runner import RunConfig, run" not in source
    assert "sealed evidence is not green" not in source
    assert "confirm_paid_smoke" not in source


def test_notebook_payload_includes_every_test_dependency_and_its_builder():
    support = set(pack_notebook.NOTEBOOK_SUPPORT_FILES)
    for required in (
            "README.md",
            "CHANGELOG.md",
            "TODO.md",
            "MANIFEST.in",
            "setup.py",
            "scripts/build_customer_pdf.py",
            "scripts/pack_notebook.py",
            "scripts/profile_from_logs.py",
            "docs/ARCHITECTURE.md",
            "docs/PRODUCTION_TESTING.md",
            "docs/RUN_YOUR_OWN_BENCHMARK.md",
            "docs/customer/benchmark-your-own-endpoint.html",
            "docs/diagrams/architecture.excalidraw",
            "docs/diagrams/architecture.svg",
            "docs/diagrams/load-model.svg",
            "docs/diagrams/request-sequence.svg"):
        assert required in support


def test_exact_notebook_canary_has_one_replay_and_passes_offline_quota_plan(
        tmp_path):
    markdown, cells = _sources()
    source = next(cell for cell in cells if "canary_command = [" in cell)
    widget_source = next(
        cell for cell in cells if 'widgets.text("extra_body_json"' in cell)

    def flag(name: str) -> str:
        match = re.search(rf'"{re.escape(name)}", "([^"]+)"', source)
        assert match is not None, name
        return match.group(1)

    widget = re.search(
        r'dbutils\.widgets\.text\("extra_body_json", \'([^\']+)\'',
        widget_source)
    assert widget is not None
    extra_body_json = widget.group(1)
    assert json.loads(extra_body_json) == {"reasoning_effort": "none"}

    profile_path = ROOT / flag("--profile")
    limits_path = ROOT / flag("--rate-limits")
    profile = Profile.from_json(profile_path)
    rate = float(flag("--fixed-rate"))
    duration = int(flag("--duration"))
    args = SimpleNamespace(
        cmd="benchmark",
        host="https://workspace.cloud.databricks.com",
        endpoint="databricks-glm-5-2",
        auth_profile=None,
        token_env="DATABRICKS_TOKEN",
        model=None,
        extra_body=extra_body_json,
        fixed_rate=rate,
        sizing_concurrency=None,
        legacy_concurrency=None,
        duration=duration,
        input_tokens="10000",
        output_tokens="200",
        cache_fraction="0.3,0.7",
        prompts=None,
        profile=str(profile_path),
        ttft_p50=None,
        ttft_p90=None,
        ttft_p95=None,
        ttft_p99=None,
        ttfg_p50=None,
        ttfg_p90=None,
        ttfg_p95=None,
        ttfg_p99=None,
        success_rate=None,
        ttft_definition=flag("--ttft-definition"),
        out_dir="/tmp/notebook-canary-plan",
        max_concurrency=int(flag("--max-concurrency")),
        max_pending_requests=int(flag("--max-pending-requests")),
        title=None,
        label=None,
        rate_limits_file=str(limits_path),
        skip_preflight=False,
        probe_extra_body=[],
    )
    cfg = _benchmark_config(args)
    frozen = tmp_path / "frozen-inputs"
    frozen.mkdir()
    cfg, prevalidated = _freeze_and_prevalidate_cli_config(cfg, frozen)
    rc = RunConfig(**cfg)
    replay_count = len(prevalidated.full_schedule["timestamps"])
    setup = _quota_setup_plans(
        cfg, args,
        representative_plans=prevalidated.representative_plans)
    plan = plan_run_quota(
        rc, setup_plans=setup, prevalidated=prevalidated)

    assert replay_count == 1
    assert len(setup) == 2
    assert min(rc.calibrate_n, replay_count) == 1
    assert len(setup) + min(rc.calibrate_n, replay_count) + replay_count == 4
    assert plan["may_start"] is True
    assert plan["status"] == "within_configured_harness_warning_budget"
    assert plan["logical_replay_requests"] == 1
    assert plan["planned_physical_attempts_worst_case"] == 12
    assert rc.endpoint["extra_body"] == {"reasoning_effort": "none"}
    assert plan["windows"]["input_tokens_per_minute"]["planned_peak"] == 89142
    assert plan["windows"]["input_tokens_per_minute"][
        "ratio_to_configured_limit"] == pytest.approx(0.44571)
    assert plan["windows"]["output_tokens_per_minute"]["planned_peak"] == 4320
    assert plan["windows"]["output_tokens_per_minute"][
        "ratio_to_configured_limit"] == pytest.approx(0.216)
    assert profile.input_tokens == {"p50": 1000.0, "p95": 2000.0}
    assert profile.output_tokens == {"p50": 320.0, "p95": 480.0}
    assert "workload shape only" in profile.provenance
    assert "reasoning controls" in profile.provenance
    assert "belong to the run configuration" in profile.provenance
    assert rc.max_output_tokens_cap == 720
    for disclosed in (
            "input p50/p95 of 1,000/2,000 tokens",
            "output-budget p50/p95 of 320/480 tokens",
            "output safety cap is 720 tokens",
            "at most 12 physical POST attempts",
            'default direct-endpoint `extra_body_json` of '
            '`{"reasoning_effort":"none"}`',
            "changing that control is replanned",
            "89,142 input tokens/minute",
            "4,320 output tokens/minute",
            "44.571% and 21.6%",
            "same field is confirmed for AI Gateway",
            "intentionally targets the direct "
            "`/serving-endpoints/.../invocations` path",
            "cannot support an AI Gateway production-capacity claim",
            "cannot verify the configured Enterprise workspace tier"):
        assert disclosed in markdown


def test_generated_notebook_provenance_binds_exact_instrument_files(tmp_path):
    files = {
        "traffic_replay/__init__.py": '__version__ = "0.6.0"\n',
        "traffic_replay/worker.py": "def value():\n    return 7\n",
        "traffic_replay/data/validation.json": '{"expected":7}\n',
    }

    pack_notebook._add_embedded_provenance(files, "a" * 40)

    target = f"traffic_replay/{PROVENANCE_FILENAME}"
    record = json.loads(files[target])
    tree, count = pack_notebook._source_inventory_from_payload(files)
    valid, reason = validate_embedded_provenance(
        record, expected_version="0.6.0",
        source_tree_sha256=tree, source_file_count=count)
    assert valid is True, reason
    assert record["git_commit"] == "a" * 40
    assert record["git_dirty"] is False
    assert record["source_file_count"] == 3

    package = tmp_path / "traffic_replay"
    for relative, text in files.items():
        target_path = tmp_path / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")
    installed_tree, installed_files = source_inventory(package)
    assert installed_tree == tree
    assert len(installed_files) == count


def test_payload_source_commit_survives_an_artifact_only_commit(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=True)
        return result.stdout.strip()

    git("init", "-q")
    (repo / "payload.txt").write_text("source bytes\n", encoding="utf-8")
    (repo / "artifact.ipynb").write_text("first\n", encoding="utf-8")
    git("add", "payload.txt", "artifact.ipynb")
    git("-c", "user.name=Debu Sinha", "-c",
        "user.email=debusinha2009@gmail.com", "commit", "-qm", "source")
    source_commit = git("rev-parse", "HEAD")

    (repo / "artifact.ipynb").write_text("generated\n", encoding="utf-8")
    git("add", "artifact.ipynb")
    git("-c", "user.name=Debu Sinha", "-c",
        "user.email=debusinha2009@gmail.com", "commit", "-qm", "artifact")
    assert git("rev-parse", "HEAD") != source_commit

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", repo.as_uri(), str(shallow)],
        capture_output=True, text=True, check=True)

    monkeypatch.setattr(pack_notebook, "ROOT", repo)
    assert pack_notebook._payload_source_commit(["payload.txt"]) == \
        source_commit

    (repo / "payload.txt").write_text("dirty bytes\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="requires a clean Git tree"):
        pack_notebook._payload_source_commit(["payload.txt"])

    monkeypatch.setattr(pack_notebook, "ROOT", shallow)
    with pytest.raises(SystemExit, match="requires complete Git history"):
        pack_notebook._payload_source_commit(["payload.txt"])


def test_notebook_contract_ignores_generated_payload_but_not_semantics(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=True)
        return result.stdout.strip()

    def notebook(title: str, payload: str, digest: str, count: int) -> str:
        return json.dumps({
            "cells": [
                {
                    "cell_type": "markdown", "metadata": {},
                    "source": [
                        f"# {title}\n",
                        "Self-contained runnable payload "
                        f"(v0.6.0, {count} payload files, {count} pytest cases)",
                    ],
                },
                {
                    "cell_type": "code", "metadata": {},
                    "source": [
                        'PACKED_VERSION = "0.6.0"\n',
                        f"EXPECTED_PAYLOAD_FILES = {count}\n",
                        f'PAYLOAD_SHA256 = "{digest}"\n',
                        f'PAYLOAD = "{payload}"\n',
                        f"# run the full pytest suite ({count} cases)\n",
                        f"EXPECTED_PYTEST_CASES = {count}\n",
                    ],
                    "outputs": [], "execution_count": None,
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }, indent=1) + "\n"

    git("init", "-q")
    payload = repo / "payload.txt"
    artifact = repo / "artifact.ipynb"
    payload.write_text("source bytes\n", encoding="utf-8")
    artifact.write_text(
        notebook("diagnostic", "YWJj", "a" * 64, 7), encoding="utf-8")
    git("add", "payload.txt", "artifact.ipynb")
    git("-c", "user.name=Debu Sinha", "-c",
        "user.email=debusinha2009@gmail.com", "commit", "-qm", "source")
    source_commit = git("rev-parse", "HEAD")

    artifact.write_text(
        notebook("diagnostic", "ZGVm", "b" * 64, 11), encoding="utf-8")
    git("add", "artifact.ipynb")
    git("-c", "user.name=Debu Sinha", "-c",
        "user.email=debusinha2009@gmail.com", "commit", "-qm", "artifact")
    monkeypatch.setattr(pack_notebook, "ROOT", repo)
    monkeypatch.setattr(pack_notebook, "NOTEBOOK", artifact)
    assert pack_notebook._normalized_notebook_contract() == \
        pack_notebook._notebook_contract_at_commit(source_commit)

    artifact.write_text(
        notebook("changed semantics", "ZGVm", "b" * 64, 11),
        encoding="utf-8")
    git("add", "artifact.ipynb")
    git("-c", "user.name=Debu Sinha", "-c",
        "user.email=debusinha2009@gmail.com", "commit", "-qm", "semantic")
    assert pack_notebook._normalized_notebook_contract() != \
        pack_notebook._notebook_contract_at_commit(source_commit)
