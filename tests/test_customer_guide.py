from __future__ import annotations

import hashlib
import html
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import build_customer_pdf


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/customer/benchmark-your-own-endpoint.html"


def test_customer_guide_publishes_the_current_safety_contract():
    body = GUIDE.read_text(encoding="utf-8")
    assert body.count('<section class="page') == 5
    assert "body { font-family: var(--sans); font-size: 13.35px" in body
    assert "font: 12.1px/1.47 var(--mono)" in body
    page_rule = re.search(r"\.page \{(?P<body>.*?)\n    \}", body, re.S)
    assert page_rule is not None
    assert "overflow: visible" in page_rule.group("body")
    assert "overflow: hidden" not in page_rule.group("body")

    for required in (
            "200,000", "20,000", "7,200", "Queries / second / workspace",
            "4,000,000", "--ttft-definition</span> first_visible",
            "command-local, no-wait guard", "*-setup-traffic",
            "response-model identity", "pre-run/post-drain endpoint stability",
            "sweep.html", "one-sided 95% Wilson lower",
            "two preflight rows + one calibration row + one measured replay row",
            "320/480 tokens", "720-token request cap",
            "12 physical POST attempts", "89,142 input",
            "4,320 output tokens/minute", "Read five decisions",
            "exactly five independent decision dimensions",
            "first nonempty bounded response-body chunk",
            "YOUR-FRACTION-STRICTLY-BETWEEN-0-AND-1",
            "default branch may not yet contain this revision",
            "non-refusal visible content or a valid non-refusal tool call",
            "results/customer-fixed-rate/RUN-DIRECTORY-verification",
            "production-connection-policy",
            "capacity stays inconclusive",
            "Published Enterprise P2T defaults: tier and headroom not verified",
            "Verify Enterprise tier and shared-workspace headroom before paid traffic",
            "Source facts rechecked: 10 Aug 2026",
            "Current implemented/tested scope",
            "text-only, streaming OpenAI-compatible Chat Completions",
            "standard Databricks workspace-origin direct route",
            "traffic_replay adapters",
            'Serving-engineering-confirmed behavior',
            '{"reasoning_effort":"none"}',
            '{"chat_template_kwargs":{"enable_thinking":false}}',
            "public P2T endpoint",
            "system.ai.glm-5-2",
            "system.ai.databricks-glm-5-2",
            "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2",
            build_customer_pdf.STAMP):
        assert required in body
    assert body.count(
        "--endpoint-adapter</span> openai.chat_completions.sse/v1") == 3
    assert body.count("--ttft-definition</span> first_visible") == 3
    assert body.count("<tr><td>") == 5
    assert "Read eight decisions" not in body
    assert "YOUR-FRACTION-0..1" not in body
    assert "first iterated response-body/SSE line" not in body
    assert "customer's own SLA" not in body
    assert "Raise --rate" not in body
    assert "results/receipts/customer-fixed-rate" not in body
    assert "Decagon" not in body
    assert "network distance" not in body.lower()
    assert "--amber: #925400" in body
    assert "color: #53647a; font-size: 10px" in body
    assert "@media screen and (max-width: 850px)" in body
    assert ".page { width: 100%; min-height: 0" in body
    assert ".footer { position: static" in body
    assert "a:focus-visible" in body
    assert ".page { width: 8.5in; height: 11in; min-height: 11in" in body
    assert ".page::after { display: none; }" in body
    code = "\n".join(re.findall(r"<pre>(.*?)</pre>", body, re.S))
    assert re.search(r"--concurrency(?:\s|<)", code) is None


def test_published_glm_canary_numbers_are_recomputed_from_its_exact_command(
        tmp_path):
    """Bind every published canary count to the planner, not copied prose."""
    from traffic_replay.cli import (
        _benchmark_config,
        _freeze_and_prevalidate_cli_config,
        _quota_setup_plans,
    )
    from traffic_replay.quota_planner import plan_run_quota
    from traffic_replay.runner import RunConfig

    body = GUIDE.read_text(encoding="utf-8")
    match = re.search(
        r"<div class=\"code-title\"><span>GLM 5\.2 instrument canary"
        r".*?</div>\s*<pre>(?P<command>.*?)</pre>", body, re.S)
    assert match is not None
    command = html.unescape(re.sub(r"<[^>]+>", "", match.group("command")))

    def flag(name: str) -> str:
        found = re.search(
            rf"{re.escape(name)}\s+"
            rf"(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
            command)
        assert found is not None, name
        return next(value for value in found.groups() if value is not None)

    profile_path = ROOT / flag("--profile")
    limits_path = ROOT / flag("--rate-limits")
    assert flag("--fixed-rate") == "0.1"
    assert flag("--duration") == "12"
    args = SimpleNamespace(
        cmd="benchmark",
        host="https://workspace.cloud.databricks.com",
        endpoint=flag("--endpoint"),
        auth_profile=None,
        token_env="DATABRICKS_TOKEN",
        model=None,
        endpoint_adapter=flag("--endpoint-adapter"),
        production_connection_policy=None,
        extra_body=flag("--extra-body"),
        fixed_rate=float(flag("--fixed-rate")),
        sizing_concurrency=None,
        legacy_concurrency=None,
        duration=int(flag("--duration")),
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
        out_dir=str(tmp_path / "canary"),
        max_concurrency=None,
        max_pending_requests=None,
        title=None,
        label=flag("--label"),
        rate_limits_file=str(limits_path),
        skip_preflight=False,
        probe_extra_body=[],
    )
    config = _benchmark_config(args)
    assert config["endpoint"]["adapter"] == (
        "openai.chat_completions.sse/v1")
    frozen = tmp_path / "frozen-inputs"
    frozen.mkdir()
    config, prevalidated = _freeze_and_prevalidate_cli_config(config, frozen)
    run_config = RunConfig(**config)
    setup = _quota_setup_plans(
        config, args,
        representative_plans=prevalidated.representative_plans)
    plan = plan_run_quota(
        run_config, setup_plans=setup, prevalidated=prevalidated)

    replay_count = len(prevalidated.full_schedule["timestamps"])
    calibration_count = min(run_config.calibrate_n, replay_count)
    assert (len(setup), calibration_count, replay_count) == (2, 1, 1)
    assert run_config.max_output_tokens_cap == 720
    assert plan["may_start"] is True
    assert plan["status"] == "within_configured_harness_warning_budget"
    assert plan["planned_physical_attempts_worst_case"] == 12
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == 89_142
    assert plan["windows"]["input_tokens_per_minute"][
        "ratio_to_configured_limit"] == pytest.approx(0.44571)
    assert plan["windows"]["output_tokens_per_minute"][
        "planned_peak"] == 4_320
    assert plan["windows"]["output_tokens_per_minute"][
        "ratio_to_configured_limit"] == pytest.approx(0.216)


def _fake_git(source: Path, *, dirty: bool):
    commit = "a" * 40

    def run(*args: str, text: bool = True):
        if args == ("rev-parse", "HEAD"):
            value = commit + "\n"
        elif args == ("status", "--porcelain=v1", "--untracked-files=all"):
            value = " M changed\n" if dirty else ""
        elif args == ("show", "-s", "--format=%ct", commit):
            value = "1704067200\n"
        elif len(args) == 2 and args[0] == "show" \
                and args[1] == f"{commit}:guide.html":
            return source.read_bytes() if not text else source.read_text()
        else:
            raise AssertionError(args)
        return value if text else value.encode("utf-8")

    return run


def _minimal_five_page_guide() -> str:
    sections = []
    page_count = len(build_customer_pdf.SEMANTIC_PAGE_REQUIREMENTS)
    for index, requirements in enumerate(
            build_customer_pdf.SEMANTIC_PAGE_REQUIREMENTS, 1):
        content = "".join(f"<p>{value}</p>" for value in requirements)
        stamp = (
            f"<p>{build_customer_pdf.STAMP}</p>"
            if index == page_count else "")
        sections.append(
            f'<section class="page">{content}{stamp}'
            f"<footer>{index:02d} / {page_count:02d}</footer></section>")
    return """<!doctype html>
<html><head><meta charset="utf-8">
<title>Benchmark your own endpoint</title>
<style>
@page { size: Letter; margin: 0; }
html, body { margin: 0; }
.page {
  position: relative;
  box-sizing: border-box;
  height: 11in;
  padding: .5in .5in 1.1in;
}
.page:not(:last-child) { break-after: page; }
footer {
  position: absolute;
  left: .5in;
  right: .5in;
  bottom: .21in;
}
</style></head><body>""" + "".join(sections) + "</body></html>"


def _semantic_text(source: Path, commit: str) -> str:
    pages = []
    page_count = len(build_customer_pdf.SEMANTIC_PAGE_REQUIREMENTS)
    for index, requirements in enumerate(
            build_customer_pdf.SEMANTIC_PAGE_REQUIREMENTS, 1):
        visible = [*requirements, f"{index:02d} / {page_count:02d}"]
        if index == page_count:
            visible.extend((
                f"Source commit {commit}",
                "canonical HTML SHA-256 "
                f"{hashlib.sha256(source.read_bytes()).hexdigest()}",
            ))
        pages.append("\n".join(visible))
    return "\f".join(pages) + "\f"


def _bbox_text(*, page_count: int = 5,
               main_bottom: float = 700.0) -> str:
    pages = []
    for index in range(1, page_count + 1):
        pages.append(
            '<page width="612" height="792"><flow><block>'
            f'<line xMin="36" yMin="50" xMax="180" '
            f'yMax="{main_bottom}"><word xMin="36" yMin="50" '
            f'xMax="180" yMax="{main_bottom}">body</word></line>'
            '<line xMin="500" yMin="760" xMax="570" yMax="775">'
            f'<word xMin="500" yMin="760" xMax="570" yMax="775">'
            f'{index:02d} / {page_count:02d}</word></line>'
            '</block></flow></page>')
    return ('<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>'
            + "".join(pages) + '</doc></body></html>')


def _fake_pdf_tools(
        source: Path, *, reported_pages: int = 5,
        extracted: str | None = None,
        bbox_main_bottom: float = 700.0):
    commit = "a" * 40

    def run(command, **_kwargs):
        if command == ["/playwright", "--version"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="Version 1.2.3\n", stderr="")
        if command[:2] == ["/playwright", "pdf"]:
            Path(command[3]).write_bytes(b"%PDF-1.4\nfake fixture\n")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["/pdfinfo", "-v"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr="pdfinfo version 25.0\n")
        if command == ["/pdftotext", "-v"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr="pdftotext version 25.0\n")
        if command[0] == "/pdfinfo":
            info = (
                "Title: Benchmark your own endpoint\n"
                "Creator: Chromium\n"
                "Producer: Skia/PDF\n"
                f"Pages: {reported_pages}\n"
                "Encrypted: no\n"
                "JavaScript: no\n"
                "Page size: 612 x 792 pts (letter)\n")
            return subprocess.CompletedProcess(
                command, 0, stdout=info, stderr="")
        if command[0] == "/pdftotext":
            if "-bbox-layout" in command:
                return subprocess.CompletedProcess(
                    command, 0,
                    stdout=_bbox_text(
                        page_count=reported_pages,
                        main_bottom=bbox_main_bottom),
                    stderr="")
            visible = (
                _semantic_text(source, commit)
                if extracted is None else extracted)
            return subprocess.CompletedProcess(
                command, 0, stdout=visible, stderr="")
        raise AssertionError(command)

    return run


def test_customer_pdf_build_stamps_source_and_writes_verifiable_sidecar(
        tmp_path, monkeypatch):
    source = tmp_path / "guide.html"
    source.write_text(_minimal_five_page_guide(), encoding="utf-8")
    output = tmp_path / "guide.pdf"
    monkeypatch.setattr(build_customer_pdf, "ROOT", tmp_path)
    monkeypatch.setattr(build_customer_pdf, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(
        build_customer_pdf, "_git", _fake_git(source, dirty=False))
    tools = {
        "playwright": "/playwright",
        "pdfinfo": "/pdfinfo",
        "pdftotext": "/pdftotext",
    }
    monkeypatch.setattr(
        build_customer_pdf.shutil, "which", lambda name: tools.get(name))
    monkeypatch.setattr(
        build_customer_pdf.subprocess, "run", _fake_pdf_tools(source))
    result = build_customer_pdf.build(source, output)
    metadata = build_customer_pdf.check(source, output)

    assert output.read_bytes().startswith(b"%PDF-")
    assert result["source_git_dirty"] is False
    assert result["source_html_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    assert result["metadata_schema_version"] == 2
    assert result["pdf_page_count"] == 5
    assert result["pdfinfo_version"] == "pdfinfo version 25.0"
    assert result["pdftotext_version"] == "pdftotext version 25.0"
    assert result["geometry_qa_version"] == 1
    assert result["minimum_footer_clearance_points"] == 60.0
    assert metadata["pdf_sha256"] == hashlib.sha256(
        output.read_bytes()).hexdigest()


def test_customer_pdf_geometry_gate_rejects_footer_collision():
    with pytest.raises(RuntimeError, match="body/footer clearance"):
        build_customer_pdf._inspect_bbox_geometry(
            _bbox_text(main_bottom=758.0), 5)


def test_customer_pdf_build_rejects_a_one_page_fake_before_publish(
        tmp_path, monkeypatch):
    source = tmp_path / "guide.html"
    source.write_text(_minimal_five_page_guide(), encoding="utf-8")
    output = tmp_path / "guide.pdf"
    output.write_bytes(b"existing PDF remains recoverable")
    monkeypatch.setattr(build_customer_pdf, "ROOT", tmp_path)
    monkeypatch.setattr(build_customer_pdf, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(
        build_customer_pdf, "_git", _fake_git(source, dirty=False))
    tools = {
        "playwright": "/playwright",
        "pdfinfo": "/pdfinfo",
        "pdftotext": "/pdftotext",
    }
    monkeypatch.setattr(
        build_customer_pdf.shutil, "which", lambda name: tools.get(name))
    monkeypatch.setattr(
        build_customer_pdf.subprocess, "run",
        _fake_pdf_tools(source, reported_pages=1))

    with pytest.raises(RuntimeError, match="has 1 pages; expected exactly 5"):
        build_customer_pdf.build(source, output)
    assert output.read_bytes() == b"existing PDF remains recoverable"
    assert not build_customer_pdf._metadata_path(output).exists()


@pytest.mark.parametrize(("forbidden", "message"), (
    ("03 / 03", "stale three-page footer marker"),
    ("UNSTAMPED SOURCE", "UNSTAMPED source marker"),
))
def test_customer_pdf_build_rejects_forbidden_visible_release_text(
        tmp_path, monkeypatch, forbidden, message):
    source = tmp_path / "guide.html"
    source.write_text(_minimal_five_page_guide(), encoding="utf-8")
    output = tmp_path / "guide.pdf"
    text = _semantic_text(source, "a" * 40).replace(
        "03 / 05", f"03 / 05 {forbidden}")
    monkeypatch.setattr(build_customer_pdf, "ROOT", tmp_path)
    monkeypatch.setattr(build_customer_pdf, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(
        build_customer_pdf, "_git", _fake_git(source, dirty=False))
    tools = {
        "playwright": "/playwright",
        "pdfinfo": "/pdfinfo",
        "pdftotext": "/pdftotext",
    }
    monkeypatch.setattr(
        build_customer_pdf.shutil, "which", lambda name: tools.get(name))
    monkeypatch.setattr(
        build_customer_pdf.subprocess, "run",
        _fake_pdf_tools(source, extracted=text))

    with pytest.raises(RuntimeError, match=message):
        build_customer_pdf.build(source, output)
    assert not output.exists()


@pytest.mark.skipif(
    any(shutil.which(name) is None
        for name in ("playwright", "pdfinfo", "pdftotext")),
    reason="real PDF QA tools are not installed",
)
def test_customer_pdf_real_renderer_produces_five_semantic_pages(
        tmp_path, monkeypatch):
    source = tmp_path / "guide.html"
    source.write_text(_minimal_five_page_guide(), encoding="utf-8")
    output = tmp_path / "guide.pdf"
    monkeypatch.setattr(build_customer_pdf, "ROOT", tmp_path)
    monkeypatch.setattr(build_customer_pdf, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(
        build_customer_pdf, "_git", _fake_git(source, dirty=False))

    result = build_customer_pdf.build(source, output)
    checked = build_customer_pdf.check(source, output)

    assert result["pdf_page_count"] == 5
    assert checked["semantic_requirements_sha256"] == \
        build_customer_pdf.SEMANTIC_REQUIREMENTS_SHA256


def test_customer_pdf_distribution_build_refuses_a_dirty_tree(
        tmp_path, monkeypatch):
    source = tmp_path / "guide.html"
    source.write_text(build_customer_pdf.STAMP, encoding="utf-8")
    output = tmp_path / "guide.pdf"
    monkeypatch.setattr(build_customer_pdf, "ROOT", tmp_path)
    monkeypatch.setattr(build_customer_pdf, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(
        build_customer_pdf, "_git", _fake_git(source, dirty=True))

    with pytest.raises(RuntimeError, match="dirty tree"):
        build_customer_pdf.build(source, output)
    assert not output.exists()
