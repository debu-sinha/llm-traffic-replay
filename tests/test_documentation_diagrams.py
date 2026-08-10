from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from traffic_replay.cli import main


ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs/diagrams"


def _svg_text(name: str) -> str:
    path = DIAGRAMS / name
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    assert root.attrib.get("role") == "img"
    labelled_by = root.attrib.get("aria-labelledby", "").split()
    assert labelled_by == ["title", "desc"]
    labelled = {
        child.attrib.get("id"): " ".join(
            value.strip() for value in child.itertext() if value.strip())
        for child in root
        if child.attrib.get("id") in labelled_by
    }
    assert set(labelled) == set(labelled_by)
    assert all(labelled.values())
    return " ".join(value.strip() for value in root.itertext()
                    if value.strip())


def test_architecture_diagram_has_the_complete_evidence_lifecycle():
    text = _svg_text("architecture.svg")
    for required in (
            "Setup evidence claimed first", "Billable preflight",
            "Versioned endpoint adapter", "exact request serialization",
            "declared media + framing", "canonical event fold",
            "outcome + normalized usage", "Fresh HTTP/1.1",
            "transport boundary", "after connect; before POST",
            "denial sends no POST", "Post-drain endpoint snapshot",
            "normalized-subset change invalidates claim",
            "five decisions + evidence gates",
            ".traffic-replay-complete promoted last"):
        assert required in text
    assert "journal claimed; metadata-only setup rows appended" not in text


def test_load_model_separates_offered_load_from_runtime_admission():
    text = _svg_text("load-model.svg")
    for required in (
            "Versioned endpoint adapter boundary",
            "serialize the exact request bytes",
            "declare request media, accepted response media",
            "fold provider events into canonical content-free state",
            "normalize usage and classify terminal/output outcome",
            "Open-loop dispatch", "Fresh HTTP/1.1",
            "transport boundary", "connect before admission",
            "Runtime admission", "immediately before POST",
            "denial sends no POST",
            "response identity", "runtime admission",
            "runtime-admission evidence gates"):
        assert required in text
    assert "quota or errors can shed work" not in text


def test_request_sequence_puts_admission_before_every_request_call():
    text = _svg_text("request-sequence.svg")
    admission = text.index("Atomic no-wait runtime admission")
    request_call = text.index("conn.request uploads POST body")
    assert admission < request_call
    for required in (
            "denial → terminal row, no conn.request",
            "Versioned endpoint adapter",
            "exact request bytes + declared media/framing contract",
            "Fresh HTTP/1.1 transport",
            "new connection per physical attempt",
            "every repeated POST needs a new reservation",
            "folds provider events → canonical content-free state",
            "normalize usage", "classify outcome",
            "response identity", "settle admission event",
            "first_content = reasoning/visible/refusal onset",
            "TTFB = first bounded body chunk",
            "TTSE = first adapter-framed event",
            "neither is a token"):
        assert required in text
    assert "network distance" not in text.lower()


def test_diagrams_keep_failures_in_reliability_and_occupancy_only():
    for name in (
            "architecture.svg", "load-model.svg", "request-sequence.svg"):
        text = _svg_text(name).casefold()
        assert "reliability" in text, name
        assert "occupancy" in text, name
        assert "latency needs acceptable outcomes" in text, name
        assert "tokens need trustworthy usage" in text, name
        assert "failed streams excluded from metrics" not in text, name


def test_excalidraw_source_matches_the_current_runtime_contract():
    payload = json.loads(
        (DIAGRAMS / "architecture.excalidraw").read_text(encoding="utf-8"))
    assert payload["type"] == "excalidraw"
    text_elements = [element for element in payload["elements"]
                     if element.get("type") == "text"]
    assert text_elements
    assert all(element["text"] == element["originalText"]
               for element in text_elements)
    text = "\n".join(element["text"] for element in text_elements)
    for required in (
            "claim sibling setup artifact before default two-request preflight",
            "versioned endpoint adapter: exact request serialization",
            "declared media/framing",
            "canonical event fold + normalized usage/outcome",
            "separate built-in transport opens a fresh HTTP/1.1 connection",
            "immediately before every physical POST",
            "command-local guard reserves QPS/QPH",
            "response model/fingerprint",
            "first_content includes reasoning, visible, or refusal onset",
            "failed/partial attempts remain in reliability and occupancy",
            "latency needs acceptable outcomes; tokens need trustworthy usage",
            "capture the normalized endpoint-metadata subset again",
            "exactly five decisions",
            "identity/stability/runtime-admission are evidence gates"):
        assert required in text


def test_markdown_claims_share_the_exact_measurement_contract():
    documents = [
        ROOT / "README.md",
        ROOT / "docs/ARCHITECTURE.md",
        ROOT / "docs/PRODUCTION_TESTING.md",
        ROOT / "docs/RUN_YOUR_OWN_BENCHMARK.md",
    ]
    for path in documents:
        body = path.read_text(encoding="utf-8")
        flat = " ".join(body.split())
        assert "visible, reasoning, or refusal" in flat, path
        assert "first nonempty bounded response-body chunk" in flat, path
        assert "sizing" in flat, path
        assert any(phrase in flat for phrase in (
            "explicit exception",
            "cannot materialize its schedule",
            "only schedule that cannot yet exist",
        )), path
        assert "not an attempt-by-attempt" in flat, path
        assert "exactly the five" in flat, path
        assert "selected subset" in flat, path
        assert ("non-refusal" in flat or "no refusal marker" in flat), path
        assert "first iterated response-body/SSE line" not in flat, path

    readme = documents[0].read_text(encoding="utf-8")
    assert "exactly 2,703 independent attempts" in readme
    assert "github.com/debu-sinha/llm-traffic-replay/blob/main" not in readme


def _command_help(command: str) -> str:
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            main([command, "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse help always exits
        raise AssertionError("argparse help did not exit")
    return stdout.getvalue()


def test_cli_help_discloses_event_and_fraction_semantics():
    for command in ("benchmark", "sweep", "quickstart"):
        help_text = " ".join(_command_help(command).split())
        assert "FRACTION_IN_(0,1)" in help_text
        assert "first_content is the first visible, reasoning, or refusal" \
            in help_text
    assert "json prints only the validation comparison object" in \
        " ".join(_command_help("validate").split())


def test_glm_reasoning_controls_keep_managed_and_sglang_contracts_separate():
    documents = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "docs/PRODUCTION_TESTING.md",
        ROOT / "docs/RUN_YOUR_OWN_BENCHMARK.md",
        ROOT / "docs/customer/benchmark-your-own-endpoint.html",
    ]
    for path in documents:
        body = path.read_text(encoding="utf-8")
        assert "serving-engineering-confirmed" in body.casefold(), path
        assert '{"reasoning_effort":"none"}' in body, path
        assert '"chat_template_kwargs":{"enable_thinking":false}' in body, path
        assert "maximum reasoning" in body, path

    for path in documents[:1] + documents[2:]:
        body = path.read_text(encoding="utf-8")
        assert "system.ai.glm-5-2" in body, path
        assert "protocol-diagnostic" in body, path
        assert "/serving-endpoints/.../invocations" in body, path
        assert "https://docs.databricks.com/aws/en/ai-gateway/" in body, path

    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    for required in (
            "production-qualified Unity AI Gateway adapter",
            "fully qualified model-service name", "destination identity",
            "routing", "fallback", "intersection of Gateway",
            "HTTP 429"):
        assert required in todo

    for command in ("benchmark", "sweep"):
        help_text = " ".join(_command_help(command).split())
        assert "--endpoint-adapter" in help_text
        assert "versioned request/response wire contract" in help_text
        assert "model behavior is never inferred from its name" in help_text
