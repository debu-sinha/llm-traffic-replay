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
    return " ".join(value.strip() for value in root.itertext()
                    if value.strip())


def test_architecture_diagram_has_the_complete_evidence_lifecycle():
    text = _svg_text("architecture.svg")
    for required in (
            "Setup evidence claimed first", "Billable preflight",
            "Runtime-admitted POST attempts",
            "reserve immediately before conn.request",
            "local denial sends no physical POST",
            "fresh HTTP/1.1 connection each attempt",
            "model/entity identity kept", "Post-drain endpoint snapshot",
            "normalized-subset change invalidates claim",
            "five decisions + evidence gates",
            ".traffic-replay-complete promoted last"):
        assert required in text
    assert "journal claimed; metadata-only setup rows appended" not in text


def test_load_model_separates_offered_load_from_runtime_admission():
    text = _svg_text("load-model.svg")
    for required in (
            "Open-loop dispatch", "fresh HTTP/1.1 POST admission",
            "one no-wait guard spans the command",
            "reserve QPS/QPH, TPM, exact bytes",
            "local denial sends no POST",
            "response identity", "runtime admission",
            "identity / stability / admission evidence gates"):
        assert required in text
    assert "quota or errors can shed work" not in text


def test_request_sequence_puts_admission_before_every_request_call():
    text = _svg_text("request-sequence.svg")
    admission = text.index("Atomic no-wait runtime admission")
    request_call = text.index("conn.request uploads POST body")
    assert admission < request_call
    for required in (
            "denial → terminal row, no conn.request",
            "Fresh HTTP/1.1", "new connection per physical attempt",
            "every repeated POST requires a new reservation",
            "reasoning/visible/refusal/tool/usage events",
            "response identity", "settle admission event",
            "first_content = reasoning/visible/refusal onset",
            "TTFB = first nonempty bounded body chunk"):
        assert required in text
    assert "network distance" not in text.lower()


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
            "immediately before every physical POST",
            "local no-wait guard reserves QPS/QPH",
            "fresh HTTP/1.1 connection per physical attempt",
            "response model/fingerprint",
            "first_content includes reasoning, visible, or refusal onset",
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
        assert "owner-confirmed" in body.casefold(), path
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
        assert '{"reasoning_effort":"none"}' in help_text
        assert '"chat_template_kwargs":{"enable_thinking":false}' in help_text
