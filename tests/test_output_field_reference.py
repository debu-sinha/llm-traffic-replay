from __future__ import annotations

import dataclasses
from pathlib import Path

from traffic_replay.client import RequestResult
from traffic_replay.metrics import render_html, render_markdown, summarize


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "docs" / "OUTPUT_FIELD_REFERENCE.md"


def _row() -> dict:
    return {
        "phase": "replay", "request_id": "r1", "scheduled_s": 0.0,
        "first_attempt_unix": 1.0, "first_send_unix": 1.0,
        "t_send_unix": 1.0, "finished_unix": 1.2,
        "request_attempts": 1, "connection_attempts": 1,
        "retries": 0, "retry_reasons": [], "status": 200, "ok": True,
        "stream_complete": True, "finish_reason": "stop",
        "visible_content_seen": True, "ttfb_ms": 10.0, "ttse_ms": 11.0,
        "ttft_ms": 12.0, "ttfv_ms": 12.0, "e2e_ms": 200.0,
        "prompt_tokens": 10, "completion_tokens": 2,
        "max_tokens_requested": 8, "physical_request_body_sha256s": ["a" * 64],
    }


def test_report_embeds_crystal_clear_calibration_and_metric_glossary():
    summary = summarize([_row()])
    for rendered in (
            render_markdown(summary, "field glossary"),
            render_html(summary, "field glossary")):
        assert "Calibration request" in rendered
        assert "real, paid, unloaded request" in rendered
        assert "not a warm-up exclusion" in rendered
        assert "Actual count is min(calibrate_n, replay rows)" in rendered
        assert "Logical request vs physical attempt" in rendered
        assert "Missing, unknown" in rendered
        assert "TTFB" in rendered and "TTSE" in rendered
        assert "TTFV" in rendered and "TTFG / E2E" in rendered


def test_reference_names_every_request_result_field():
    reference = REFERENCE.read_text()
    missing = [
        field.name for field in dataclasses.fields(RequestResult)
        if f"`{field.name}`" not in reference
    ]
    assert missing == []


def test_reference_names_every_emitted_summary_top_level_field():
    reference = REFERENCE.read_text()
    summary = summarize([_row()])
    missing = [key for key in summary if f"`{key}`" not in reference]
    assert missing == []
