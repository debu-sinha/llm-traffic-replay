"""The HTML report: self-contained, unit-labeled, color-coded, and safe.

Covers the parts a markdown report can't: an SLA verdict a reader can see at
a glance, units on every metric, and HTML-escaping of untrusted label text.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

from traffic_replay.metrics import render_html, write_outputs
from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run


def _summary(met_p95, label="run", n=250):
    """n defaults above the 100-request tail floor, because the green banner
    now requires a run big enough to support the numbers it prints."""
    return {
        "requests_total": n, "requests_ok": n, "requests_failed": 0,
        "error_rate": 0.0, "failures_by_error": {},
        "ttft_ms": {"p50": 100, "p90": 150, "p95": 180, "p99": 200, "n": n},
        "e2e_ms": {"p50": 300, "p90": 400, "p95": 450, "p99": 500, "n": n},
        "ttfb_ms": {"n": 0}, "interchunk_max_ms": {"n": 0},
        "throughput": {"input_tokens_per_min": 1000,
                       "output_tokens_per_min": 50},
        "achieved_cache_fraction": {"p50": 0.5, "p95": 0.7, "n": n,
                                    "reported_for_n": n,
                                    "source_fields": ["prompt_tokens_details.cached_tokens"]},
        "intended_cache_fraction": {"p50": 0.45, "p95": 0.72, "n": n},
        "arrivals": {"achieved_qps_overall": 2.0,
                     "dispatch_lag_ms": {"p95": 5}},
        "token_targeting": {"finish_reasons": {"stop": n}},
        "run": {"input_mode": "profile", "endpoint_path": "/e",
                "label": label,
                "request_params": {"temperature": 0.0,
                                   "max_output_tokens_cap": 40,
                                   "extra_body": {}}},
        "sla": {"ttft_definition": "first_content",
                "ttft_vs_target": [{"quantile": "p95", "target_ms": 150,
                                    "actual_ms": 180, "met": met_p95}],
                "ttfg_vs_target": [],
                "hard_timeout_breaches": 0,
                "success_rate": {"target": 0.99, "actual": 1.0, "met": True}},
    }


def test_html_is_self_contained_and_has_units():
    h = render_html(_summary(True), "My Run")
    assert h.startswith("<!doctype html>")
    # no external assets, safe to open or attach anywhere
    assert "http://" not in h and "https://" not in h
    assert "<link" not in h and "<script" not in h
    # units are spelled out for every metric family
    for unit in ("milliseconds", "(ms)", "hit fraction (0-1)",
                 "requests/second (QPS)", "tok/min", "(count)",
                 "fraction 0-1"):
        assert unit in h, f"missing unit label: {unit}"


def test_html_color_codes_pass_and_fail():
    passed = render_html(_summary(True), "ok run")
    assert "Meets every acceptance target" in passed
    assert "class='no'" not in passed

    missed = render_html(_summary(False), "bad run")
    assert "1 acceptance target missed" in missed
    assert "class='no'" in missed          # the missed row is flagged red
    assert "class='yes'" in missed          # success rate still passes


def test_html_escapes_untrusted_label():
    h = render_html(_summary(True, label="<script>alert(1)</script>"), "T")
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


def test_write_outputs_emits_html_end_to_end():
    d = tempfile.mkdtemp()
    truth = Path(d) / "t.jsonl"
    srv = serve(0, truth)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.3)
    try:
        rc = RunConfig(
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "NONE"},
            profile_path="configs/profile_validation_small.json",
            duration_s=5, qps_base=2.0, qps_burst=4.0, qps_min=1.0,
            qps_max=6.0, max_concurrency=4, calibrate_n=2,
            out_dir=os.path.join(d, "r"), title="e2e html",
            max_output_tokens_cap=16)
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()
    html_path = Path(out["out_dir"], "report.html")
    assert html_path.exists()
    body = html_path.read_text()
    assert "e2e html" in body and "Latency (milliseconds)" in body
    assert body.startswith("<!doctype html>")


def test_html_escapes_structured_payloads():
    s = _summary(True)
    s["run"]["request_params"]["extra_body"] = {
        "x": "<img src=x onerror=alert(1)>"}
    s["token_targeting"]["finish_reasons"] = {"</script><b>evil</b>": 1}
    s["achieved_cache_fraction"]["source_fields"] = ["<i>field</i>"]
    h = render_html(s, "T")
    assert "<img src=x onerror=alert(1)>" not in h
    assert "</script><b>evil</b>" not in h
    assert "<i>field</i>" not in h
