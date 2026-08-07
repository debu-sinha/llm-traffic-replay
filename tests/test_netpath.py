"""Where the client sits relative to the endpoint.

Every latency figure contains at least one round trip: the request goes out
and the first token comes back. A run generated from the wrong region folds
that into TTFT and into any SLA judgment made from it. That happened for
real: a load test reporting TTFT p50 842 ms against a 500 ms target was run
from the US east coast against an endpoint in us-west-2, and 82 ms of the
number was the width of the country. Nothing in the report said so.
"""
from __future__ import annotations

import http.server
import socket
import threading

from traffic_replay.metrics import render_html, render_markdown, summarize
from traffic_replay.netpath import measure_network_path


def _rows(n, ttft, base=1_700_000_000.0):
    return [{"ok": True, "phase": "replay", "ttft_ms": ttft,
             "e2e_ms": ttft * 2, "prompt_tokens": 100,
             "completion_tokens": 10, "stream_complete": True,
             "visible_content_seen": True, "truncated": False,
             "parse_errors": 0,
             "t_send_unix": base + i * 0.3,
             "first_send_unix": base + i * 0.3} for i in range(n)]


def _meta(rtt):
    return {"network_path": {"endpoint_host": "ws.example.com",
                             "endpoint_ips": ["44.234.192.45"],
                             "tcp_connect_min_ms": rtt, "samples": 5}}


def test_it_measures_a_real_round_trip_to_a_local_server():
    """A loopback server is the only endpoint whose true distance we know:
    effectively zero."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = measure_network_path(f"http://127.0.0.1:{port}", samples=3)
    finally:
        srv.shutdown()
    assert r is not None
    assert r["endpoint_ips"] == ["127.0.0.1"]
    assert r["samples"] == 3
    assert r["tcp_connect_min_ms"] < 50, r
    assert "rtt_ms" not in r
    assert "client_hostname" not in r
    assert "client_egress_ip" not in r


def test_an_unresolvable_host_does_not_break_the_run():
    """A benchmark must never fail because it could not describe its own
    network position."""
    assert measure_network_path("https://no-such-host.invalid.") is None
    assert measure_network_path("not a url at all") is None


def test_invalid_probe_controls_fail_closed_without_connecting():
    assert measure_network_path("https://example.invalid", samples=0) is None
    assert measure_network_path("https://example.invalid", samples=True) is None
    assert measure_network_path("https://example.invalid", timeout=0) is None


def test_ipv6_is_supported_and_even_sample_median_is_arithmetic(monkeypatch):
    connected = []

    class FakeSocket:
        def settimeout(self, value):
            assert value == 5.0

        def connect(self, address):
            connected.append(address)

        def close(self):
            pass

    monkeypatch.setattr("traffic_replay.netpath.socket.getaddrinfo",
                        lambda *args, **kwargs: [
                            (socket.AF_INET6, socket.SOCK_STREAM,
                             socket.IPPROTO_TCP, "", ("::1", 443, 0, 0))])
    monkeypatch.setattr("traffic_replay.netpath.socket.socket",
                        lambda *args: FakeSocket())
    times = iter((0.0, 0.010, 1.0, 1.030))
    monkeypatch.setattr("traffic_replay.netpath.time.perf_counter",
                        lambda: next(times))
    result = measure_network_path("https://[::1]", samples=2)
    assert connected == [("::1", 443, 0, 0), ("::1", 443, 0, 0)]
    assert result["tcp_connect_min_ms"] == 10.0
    assert result["tcp_connect_median_ms"] == 20.0


def test_tcp_connect_floor_is_context_and_never_subtracted_from_ttft():
    s = summarize(_rows(300, 842.0), run_meta=_meta(82.0))
    np = s["network_path"]
    assert "ttft_p50_less_rtt" not in np
    assert 0.09 < np["tcp_connect_floor_to_ttft_p50_ratio"] < 0.10
    assert "must not be subtracted" in np["interpretation"]


def test_a_network_floor_is_accurately_labeled_in_both_reports():
    s = summarize(_rows(300, 842.0), run_meta=_meta(82.0),
                  acceptance={"ttft_ms": {"p50": 5000}})
    assert "warning" not in s["network_path"]
    md = render_markdown(s, "x")
    assert "network-path floor: 82 ms minimum TCP connect" in md
    assert "do not subtract it from TTFT" in md
    assert "endpoint time" not in md
    html = render_html(s, "x")
    assert "Network-path floor" in html
    assert "minimum TCP connect to ws.example.com" in html
    assert "endpoint time" not in html


def test_a_nearby_client_says_the_distance_without_crying_about_it():
    """In-region is the normal case and must not raise a caution."""
    s = summarize(_rows(300, 842.0), run_meta=_meta(2.0),
                  acceptance={"ttft_ms": {"p50": 5000}})
    assert "warning" not in s["network_path"]
    md = render_markdown(s, "x")
    assert "CAUTION (network distance)" not in md
    assert "network-path floor: 2 ms minimum TCP connect" in md


def test_no_network_block_when_it_could_not_be_measured():
    s = summarize(_rows(300, 842.0))
    assert "network_path" not in s
    assert "CAUTION (network distance)" not in render_markdown(s, "x")
