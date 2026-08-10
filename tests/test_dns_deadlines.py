"""Every production network path must bound DNS before a socket exists."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from traffic_replay.client import EndpointClient, EndpointConfig


def _blocking_resolver(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    resolver_threads = []
    socket_calls = []

    def getaddrinfo(host, port, family=0, socktype=0, proto=0, flags=0):
        resolver_threads.append(threading.current_thread())
        entered.set()
        try:
            release.wait(timeout=2.0)
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                     "", ("192.0.2.10", port))]
        finally:
            finished.set()

    def forbidden_socket(*args, **kwargs):
        socket_calls.append((args, kwargs))
        raise AssertionError("a socket was opened after blocked DNS")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    return entered, release, finished, resolver_threads, socket_calls


def _release_resolver(release, finished, resolver_threads):
    release.set()
    assert finished.wait(timeout=1.0)
    for thread in resolver_threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()


def test_inference_absolute_deadline_bounds_blocked_dns_without_late_post(
        monkeypatch):
    entered, release, finished, resolver_threads, socket_calls = \
        _blocking_resolver(monkeypatch)
    client = EndpointClient(EndpointConfig(
        base_url="http://inference-dns-deadline.invalid", path="/invoke",
        connect_timeout_s=0.5, read_timeout_s=0.5,
        total_timeout_s=0.04, include_usage=False), None)

    started = time.monotonic()
    try:
        result = client.send(
            [{"role": "user", "content": "hello"}], 8, "dns-deadline",
            0.0, 0.0, (1, 1, None, -1), 5)
        elapsed = time.monotonic() - started

        assert entered.is_set()
        assert 0.025 <= elapsed < 0.25
        assert result.error == (
            "request exceeded total timeout (total_timeout_s=0.04)")
        assert result.connection_attempts == 1
        assert result.request_attempts == 0
        assert result.first_attempt_unix is not None
        assert result.first_send_unix is None
        assert socket_calls == []
        assert len(resolver_threads) == 1
        assert resolver_threads[0].name == "traffic-replay-dns-resolver"
        assert resolver_threads[0].daemon is True
    finally:
        _release_resolver(release, finished, resolver_threads)
    # A late DNS result is discarded by the resolver-only daemon. It cannot
    # resume connection or request code after send() has returned.
    assert socket_calls == []


def test_operator_cancellation_interrupts_a_blocked_dns_wait(monkeypatch):
    entered, release, finished, resolver_threads, socket_calls = \
        _blocking_resolver(monkeypatch)
    cancelled = threading.Event()
    client = EndpointClient(EndpointConfig(
        base_url="http://cancelled-dns.invalid", path="/invoke",
        connect_timeout_s=1.0, total_timeout_s=1.0,
        include_usage=False), None)
    result_box = []

    worker = threading.Thread(target=lambda: result_box.append(client.send(
        [{"role": "user", "content": "hello"}], 8, "dns-cancel",
        0.0, 0.0, (1, 1, None, -1), 5,
        cancellation_event=cancelled)))
    worker.start()
    try:
        assert entered.wait(timeout=1.0)
        cancelled.set()
        worker.join(timeout=0.25)
        assert not worker.is_alive()
        assert result_box[0].request_attempts == 0
        assert "cancelled before HTTP POST" in (result_box[0].error or "")
        assert socket_calls == []
    finally:
        _release_resolver(release, finished, resolver_threads)
        worker.join(timeout=1.0)
    assert socket_calls == []


def test_network_path_probe_bounds_blocked_dns_without_connecting(monkeypatch):
    from traffic_replay.netpath import measure_network_path

    entered, release, finished, resolver_threads, socket_calls = \
        _blocking_resolver(monkeypatch)
    started = time.monotonic()
    try:
        result = measure_network_path(
            "https://netpath-dns-deadline.invalid", samples=3,
            timeout=0.04)
        elapsed = time.monotonic() - started
        assert entered.is_set()
        assert result is None
        assert 0.025 <= elapsed < 0.25
        assert socket_calls == []
        assert resolver_threads[0].daemon is True
    finally:
        _release_resolver(release, finished, resolver_threads)
    assert socket_calls == []


def test_endpoint_metadata_bounds_blocked_dns_without_late_get_or_socket(
        monkeypatch):
    from traffic_replay.endpoint_meta import fetch_endpoint_metadata

    entered, release, finished, resolver_threads, socket_calls = \
        _blocking_resolver(monkeypatch)
    started = time.monotonic()
    try:
        result = fetch_endpoint_metadata(
            "https://metadata-dns-deadline.invalid",
            "/serving-endpoints/example/invocations", "test-token",
            timeout=0.04)
        elapsed = time.monotonic() - started
        assert entered.is_set()
        assert result is None
        assert 0.025 <= elapsed < 0.25
        assert socket_calls == []
        assert resolver_threads[0].daemon is True
    finally:
        _release_resolver(release, finished, resolver_threads)
    assert socket_calls == []


def test_workspace_oauth_m2m_bounds_blocked_dns_without_late_post(
        monkeypatch):
    from traffic_replay import runner

    entered, release, finished, resolver_threads, socket_calls = \
        _blocking_resolver(monkeypatch)
    monkeypatch.setattr(runner, "_AUTH_M2M_TIMEOUT_S", 0.04)
    started = time.monotonic()
    try:
        with pytest.raises(runner.AuthProfileError, match="timed out after"):
            runner._mint_workspace_m2m_token(
                ("https", "oauth-dns-deadline.invalid", 443),
                "client-id", "client-secret", profile_name="blocked")
        elapsed = time.monotonic() - started
        assert entered.is_set()
        assert 0.025 <= elapsed < 0.25
        assert socket_calls == []
        assert resolver_threads[0].daemon is True
    finally:
        _release_resolver(release, finished, resolver_threads)
    assert socket_calls == []


def test_same_target_concurrent_dns_waiters_share_one_daemon_lookup(
        monkeypatch):
    from traffic_replay.network import bounded_getaddrinfo

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocked(host, port, family=0, socktype=0, proto=0, flags=0):
        calls.append(threading.current_thread())
        entered.set()
        assert release.wait(timeout=1.0)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 "", ("192.0.2.20", port))]

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    results = []
    workers = [threading.Thread(target=lambda: results.append(
        bounded_getaddrinfo(
            "singleflight-dns.invalid", 443, timeout=0.5)))
        for _ in range(2)]
    for worker in workers:
        worker.start()
    assert entered.wait(timeout=1.0)
    time.sleep(0.02)
    assert len(calls) == 1
    assert calls[0].daemon is True
    release.set()
    for worker in workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    calls[0].join(timeout=1.0)
    assert len(results) == 2
    assert results[0] == results[1]


def test_dns_active_unique_lookup_cap_fails_closed_without_new_thread(
        monkeypatch):
    from traffic_replay import network

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocked(host, port, family=0, socktype=0, proto=0, flags=0):
        calls.append(host)
        entered.set()
        assert release.wait(timeout=1.0)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 "", ("192.0.2.30", port))]

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(network, "_MAX_ACTIVE_DNS_LOOKUPS", 1)
    first_result = []
    first = threading.Thread(target=lambda: first_result.append(
        network.bounded_getaddrinfo(
            "first-capped-dns.invalid", 443, timeout=0.5)))
    first.start()
    try:
        assert entered.wait(timeout=1.0)
        with pytest.raises(socket.gaierror) as exc:
            network.bounded_getaddrinfo(
                "second-capped-dns.invalid", 443, timeout=0.05)
        assert exc.value.errno == socket.EAI_AGAIN
        assert calls == ["first-capped-dns.invalid"]
    finally:
        release.set()
        first.join(timeout=1.0)
    assert not first.is_alive()
    assert len(first_result) == 1


def test_endpoint_metadata_absolute_deadline_stops_a_dribbling_body(
        monkeypatch):
    from traffic_replay.endpoint_meta import fetch_endpoint_metadata

    released = threading.Event()
    watchdog_threads = []
    requests = []

    class Sock:
        def shutdown(self, _how):
            watchdog_threads.append(threading.current_thread())
            released.set()

    class Response:
        status = 200

        def getheader(self, _name):
            return None

        def read(self, _limit):
            assert released.wait(timeout=1.0)
            return b'{"name":"late","config":{"served_entities":[]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.sock = Sock()

        def request(self, method, path, **_kwargs):
            requests.append((method, path))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    started = time.monotonic()
    result = fetch_endpoint_metadata(
        "https://metadata-dribble.invalid",
        "/serving-endpoints/example/invocations", "test-token",
        timeout=0.04)
    elapsed = time.monotonic() - started

    assert result is None
    assert requests == [("GET", "/api/2.0/serving-endpoints/example")]
    assert 0.025 <= elapsed < 0.25
    assert watchdog_threads
    assert all(thread.daemon for thread in watchdog_threads)
    assert all(thread.name == "traffic-replay-http-deadline-watchdog"
               for thread in watchdog_threads)


def test_workspace_oauth_absolute_deadline_stops_a_dribbling_body(
        monkeypatch):
    from traffic_replay import runner

    released = threading.Event()
    watchdog_threads = []
    requests = []

    class Sock:
        def shutdown(self, _how):
            watchdog_threads.append(threading.current_thread())
            released.set()

    class Response:
        status = 200

        def getheader(self, name):
            return ("application/json" if name == "Content-Type" else None)

        def read(self, _limit):
            assert released.wait(timeout=1.0)
            return (b'{"access_token":"late-token","token_type":"Bearer",'
                    b'"scope":"all-apis"}')

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.sock = Sock()

        def request(self, method, path, **_kwargs):
            requests.append((method, path))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    monkeypatch.setattr(runner, "_AUTH_M2M_TIMEOUT_S", 0.04)
    started = time.monotonic()
    with pytest.raises(runner.AuthProfileError, match="timed out after"):
        runner._mint_workspace_m2m_token(
            ("https", "oauth-dribble.invalid", 443),
            "client-id", "client-secret", profile_name="blocked")
    elapsed = time.monotonic() - started

    assert requests == [("POST", "/oidc/v1/token")]
    assert 0.025 <= elapsed < 0.25
    assert watchdog_threads
    assert all(thread.daemon for thread in watchdog_threads)
    assert all(thread.name == "traffic-replay-http-deadline-watchdog"
               for thread in watchdog_threads)
