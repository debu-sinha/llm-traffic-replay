"""Operator cancellation must stop queued work and every later physical POST."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from traffic_replay.client import EndpointClient, EndpointConfig, RequestResult
from traffic_replay.runner import RunConfig, run


def _client(*, retries: int = 0) -> EndpointClient:
    return EndpointClient(EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/serving-endpoints/mock/invocations",
        max_retries=retries,
        include_usage=False), None)


def _send(client: EndpointClient, event: threading.Event):
    return client.send(
        [{"role": "user", "content": "test"}], 1, "request", 0.0, 0.0,
        (1, 1, 0.0, -1), 4, cancellation_event=event)


def test_preset_cancellation_never_connects_or_posts(monkeypatch):
    client = _client()
    event = threading.Event()
    event.set()
    monkeypatch.setattr(
        client, "_connect",
        lambda: (_ for _ in ()).throw(AssertionError("connection attempted")))

    result = _send(client, event)

    assert result.ok is False
    assert result.connection_attempts == 0
    assert result.request_attempts == 0
    assert result.first_send_unix is None
    assert "cancelled before HTTP POST" in (result.error or "")


def test_cancellation_after_connect_is_rechecked_immediately_before_post(
        monkeypatch):
    event = threading.Event()
    posts = []

    class Connection:
        sock = None
        timeout = None

        def connect(self):
            event.set()

        def request(self, *_args, **_kwargs):
            posts.append("POST")

        def close(self):
            pass

    client = _client()
    monkeypatch.setattr(client, "_connect", Connection)

    result = _send(client, event)

    assert posts == []
    assert result.connection_attempts == 1
    assert result.request_attempts == 0
    assert "cancelled before HTTP POST" in (result.error or "")


def test_cancellation_during_quota_post_mark_never_starts_request(monkeypatch):
    """Guard-lock delay must not reopen the final cancellation race."""
    from traffic_replay.quota_planner import RuntimeQuotaGuard

    event = threading.Event()
    marked = threading.Event()
    release_mark = threading.Event()
    request_calls = []

    class PausingGuard(RuntimeQuotaGuard):
        def mark_post_may_have_started(self, handle):
            evidence = super().mark_post_may_have_started(handle)
            marked.set()
            assert release_mark.wait(timeout=2.0)
            return evidence

    class Sock:
        def settimeout(self, _timeout):
            pass

        def shutdown(self, _how):
            pass

    class Connection:
        def __init__(self):
            self.sock = Sock()

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            request_calls.append(event.is_set())

        def close(self):
            pass

    guard = PausingGuard({
        "queries_per_hour": 100,
        "warning_utilization": 1.0,
    })
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False),
        None, runtime_quota_guard=guard)
    monkeypatch.setattr(client, "_connect", Connection)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_send, client, event)
        assert marked.wait(timeout=2.0)
        event.set()
        client.cancel_active_requests()
        release_mark.set()
        result = future.result(timeout=2.0)

    assert request_calls == []
    assert result.request_attempts == 0
    assert result.connection_attempts == 1
    assert result.quota_guard_events[0]["post_may_have_started"] is True
    assert result.quota_guard_events[0]["state"] == "committed"
    assert "cancelled before HTTP POST" in (result.error or "")


def test_cancellation_after_transport_error_prevents_second_post(monkeypatch):
    event = threading.Event()
    posts = []

    class Connection:
        sock = None
        timeout = None

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            posts.append("POST")
            event.set()
            raise OSError("ambiguous failure after POST began")

        def close(self):
            pass

    client = _client(retries=1)
    monkeypatch.setattr(client, "_connect", Connection)

    result = _send(client, event)

    assert posts == ["POST"]
    assert result.request_attempts == 1
    # Cancellation prevented the configured retry; the ambiguous first POST
    # remains visible in request_attempts and the error text.
    assert result.retry_reasons == []
    assert "earlier POST may have reached" in (result.error or "")


def test_active_socket_shutdown_interrupts_read_without_retry(monkeypatch):
    event = threading.Event()
    entered_read = threading.Event()
    released = threading.Event()

    class Socket:
        def settimeout(self, _timeout):
            pass

        def shutdown(self, _how):
            released.set()

    class Connection:
        def __init__(self):
            self.sock = Socket()
            self.timeout = None

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            entered_read.set()
            assert released.wait(timeout=2.0)
            raise OSError("socket interrupted by operator cancellation")

        def close(self):
            released.set()

    client = _client(retries=1)
    monkeypatch.setattr(client, "_connect", Connection)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_send, client, event)
        assert entered_read.wait(timeout=1.0)
        event.set()
        assert client.cancel_active_requests() == 1
        result = future.result(timeout=1.0)

    assert result.ok is False
    assert result.request_attempts == 1
    assert result.connection_attempts == 1
    assert result.retry_reasons == []
    assert "earlier POST may have reached" in (result.error or "")


def test_canceller_shuts_socket_without_closing_connection_or_enabling_reconnect():
    shutdowns = []
    closes = []

    class Socket:
        def shutdown(self, how):
            shutdowns.append(how)

    class Connection:
        sock = Socket()

        def close(self):
            # http.client.HTTPConnection.close() clears ``sock``. A later
            # request() would then reconnect automatically.
            closes.append(True)
            self.sock = None

    client = _client()
    connection = Connection()
    client._register_connection(connection)
    try:
        assert client.cancel_active_requests() == 1
    finally:
        client._discard_connection(connection)

    assert len(shutdowns) == 1
    assert closes == []
    assert connection.sock is not None


def test_deadline_watchdog_retries_a_transient_shutdown_failure():
    interrupted = threading.Event()

    class Socket:
        def __init__(self):
            self.calls = 0

        def shutdown(self, _how):
            self.calls += 1
            if self.calls == 1:
                raise OSError("not connected yet")
            interrupted.set()

    class Connection:
        sock = Socket()

    client = _client()
    connection = Connection()
    expired = threading.Event()
    client._register_connection(
        connection, time.monotonic() + 0.01, expired)
    try:
        assert expired.wait(timeout=1.0)
        assert interrupted.wait(timeout=1.0)
        assert connection.sock.calls >= 2
    finally:
        client._discard_connection(connection)


def test_cancellation_during_final_socket_setup_never_posts_or_reconnects(
        monkeypatch):
    event = threading.Event()
    socket_setup = threading.Event()
    release_setup = threading.Event()
    posts = []

    class Socket:
        shutdown_called = False

        def settimeout(self, _timeout):
            socket_setup.set()
            assert release_setup.wait(timeout=2.0)

        def shutdown(self, _how):
            self.shutdown_called = True

    class Connection:
        def __init__(self):
            self.sock = Socket()
            self.timeout = None

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            # If cancellation closed the connection, HTTPConnection.request()
            # would reconnect here. Record either path as a forbidden late POST.
            posts.append("POST")
            raise OSError("late POST attempted")

        def close(self):
            self.sock = None

    client = _client(retries=1)
    monkeypatch.setattr(client, "_connect", Connection)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_send, client, event)
        assert socket_setup.wait(timeout=1.0)
        event.set()
        assert client.cancel_active_requests() == 1
        release_setup.set()
        result = future.result(timeout=1.0)

    assert posts == []
    assert result.ok is False
    assert result.request_attempts == 0
    assert result.first_send_unix is None
    assert "cancelled before HTTP POST" in (result.error or "")


def _result(request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent):
    now = time.time()
    return RequestResult(
        request_id=request_id, scheduled_s=scheduled_s,
        dispatch_lag_ms=dispatch_lag_ms, t_send_unix=now,
        ttfb_ms=1.0, ttft_ms=1.0, ttfr_ms=None, ttfv_ms=1.0,
        e2e_ms=2.0, status=200, ok=True, error=None,
        content_chunks=1, interchunk_max_ms=None, finish_reason="stop",
        prompt_tokens=max(1, intended[0]), completion_tokens=1,
        cached_tokens=0, cached_tokens_source="test",
        intended_input_tokens=intended[0], intended_output_tokens=intended[1],
        intended_cache_fraction=intended[2], doc_id=intended[3],
        chars_sent=chars_sent, stream_complete=True,
        visible_content_seen=True, first_send_unix=now,
        max_tokens_requested=1, connection_attempts=1, request_attempts=1)


def test_keyboard_interrupt_cancels_queued_replay_without_late_send(
        tmp_path, monkeypatch):
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n0\n0\n")
    physical_posts = []
    worker_started = threading.Event()

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, _messages, _max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent, *,
                 scheduled_monotonic=None, cancellation_event=None):
            worker_started.set()
            while not cancellation_event.is_set():
                time.sleep(0.001)
            # This is the exact guard a worker racing out of the queue needs:
            # no physical send begins once cancellation is visible.
            if not cancellation_event.is_set():
                physical_posts.append(request_id)
            return _result(
                request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent)

    class InterruptingProgress:
        def __init__(self, *_args, **_kwargs):
            self.paints = 0

        def sent(self):
            pass

        def done(self, _result):
            pass

        def paint(self):
            self.paints += 1
            if self.paints == 2:
                raise KeyboardInterrupt

        def finish(self):
            pass

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    monkeypatch.setattr("traffic_replay.progress.Progress", InterruptingProgress)
    rc = RunConfig(
        endpoint={
            "base_url": "http://127.0.0.1:1",
            "path": "/serving-endpoints/mock/invocations",
        },
        profile_path="configs/profile_validation_small.json",
        timestamps_file=str(trace), duration_s=1, calibrate_n=0,
        max_concurrency=1, max_pending_requests=3,
        capture_endpoint_metadata=False, measure_network_path=False,
        out_dir=str(tmp_path / "runs"), max_output_tokens_cap=1)

    with pytest.raises(KeyboardInterrupt):
        run(rc, quiet=True)

    assert worker_started.wait(timeout=1.0)
    assert physical_posts == []
    artifacts = list((tmp_path / "runs").iterdir())
    assert len(artifacts) == 1
    start = json.loads((artifacts[0] / "start.json").read_text())
    assert start["status"] != "complete"
    if (artifacts[0] / "requests.jsonl").exists():
        rows = [json.loads(line) for line in
                (artifacts[0] / "requests.jsonl").read_text().splitlines()]
        assert all(row.get("request_attempts") in (0, None) for row in rows)


def test_keyboard_interrupt_journals_a_running_post_outcome(
        tmp_path, monkeypatch):
    trace = tmp_path / "trace-running.txt"
    trace.write_text("0\n")
    posted = threading.Event()

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, _messages, _max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent, *,
                 scheduled_monotonic=None, cancellation_event=None):
            # This represents the irreversible transport boundary.
            posted.set()
            assert cancellation_event.wait(timeout=2.0)
            return _result(
                request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent)

        def cancel_active_requests(self):
            return 1

    class InterruptAfterPost:
        def __init__(self, *_args, **_kwargs):
            pass

        def sent(self):
            pass

        def done(self, _result):
            pass

        def paint(self):
            assert posted.wait(timeout=1.0)
            raise KeyboardInterrupt

        def finish(self):
            pass

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    monkeypatch.setattr("traffic_replay.progress.Progress", InterruptAfterPost)
    rc = RunConfig(
        endpoint={"base_url": "http://127.0.0.1:1",
                  "path": "/serving-endpoints/mock/invocations"},
        profile_path="configs/profile_validation_small.json",
        timestamps_file=str(trace), duration_s=1, calibrate_n=0,
        max_concurrency=1, max_pending_requests=1,
        capture_endpoint_metadata=False, measure_network_path=False,
        out_dir=str(tmp_path / "runs-running"), max_output_tokens_cap=1)

    with pytest.raises(KeyboardInterrupt):
        run(rc, quiet=True)

    artifacts = list((tmp_path / "runs-running").iterdir())
    assert len(artifacts) == 1
    journal = artifacts[0] / "requests.jsonl.partial"
    if not journal.exists():
        journal = artifacts[0] / "requests.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["request_attempts"] == 1
    assert rows[0]["ok"] is True
