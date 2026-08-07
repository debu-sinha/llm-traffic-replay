"""Security and accounting invariants at the credential/transport boundary."""
from __future__ import annotations

import json
import time

import pytest

from traffic_replay.client import (EndpointClient, EndpointConfig,
                                   UnsafeBearerTransport, normalized_origin)
from traffic_replay.runner import AuthProfileError, _token, _token_from_profile


def _profile_file(tmp_path, text: str) -> str:
    path = tmp_path / "databrickscfg"
    path.write_text(text)
    return str(path)


def test_profile_token_is_bound_to_its_normalized_origin(tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[work]\nhost = HTTPS://EXAMPLE.COM./\ntoken = dapi-not-real\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    # Case, a terminal DNS dot, a trailing slash, and an explicit default port
    # do not turn one origin into four different security identities.
    assert _token_from_profile("work", "https://example.com:443") == \
        "dapi-not-real"
    assert normalized_origin("HTTPS://EXAMPLE.COM./") == \
        ("https", "example.com", 443)


@pytest.mark.parametrize("url", [
    "https://example.com/serving",
    "https://example.com?redirect=elsewhere",
    "https://example.com#fragment",
])
def test_base_url_is_an_origin_and_request_path_is_configured_separately(url):
    with pytest.raises(ValueError, match="must be an origin"):
        EndpointConfig(base_url=url, path="/invocations")


@pytest.mark.parametrize("path", ["relative", "//other-host/path", "/bad\npath"])
def test_request_path_rejects_ambiguous_or_unsafe_forms(path):
    with pytest.raises(ValueError, match="path"):
        EndpointConfig(base_url="https://example.com", path=path)


def test_profile_host_mismatch_fails_before_credential_can_escape(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[work]\nhost = https://trusted.example\ntoken = dapi-secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setenv("SHOULD_NOT_FALL_BACK", "environment-secret")
    endpoint = EndpointConfig(
        base_url="https://attacker.example", path="/invocations",
        auth_profile="work", auth_token_env="SHOULD_NOT_FALL_BACK",
    )

    with pytest.raises(AuthProfileError, match="is bound to") as err:
        _token(endpoint)
    assert "dapi-secret" not in str(err.value)
    assert "environment-secret" not in str(err.value)


def test_missing_profile_host_fails_closed_without_invoking_cli(
        tmp_path, monkeypatch):
    cfg = _profile_file(tmp_path, "[oauth]\nauth_type = databricks-cli\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("CLI must not mint an unbound token")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(AuthProfileError, match="no configured host"):
        _token_from_profile("oauth", "https://workspace.example")
    assert called is False


@pytest.mark.parametrize("url", [
    "http://example.com",
    "http://localhost.example.com",
    "http://10.0.0.1",
])
def test_bearer_token_is_rejected_on_remote_cleartext(url):
    with pytest.raises(UnsafeBearerTransport, match="cleartext HTTP"):
        EndpointClient(EndpointConfig(base_url=url, path="/p"), "secret")


@pytest.mark.parametrize("url", [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://127.255.255.254:8080",
    "http://[::1]:8080",
])
def test_bearer_token_is_allowed_only_on_explicit_loopback_test_hosts(url):
    client = EndpointClient(EndpointConfig(base_url=url, path="/p"), "test")
    assert client.scheme == "http"


class _Sock:
    def settimeout(self, value):
        self.timeout = value


class _Response:
    def __init__(self, status: int, body: bytes = b"", events=()):
        self.status = status
        self._body = body
        self._events = events

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __iter__(self):
        return iter(self._events)


class _TimedFailure:
    sock = _Sock()

    def __init__(self):
        self.request_called_at = None

    def connect(self):
        time.sleep(0.04)

    def request(self, *args, **kwargs):
        self.request_called_at = time.time()
        raise OSError("reset after write began")

    def close(self):
        pass


def test_send_timestamp_excludes_connection_setup_and_attempts_are_explicit():
    conn = _TimedFailure()
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token=None,
    )
    client._connect = lambda: conn
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r1", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.first_attempt_unix is not None
    assert result.first_send_unix is not None
    assert result.first_send_unix - result.first_attempt_unix >= 0.035
    assert abs(result.first_send_unix - conn.request_called_at) < 0.02
    assert result.connection_attempts == 1
    assert result.request_attempts == 1
    assert result.retries == 0
    assert result.retry_reasons == []


def test_exact_caller_clocks_preserve_uniform_schedule_delay():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    scheduled = time.monotonic() - 2.0
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "late", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=scheduled,
    )
    assert 1900 <= result.queue_wait_ms <= 2300
    assert 1900 <= result.caller_ttfb_ms <= 2300
    assert 1900 <= result.caller_ttft_ms <= 2300
    assert 1900 <= result.caller_ttfv_ms <= 2300
    assert result.caller_e2e_ms >= result.caller_ttft_ms
    assert result.ttft_ms < 300


def test_queue_wait_excludes_connection_setup_but_caller_latency_includes_it():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class SlowConnection:
        sock = _Sock()

        def connect(self):
            time.sleep(0.05)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = SlowConnection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "slow-connect", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=time.monotonic(),
    )
    assert result.connect_ms >= 40
    assert result.queue_wait_ms < result.connect_ms
    assert result.caller_ttft_ms >= result.connect_ms


class _ConnectFailure:
    sock = _Sock()

    def connect(self):
        raise OSError("connect refused")

    def close(self):
        pass


def test_failure_before_http_send_is_not_claimed_as_a_wire_send():
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=1),
        token=None,
    )
    client._connect = lambda: _ConnectFailure()
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r2", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.first_attempt_unix is not None
    assert result.first_send_unix is None
    assert result.connection_attempts == 2
    assert result.request_attempts == 0
    assert result.retries == 1
    assert result.retry_reasons == ["connection_error_before_post"]


def test_stream_options_fallback_is_counted_as_a_physical_request_retry():
    seen = []

    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, method, path, body, headers):
            seen.append(json.loads(body))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(_Response(
            400, b'{"error":"stream_options include_usage unsupported"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token="local-test-token",
    )
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r3", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.ok is True
    assert len(seen) == 2
    assert "stream_options" in seen[0]
    assert "stream_options" not in seen[1]
    assert result.connection_attempts == 2
    assert result.request_attempts == 2
    assert result.retries == 1
    assert result.retry_reasons == ["stream_options_rejected"]


def test_exact_caller_clock_includes_automatic_fallback_elapsed_time():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class DelayedResponse(_Response):
        def read(self, n=-1):
            time.sleep(0.04)
            return super().read(n)

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(DelayedResponse(
            400, b'{"error":"stream_options unsupported"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "fallback", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=time.monotonic(),
    )
    assert result.ok is True
    assert result.request_attempts == 2
    assert result.caller_ttft_ms >= 35
    assert result.caller_e2e_ms >= 35
    assert result.ttft_ms < result.caller_ttft_ms


def test_generic_400_is_not_retried_or_persisted_verbatim():
    secret_body = b'{"error":"customer prompt: private-value"}'

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(400, secret_body)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "bad400", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.request_attempts == 1
    assert result.retry_reasons == []
    assert "private-value" not in result.error
    assert "sha256=" in result.error


def test_error_echoing_optional_field_without_rejecting_it_is_not_retried():
    body = (b'{"error":"invalid messages; received request with '
            b'stream_options.include_usage=true"}')

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(400, body)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "bad-messages", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.request_attempts == 1
    assert result.retry_reasons == []


def test_request_serialization_failure_never_opens_a_connection():
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = lambda: (_ for _ in ()).throw(
        AssertionError("must not connect"))
    result = client.send(
        [{"role": "user", "content": object()}], 8, "bad-json", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.first_attempt_unix is None
    assert result.first_send_unix is None
    assert result.connection_attempts == 0
    assert result.request_attempts == 0
    assert result.error == "request serialization failed: TypeError"


def test_permission_403_does_not_trigger_token_refresh():
    refreshed = False

    def refresh():
        nonlocal refreshed
        refreshed = True
        return "new-token"

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(403, b'{"error":"permission denied"}')

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"),
        "old-token", refresh=refresh)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "forbidden", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.request_attempts == 1
    assert refreshed is False


def test_auth_refresh_is_counted_and_only_the_fresh_token_is_retried():
    seen_auth = []
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, method, path, body, headers):
            seen_auth.append(headers.get("Authorization"))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(_Response(401, b'{"error":"expired"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token="old-local-token", refresh=lambda: "new-local-token",
    )
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r4", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.ok is True
    assert seen_auth == ["Bearer old-local-token", "Bearer new-local-token"]
    assert result.request_attempts == 2
    assert result.retries == 1
    assert result.retry_reasons == ["auth_token_refreshed"]


def test_refresh_callback_failure_is_a_result_not_a_worker_exception():
    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(401, b'{"error":"expired"}')

        def close(self):
            pass

    def fail_refresh():
        raise RuntimeError("secret provider detail")

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"),
        "old-token", refresh=fail_refresh)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "refresh-fail", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.error == "credential refresh failed: RuntimeError"
    assert "secret provider detail" not in result.error


def test_refresh_capable_bearer_flow_is_rejected_before_remote_cleartext_io():
    refreshed = False

    def refresh():
        nonlocal refreshed
        refreshed = True
        return "must-not-leak"

    with pytest.raises(UnsafeBearerTransport, match="cleartext HTTP"):
        EndpointClient(
            EndpointConfig(base_url="http://example.com", path="/p"),
            token=None, refresh=refresh,
        )
    assert refreshed is False
