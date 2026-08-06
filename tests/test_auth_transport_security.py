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
    assert result.retry_reasons == ["connection_error"]


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
        Connection(_Response(400, b'{"error":"unsupported"}')),
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
