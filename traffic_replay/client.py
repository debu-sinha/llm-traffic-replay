"""Blocking streaming client for OpenAI-compatible chat completions.

Standard library only (http.client), one connection per request, precise
monotonic timing. Concurrency is provided by the runner's thread pool; a
blocked socket read releases the GIL, so hundreds of in-flight requests are
fine, and the runner MEASURES client-side lateness rather than assuming
the client kept up (see runner.py / metrics.py).

Timing definitions, used consistently everywhere:
  t_send           immediately before ``conn.request``; includes upload
  ttfb_ms          first bounded response-body chunk returned by read1
                    (not necessarily the first response byte)
  ttft_ms          first visible, reasoning, or refusal delta; excludes tools
  e2e_ms           stream finished ([DONE] or final chunk)

Usage (prompt/completion/cached token counts) is read from the endpoint's
latest internally consistent cumulative usage block when present.
stream_options.include_usage is requested and automatically retried without
it for endpoints that reject the field.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import copy
import socket
import ssl
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass, asdict, field

from .sse import (StreamState, extract_usage, finalize_tool_calls,
                  iter_sse_events, update_state)
from .network import bind_deadline_bounded_dns


_MAX_PHYSICAL_RETRIES = 2
_OUTPUT_BUDGET_ALIASES = (
    "max_completion_tokens", "max_output_tokens", "max_new_tokens")
_BEARER_TOKEN_MAX_BYTES = 64 * 1024
_STREAM_READ_CHUNK_BYTES = 64 * 1024
_MAX_STREAM_BYTES = 16 * 1024 * 1024
_MAX_STREAM_EVENTS = 100_000
_MAX_STREAM_ERRORS = 64
_MAX_RESPONSE_IDENTITY_FIELD_CHARS = 512
_FRESH_HTTP1_CONNECTION_POLICY = "fresh_http1_per_physical_attempt"


def validate_extra_body_safety(value: dict | None) -> None:
    """Reject credentials from a request-body field persisted as evidence."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("endpoint extra_body must be an object")
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "endpoint extra_body must contain finite JSON values") from exc

    # The exact request parameters are written to run-config.json, start.json,
    # summary.json, reports, and the manifest so a benchmark can be reproduced.
    # Authentication belongs in auth_profile/auth_token_env, never this body.
    from .artifacts import redact_secrets
    safe = json.dumps(
        redact_secrets(value), allow_nan=False, separators=(",", ":"))
    if raw != safe:
        raise ValueError(
            "endpoint extra_body must not contain credentials or secret-like "
            "values because request parameters are persisted as evidence; "
            "use auth_profile or auth_token_env for authentication")


@dataclass
class EndpointConfig:
    base_url: str                    # e.g. https://<workspace-host>
    path: str                        # e.g. /serving-endpoints/<name>/invocations
    auth_token_env: str = "DATABRICKS_TOKEN"
    auth_profile: str | None = None   # a ~/.databrickscfg profile name. takes
                                      # precedence over auth_token_env: PAT is
                                      # direct, databricks-cli is U2M, and a
                                      # client pair is workspace OAuth M2M.
                                      # Route-optimized endpoint-scoped OAuth
                                      # is not implemented by this resolver.
    model: str | None = None         # set for shared /chat/completions routes
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    total_timeout_s: float = 180.0   # absolute worker/request deadline; SSE
                                     # traffic cannot extend it
    temperature: float = 0.0
    max_retries: int = 0             # physical inference retries; 0-2 only
    include_usage: bool = True       # request streamed usage when supported
    extra_body: dict | None = None   # passthrough request params (see _body)
    # Optional declaration of the real application's connection behavior.
    # Capacity conclusions are qualified unless production uses this tool's
    # exact transport contract. A closed enum prevents a vague "equivalent"
    # assertion from silently clearing that gate.
    production_connection_policy: str | None = None

    def __post_init__(self) -> None:
        normalized_origin(self.base_url)
        if not isinstance(self.path, str) or not self.path.startswith("/") \
                or self.path.startswith("//"):
            raise ValueError("endpoint path must start with one / character")
        if any(ord(char) < 0x21 or ord(char) > 0x7e for char in self.path):
            raise ValueError(
                "endpoint path must contain printable ASCII without spaces "
                "or control characters")
        if urllib.parse.urlsplit(self.path).fragment:
            raise ValueError("endpoint path must not contain a URL fragment")
        from .artifacts import redact_secrets
        if redact_secrets(self.path) != self.path:
            raise ValueError(
                "endpoint path must not contain credentials or secret-like "
                "query values; use auth_profile or auth_token_env")
        if self.model is not None \
                and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("endpoint model must be a non-empty string")
        if not isinstance(self.auth_token_env, str) \
                or not self.auth_token_env \
                or not (self.auth_token_env[0].isalpha()
                        or self.auth_token_env[0] == "_") \
                or not all(char.isascii() and (char.isalnum() or char == "_")
                           for char in self.auth_token_env):
            raise ValueError(
                "endpoint auth_token_env must be a valid environment "
                "variable name")
        if self.auth_profile is not None and (
                not isinstance(self.auth_profile, str)
                or not self.auth_profile.strip()
                or self.auth_profile != self.auth_profile.strip()
                or any(ord(char) < 0x21 or ord(char) > 0x7e
                       for char in self.auth_profile)):
            raise ValueError(
                "endpoint auth_profile must be non-empty printable ASCII "
                "without surrounding whitespace")
        for name, value in (("connect_timeout_s", self.connect_timeout_s),
                            ("read_timeout_s", self.read_timeout_s),
                            ("total_timeout_s", self.total_timeout_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"endpoint {name} must be positive and finite")
        if isinstance(self.temperature, bool) \
                or not isinstance(self.temperature, (int, float)) \
                or not math.isfinite(float(self.temperature)):
            raise ValueError("endpoint temperature must be finite")
        if not isinstance(self.max_retries, int) \
                or isinstance(self.max_retries, bool) \
                or not 0 <= self.max_retries <= _MAX_PHYSICAL_RETRIES:
            raise ValueError(
                "endpoint max_retries must be an integer from 0 to "
                f"{_MAX_PHYSICAL_RETRIES}; retries replay inference, consume "
                "quota, and can bias a load test")
        if not isinstance(self.include_usage, bool):
            raise ValueError("endpoint include_usage must be boolean")
        if self.production_connection_policy not in (
                None, _FRESH_HTTP1_CONNECTION_POLICY):
            raise ValueError(
                "endpoint production_connection_policy must be null or "
                f"{_FRESH_HTTP1_CONNECTION_POLICY!r}")
        validate_extra_body_safety(self.extra_body)
        if self.extra_body is not None:
            aliases = [
                key for key in _OUTPUT_BUDGET_ALIASES
                if key in self.extra_body
            ]
            if aliases:
                raise ValueError(
                    "endpoint extra_body must not set output-token budget "
                    "aliases (" + ", ".join(aliases) + "); the harness owns "
                    "max_tokens and the run's max_output_tokens_cap")
        if self.extra_body is not None and "n" in self.extra_body:
            choices = self.extra_body["n"]
            if isinstance(choices, bool) or not isinstance(choices, int) \
                    or choices != 1:
                raise ValueError(
                    "endpoint extra_body.n must be exactly 1 because one "
                    "benchmark request must produce one measured choice")


def serialize_request_body(cfg: EndpointConfig, messages: list[dict],
                           max_tokens: int, include_usage: bool) -> bytes:
    """Build the exact JSON bytes submitted by :class:`EndpointClient`.

    Quota planning uses this same function so roles, message metadata, model,
    tool schemas, provider controls, and JSON framing cannot be omitted from
    its conservative input bound while still appearing on the wire.
    """
    owned = ("messages", "max_tokens", "temperature", "stream",
             "model", "stream_options")
    payload: dict = {k: v for k, v in (cfg.extra_body or {}).items()
                     if k not in owned}
    payload["messages"] = messages
    payload["max_tokens"] = int(max_tokens)
    payload["temperature"] = cfg.temperature
    payload["stream"] = True
    if cfg.model:
        payload["model"] = cfg.model
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")


@dataclass
class RequestResult:
    request_id: str
    scheduled_s: float
    dispatch_lag_ms: float           # dispatcher lateness only. a full pool
                                     # queues, so this does NOT see client
                                     # saturation. metrics computes wire
                                     # lateness from first_send_unix.
    t_send_unix: float | None
    ttfb_ms: float | None
    ttft_ms: float | None            # first content of either kind (back compat)
    ttfr_ms: float | None            # first reasoning-channel delta, else None
    ttfv_ms: float | None            # first visible content delta, else None
    e2e_ms: float | None
    status: int | None
    ok: bool
    error: str | None
    content_chunks: int
    # Widest gap between SSE content-delta events (visible, reasoning, or
    # refusal). This is chunk pacing, not token-level inter-token latency;
    # tool-call-only fragments are excluded.
    interchunk_max_ms: float | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cached_tokens_source: str | None
    intended_input_tokens: int
    intended_output_tokens: int
    intended_cache_fraction: float | None
    doc_id: int                      # pooled document; -1 = no shared prefix
    chars_sent: int
    retries: int = 0
    service_tier: str | None = None        # exact stable tier from SSE chunks
    reasoning_tokens: int | None = None   # thinking tokens, when reported
    reasoning_tokens_source: str | None = None  # usage field it was read from
    reasoning_chunks: int = 0             # reasoning deltas seen in the stream
    connect_ms: float | None = None       # DNS + TCP + TLS setup time
    # transport success (`ok`) is not answer success. a reasoning model that
    # spends its whole token budget thinking returns HTTP 200, a well formed
    # stream, and no answer. these fields carry the facts so metrics can
    # apply the policy in one place.
    stream_complete: bool = False    # saw [DONE] or a finish_reason
    visible_content_seen: bool = False   # at least one visible delta
    reasoning_seen: bool = False
    truncated: bool = False          # finish_reason == "length"
    parse_errors: int = 0            # unrecoverable SSE parse failures
    # Content-free parser diagnostics. The SSE layer never includes streamed
    # text in these strings; malformed payloads are represented only by byte
    # length and a short SHA-256 digest. Bound the list again at persistence.
    parse_error_details: list[str] = field(default_factory=list)
    max_tokens_requested: int | None = None
    first_send_unix: float | None = None  # when the FIRST HTTP request began.
                                          # t_send_unix belongs to whichever
                                          # attempt produced this result, so a
                                          # retried row carries the endpoint's
                                          # delay. connection setup is tracked
                                          # separately by first_attempt_unix.
    first_attempt_unix: float | None = None  # before the first DNS/TCP/TLS try
    connection_attempts: int = 0
    request_attempts: int = 0             # calls that may have emitted a POST
    retry_reasons: list[str] = field(default_factory=list)
    tool_call_seen: bool = False
    tool_call_chunks: int = 0
    ttf_tool_call_ms: float | None = None
    valid_tool_calls: int = 0
    refusal_seen: bool = False
    refusal_chunks: int = 0
    response_content_type: str | None = None
    served_model_name: str | None = None
    response_model: str | None = None
    response_object: str | None = None
    response_id_sha256: str | None = None
    system_fingerprint: str | None = None
    quota_guard_id: str | None = None
    quota_guard_denied: bool = False
    quota_guard_events: list[dict] = field(default_factory=list)
    # Exact caller-experienced clocks, measured from the runner's monotonic
    # scheduled target. These include pool wait, connection setup, and every
    # automatic retry/fallback. They are intentionally separate from the
    # final-attempt request-path clocks above.
    queue_wait_ms: float | None = None
    caller_ttfb_ms: float | None = None
    caller_ttft_ms: float | None = None
    caller_ttfr_ms: float | None = None
    caller_ttfv_ms: float | None = None
    caller_ttf_tool_call_ms: float | None = None
    # Scheduled target to the first conn.request invocation. Request-body
    # upload completion and endpoint receipt are not observed.
    caller_send_ms: float | None = None
    caller_e2e_ms: float | None = None
    # Exact wall-clock completion for every worker result, including HTTP and
    # transport failures. This closes the interval started by first_send_unix
    # without pretending that a failed request occupied zero time.
    finished_unix: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class UnsafeBearerTransport(ValueError):
    """A bearer credential would cross an untrusted cleartext transport."""


def validate_bearer_token(value: object, *, source: str) -> str:
    """Return one header-safe bearer token without echoing credential bytes."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} did not provide a non-empty access token")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        raise ValueError(
            f"{source} returned a bearer token with non-ASCII characters") \
            from None
    if len(encoded) > _BEARER_TOKEN_MAX_BYTES:
        raise ValueError(
            f"{source} returned an oversized bearer token "
            f"(bytes={len(encoded)})")
    if any(byte < 0x21 or byte > 0x7e for byte in encoded):
        raise ValueError(
            f"{source} returned a bearer token with unsafe whitespace or "
            "control characters")
    return value


class _RequestDeadlineExceeded(TimeoutError):
    """The absolute per-request deadline expired."""


def normalized_origin(value: str) -> tuple[str, str, int]:
    """Return a canonical HTTP(S) origin for credential binding.

    Host names are case-folded, IDNA-normalized, and stripped of a terminal
    dot. Explicit default ports compare equal to implicit ones. Userinfo is
    rejected because it makes security-sensitive URL review needlessly
    ambiguous (and is never needed for a serving endpoint).
    """
    if not isinstance(value, str) or not value:
        raise ValueError("endpoint base_url must be a non-empty URL")
    if value != value.strip() or any(
            ord(char) < 0x21 or ord(char) > 0x7e for char in value):
        raise ValueError(
            "endpoint base_url must contain printable ASCII without spaces "
            "or control characters")
    u = urllib.parse.urlsplit(value)
    scheme = u.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("endpoint base_url must use an explicit http or https scheme")
    if u.username is not None or u.password is not None:
        raise ValueError("endpoint base_url must not contain userinfo")
    if u.path not in ("", "/") or u.query or u.fragment:
        raise ValueError(
            "endpoint base_url must be an origin without a path, query, or "
            "fragment; configure the request path separately")
    if u.hostname is None:
        raise ValueError("endpoint base_url must contain a host")
    try:
        port = u.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"endpoint base_url has an invalid port: {exc}") from exc
    raw_host = u.hostname.rstrip(".").lower()
    if not raw_host:
        raise ValueError("endpoint base_url must contain a host")
    try:
        # IPv6 literals contain ':' and are not IDNA names.
        host = (str(ipaddress.ip_address(raw_host)) if ":" in raw_host
                else raw_host.encode("idna").decode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("endpoint base_url contains an invalid host") from exc
    return scheme, host, port


def _is_explicit_loopback(host: str) -> bool:
    """True only for literal loopback addresses or the exact localhost name.

    We intentionally do not resolve arbitrary DNS names: allowing a hostname
    merely because it currently resolves to loopback would permit DNS
    rebinding to turn an approved test URL into a credential sink.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bearer_transport(base_url: str) -> tuple[str, str, int]:
    """Validate where a bearer token may be sent and return its origin."""
    origin = normalized_origin(base_url)
    scheme, host, _ = origin
    if scheme != "https" and not _is_explicit_loopback(host):
        raise UnsafeBearerTransport(
            "refusing to send a bearer token over cleartext HTTP; use HTTPS "
            "or an explicit loopback host for a local test")
    return origin


def _safe_http_error(status: int, body: bytes) -> str:
    """Describe a sampled HTTP error without persisting response content."""
    digest = hashlib.sha256(body).hexdigest()[:16]
    return (f"http {status} (body sample bytes={len(body)}, "
            f"sha256={digest})")


def _stream_options_rejected(body: bytes) -> bool:
    """Only retry a 400 that explicitly identifies our optional field."""
    text = body.decode("utf-8", "replace").casefold()
    names_field = "stream_options" in text or "include_usage" in text
    rejects_field = any(term in text for term in (
        "unsupported", "not supported", "unknown", "unrecognized",
        "unexpected", "not allowed", "not permitted", "cannot",
        "additional propert", "extra field", "invalid field",
        "invalid parameter",
    ))
    return names_field and rejects_field


def _credential_may_be_expired(status: int, body: bytes) -> bool:
    if status == 401:
        return True
    if status != 403:
        return False
    text = body.decode("utf-8", "replace").casefold()
    return any(word in text for word in (
        "invalid token", "expired token", "token expired", "unauthenticated",
    ))


class EndpointClient:
    def __init__(self, cfg: EndpointConfig, token: str | None,
                 refresh: Callable[[], str | None] | None = None,
                 runtime_quota_guard=None):
        """`refresh` returns a fresh token, or None if it cannot.

        An OAuth token is minted once and a load test can outlive it. When
        it expires mid-run every remaining request comes back 401 or 403 and
        reads as an endpoint failure, which is both a wasted run and a
        misleading one. Measured for real: a 90 second run lost 171 of 281
        requests to `http 403: Invalid Token`.
        """
        if token is not None:
            token = validate_bearer_token(token, source="initial credential")
        self.cfg = cfg
        self.token = token
        self._refresh = refresh
        if runtime_quota_guard is not None:
            required = (
                "reserve", "mark_post_may_have_started", "commit",
                "cancel_before_post", "snapshot")
            missing = [name for name in required
                       if not callable(getattr(runtime_quota_guard, name, None))]
            if missing:
                raise TypeError(
                    "runtime_quota_guard is missing method(s): "
                    + ", ".join(missing))
        self.runtime_quota_guard = runtime_quota_guard
        self._lock = threading.Lock()
        self._refresh_condition = threading.Condition(self._lock)
        self._refresh_inflight_for: str | None = None
        self._refresh_failed_for: str | None = None
        self._refresh_failure_error: str | None = None
        self._active_connections_lock = threading.Lock()
        self._deadline_condition = threading.Condition(
            self._active_connections_lock)
        self._active_connections: dict[
            int, tuple[object, float | None, threading.Event | None]] = {}
        self._deadline_thread_started = False
        self.scheme, self.host, self.port = normalized_origin(cfg.base_url)
        # A refresh callback means this is a bearer-auth flow even when the
        # initial token is absent or expired. Reject its transport before the
        # first unauthenticated probe rather than waiting until a token exists.
        if token or refresh is not None:
            validate_bearer_transport(cfg.base_url)
        self._ssl = ssl.create_default_context() if self.scheme == "https" else None
        self._include_usage_supported: bool | None = (
            None if cfg.include_usage else False)  # learned or explicitly off

    def transport_contract(self) -> dict:
        """Public, artifact-safe description of this client's wire behavior."""
        with self._lock:
            include_usage_state = self._include_usage_supported
        actual_policy = _FRESH_HTTP1_CONNECTION_POLICY
        declared_policy = self.cfg.production_connection_policy
        production_match = declared_policy == actual_policy
        warning = None if production_match else (
            "production connection behavior was not declared to match this "
            "fresh-connection HTTP/1.1 client. Fresh connections add "
            "DNS/TCP/TLS pressure and do not reproduce a pooled keep-alive "
            "or HTTP/2 client; transport-limited capacity is inconclusive")
        return {
            "implementation": "python_stdlib_http_client",
            "http_protocol": "HTTP/1.1",
            "connection_reuse": False,
            "connection_policy": "fresh connection per physical attempt",
            "connection_policy_id": actual_policy,
            "http2": False,
            "production_connection_policy_declared": declared_policy,
            "production_connection_policy_match": production_match,
            "production_connection_policy_assurance": (
                "operator asserted that the production application opens a "
                "fresh HTTP/1.1 connection for every physical attempt; the "
                "harness recorded the assertion but did not observe the "
                "production client"
                if production_match else None),
            "production_comparability_warning": warning,
            "connect_timeout_s": float(self.cfg.connect_timeout_s),
            "read_idle_timeout_s": float(self.cfg.read_timeout_s),
            "absolute_request_timeout_s": float(self.cfg.total_timeout_s),
            "stream_read_chunk_bytes": _STREAM_READ_CHUNK_BYTES,
            "max_stream_bytes": _MAX_STREAM_BYTES,
            "max_stream_events": _MAX_STREAM_EVENTS,
            "max_stream_validation_errors": _MAX_STREAM_ERRORS,
            "required_success_content_type": "text/event-stream",
            "include_usage_configured": self.cfg.include_usage,
            "include_usage_support_state": include_usage_state,
        }

    def _refresh_after_rejection(
            self, rejected_token: str | None,
            deadline_monotonic: float) -> tuple[bool, str | None]:
        """Single-flight a refresh and wait no longer than this request.

        The refresh provider may itself be a blocking CLI or HTTP operation.
        It therefore runs in one daemon thread while request workers wait on a
        condition bounded by their own absolute deadlines. A failed refresh is
        cached for the rejected token generation so a burst of 401s cannot
        serialize the same doomed operation hundreds of times.
        """
        if self._refresh is None:
            return False, None

        def refresh_worker() -> None:
            fresh = None
            error = None
            try:
                fresh = self._refresh()
                if fresh is not None:
                    fresh = validate_bearer_token(
                        fresh, source="credential refresh")
                    validate_bearer_transport(self.cfg.base_url)
                if not fresh or fresh == rejected_token:
                    error = "credential refresh did not replace rejected token"
            except Exception as exc:
                error = f"credential refresh failed: {type(exc).__name__}"
                fresh = None
            with self._refresh_condition:
                if (fresh and self.token == rejected_token
                        and self._refresh_inflight_for == rejected_token):
                    self.token = fresh
                    self._refresh_failed_for = None
                    self._refresh_failure_error = None
                elif self.token == rejected_token:
                    self._refresh_failed_for = rejected_token
                    self._refresh_failure_error = error or \
                        "credential refresh failed"
                self._refresh_inflight_for = None
                self._refresh_condition.notify_all()

        with self._refresh_condition:
            while True:
                if self.token != rejected_token:
                    return True, None
                if self._refresh_failed_for == rejected_token:
                    return False, self._refresh_failure_error
                if self._refresh_inflight_for is None:
                    self._refresh_inflight_for = rejected_token
                    threading.Thread(
                        target=refresh_worker,
                        name="traffic-replay-auth-refresh",
                        daemon=True,
                    ).start()
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    raise _RequestDeadlineExceeded
                self._refresh_condition.wait(timeout=remaining)

    def _connect(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.cfg.connect_timeout_s,
                context=self._ssl)
        return http.client.HTTPConnection(
            self.host, self.port, timeout=self.cfg.connect_timeout_s)

    def _deadline_loop(self) -> None:
        """One watchdog interrupts every connection at its wall deadline."""
        while True:
            expired = []
            with self._deadline_condition:
                timed = [
                    item for item in self._active_connections.values()
                    if item[1] is not None]
                if not timed:
                    # Do not retain every short-lived EndpointClient forever
                    # through an idle bound-method daemon. Registration and
                    # this transition share the same lock, so a later lease
                    # reliably starts a fresh monitor.
                    self._deadline_thread_started = False
                    return
                now = time.monotonic()
                next_deadline = min(float(item[1]) for item in timed)
                if next_deadline > now:
                    self._deadline_condition.wait(next_deadline - now)
                    continue
                for key, (conn, deadline, expired_event) in list(
                        self._active_connections.items()):
                    if deadline is not None and deadline <= now:
                        if expired_event is not None:
                            expired_event.set()
                        # Revisit until shutdown actually succeeds. A socket
                        # may exist while connect/TLS is still in a state where
                        # shutdown returns ENOTCONN; swallowing that once and
                        # clearing the deadline would strand the worker.
                        self._active_connections[key] = (
                            conn, now + 0.01, expired_event)
                        if getattr(conn, "sock", None) is not None:
                            expired.append((key, conn))
            for key, conn in expired:
                sock = getattr(conn, "sock", None)
                interrupted = False
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                        interrupted = True
                    except (AttributeError, OSError, ValueError):
                        pass
                if interrupted:
                    with self._deadline_condition:
                        current = self._active_connections.get(key)
                        if current is not None and current[0] is conn:
                            self._active_connections[key] = (
                                conn, None, current[2])
                            self._deadline_condition.notify_all()

    def _register_connection(
            self, conn, deadline_monotonic: float | None = None,
            expired_event: threading.Event | None = None) -> None:
        with self._deadline_condition:
            self._active_connections[id(conn)] = (
                conn, deadline_monotonic, expired_event)
            if deadline_monotonic is not None \
                    and not self._deadline_thread_started:
                self._deadline_thread_started = True
                threading.Thread(
                    target=self._deadline_loop,
                    name="traffic-replay-deadline-watchdog",
                    daemon=True,
                ).start()
            self._deadline_condition.notify_all()

    def _discard_connection(self, conn) -> None:
        with self._deadline_condition:
            self._active_connections.pop(id(conn), None)
            self._deadline_condition.notify_all()

    def cancel_active_requests(self) -> int:
        """Best-effort interruption of sockets already blocked in I/O.

        The runner sets each request's cooperative cancellation event before
        calling this method. Shutting down the socket wakes a blocked read;
        the worker then observes cancellation and cannot retry. A POST that
        was already on the wire remains an unknown provider outcome and is
        reported as such.

        Do not call ``HTTPConnection.close`` from this thread. ``close`` sets
        ``conn.sock`` to ``None``; a worker in the narrow interval between its
        cancellation check and ``conn.request`` would then auto-connect a new
        socket and could emit a late POST. Keeping the shut-down socket attached
        makes that request fail instead. The worker owns the final close in its
        ``finally`` block.
        """
        with self._active_connections_lock:
            active = [item[0] for item in self._active_connections.values()]
        for conn in active:
            sock = getattr(conn, "sock", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except (OSError, ValueError):
                    pass
        return len(active)

    def _body(self, messages: list[dict], max_tokens: int,
              include_usage: bool) -> bytes:
        # extra_body is user passthrough (top_p, stop, response_format, and
        # provider thinking control like reasoning_effort / thinking /
        # chat_template_kwargs). The harness owns the keys below: they are
        # popped first so nothing in extra_body can survive, then set from
        # their dedicated config, so a run stays measurable no matter what
        # the user put in extra_body.
        return serialize_request_body(
            self.cfg, messages, max_tokens, include_usage)

    def send(self, messages: list[dict], max_tokens: int, request_id: str,
             scheduled_s: float, dispatch_lag_ms: float,
             intended: tuple[int, int, float, int],
             chars_sent: int, *,
             scheduled_monotonic: float | None = None,
             cancellation_event: threading.Event | None = None) \
            -> RequestResult:
        """One request, fully measured. Never raises; errors land in result."""
        attempt = 0
        with self._lock:
            include_usage = self._include_usage_supported is not False
        last_err: str | None = None
        # Connection start and HTTP send are different events. In particular,
        # a DNS/TCP/TLS failure did not put a request on the wire and must not
        # be recorded as though it did.
        first_attempt_unix: float | None = None
        first_send_unix: float | None = None
        last_send_unix: float | None = None
        connection_attempts = 0
        request_attempts = 0
        retry_reasons: list[str] = []
        auth_retried = False
        queue_wait_ms = None
        caller_ttfb_ms = caller_ttft_ms = None
        caller_ttfr_ms = caller_ttfv_ms = None
        caller_ttf_tool_call_ms = None
        caller_send_ms = None
        quota_guard_events: list[dict] = []
        quota_guard_denied = False
        # These live outside the retry body so result construction can settle
        # the current physical-attempt reservation before copying its event
        # evidence.  Python evaluates a return value before running ``finally``;
        # relying on the attempt's finally block alone would persist an event
        # as provisional even though the guard committed it moments later.
        quota_handle = None
        quota_post_marked = False
        worker_started_unix = time.time()
        worker_started_monotonic = time.monotonic()
        deadline_monotonic = (
            worker_started_monotonic + float(self.cfg.total_timeout_s))
        deadline_expired = threading.Event()

        def remaining_s(now: float | None = None) -> float:
            if deadline_expired.is_set():
                raise _RequestDeadlineExceeded
            if now is None:
                now = time.monotonic()
            remaining = deadline_monotonic - now
            if remaining <= 0:
                raise _RequestDeadlineExceeded
            return remaining

        def cap_connect_timeout(conn) -> None:
            # HTTPConnection.connect() reads this attribute when creating its
            # socket. Cap it to the absolute request budget so DNS/TCP/TLS
            # setup cannot outlive the request as a whole.
            conn.timeout = min(
                float(self.cfg.connect_timeout_s), remaining_s())

        def cap_socket_timeout(conn) -> None:
            # Socket timeouts are idle timeouts. Recompute the timeout before
            # every blocking read so a stream of heartbeats cannot keep a
            # request alive beyond total_timeout_s.
            timeout = min(float(self.cfg.read_timeout_s), remaining_s())
            sock = getattr(conn, "sock", None)
            if sock is None:
                conn.timeout = timeout
            else:
                sock.settimeout(timeout)

        timeout_error = (
            "request exceeded total timeout "
            f"(total_timeout_s={float(self.cfg.total_timeout_s):g})")

        def caller_elapsed(now: float | None = None) -> float | None:
            if scheduled_monotonic is None:
                return None
            if now is None:
                now = time.monotonic()
            return max((now - scheduled_monotonic) * 1000.0, 0.0)

        def settle_quota_attempt() -> None:
            if quota_handle is None:
                return
            try:
                if quota_post_marked:
                    # Idempotent after a response-header commit.  Otherwise
                    # conn.request may have reached the provider, so retain
                    # the reservation rather than manufacture quota headroom.
                    self.runtime_quota_guard.commit(
                        quota_handle,
                        reason="attempt_ended_without_receipt_proof")
                else:
                    self.runtime_quota_guard.cancel_before_post(
                        quota_handle, reason="attempt_ended_before_post")
            except Exception:
                # The guard records/trips on internal transition failures.
                # Preserve the transport result; report validation will reject
                # any nonterminal or otherwise inconsistent guard evidence.
                pass

        def caller_kwargs() -> dict:
            # Must precede the deep copy in _finish().  A return expression is
            # evaluated before the surrounding attempt finally block runs.
            settle_quota_attempt()
            return {
                "scheduled_monotonic": scheduled_monotonic,
                "queue_wait_ms": queue_wait_ms,
                "caller_ttfb_ms": caller_ttfb_ms,
                "caller_ttft_ms": caller_ttft_ms,
                "caller_ttfr_ms": caller_ttfr_ms,
                "caller_ttfv_ms": caller_ttfv_ms,
                "caller_ttf_tool_call_ms": caller_ttf_tool_call_ms,
                "caller_send_ms": caller_send_ms,
                "quota_guard_id": (
                    getattr(self.runtime_quota_guard, "guard_id", None)
                    if self.runtime_quota_guard is not None else None),
                "quota_guard_denied": quota_guard_denied,
                "quota_guard_events": quota_guard_events,
                "worker_started_unix": worker_started_unix,
                "worker_started_monotonic": worker_started_monotonic,
            }

        # send() begins when a worker actually receives this request. Capture
        # schedule-to-worker delay here; connection setup is a separate clock
        # and must not be mislabeled as queue wait.
        queue_wait_ms = caller_elapsed()
        state = StreamState()

        def is_cancelled() -> bool:
            return bool(cancellation_event is not None
                        and cancellation_event.is_set())

        def cancelled_result() -> RequestResult:
            stage = ("before HTTP POST" if request_attempts == 0 else
                     "before retry; an earlier POST may have reached the "
                     "provider")
            return self._finish(
                request_id, scheduled_s, dispatch_lag_ms,
                last_send_unix, None, None, None, None, False,
                f"request cancelled {stage}", state, intended, chars_sent,
                len(retry_reasons), None, None, None, None,
                first_send_unix, max_tokens,
                first_attempt_unix=first_attempt_unix,
                connection_attempts=connection_attempts,
                request_attempts=request_attempts,
                retry_reasons=retry_reasons,
                **caller_kwargs())

        while attempt <= self.cfg.max_retries:
            # Checked at worker entry and at the top of every retry. The
            # runner sets this before cancelling queued futures, so a task
            # racing out of the queue still cannot emit a POST.
            if is_cancelled():
                return cancelled_result()
            attempt += 1
            conn = None
            posts_before_attempt = request_attempts
            state = StreamState()
            t_send = None
            t_send_unix = None
            connect_ms = None
            response_status = None
            ttfb_ms = ttft_ms = ttfr_ms = ttfv_ms = None
            ttf_tool_call_ms = None
            # These are caller-experienced clocks for the response produced
            # by this physical attempt. Their origin stays the logical
            # request's scheduled target, so retry delay is still included,
            # but an event observed on a failed earlier stream must not be
            # attached to the final response.
            caller_ttfb_ms = caller_ttft_ms = None
            caller_ttfr_ms = caller_ttfv_ms = None
            caller_ttf_tool_call_ms = None
            interchunk_max = None
            quota_handle = None
            quota_post_marked = False
            try:
                remaining_s()
                try:
                    body = self._body(messages, max_tokens, include_usage)
                except (TypeError, ValueError, OverflowError) as exc:
                    return self._finish(
                        request_id, scheduled_s, dispatch_lag_ms,
                        None, None, None, None, None, False,
                        f"request serialization failed: {type(exc).__name__}",
                        StreamState(), intended, chars_sent,
                        len(retry_reasons), None, None, None, None,
                        first_send_unix, max_tokens,
                        first_attempt_unix=first_attempt_unix,
                        connection_attempts=connection_attempts,
                        request_attempts=request_attempts,
                        retry_reasons=retry_reasons,
                        **caller_kwargs())
                conn = self._connect()
                # ``socket`` timeouts begin only after DNS. Replace
                # http.client's socket factory so resolution and TCP setup
                # share the capped connect budget. The resolver helper is
                # daemon-only and DNS-only: cancellation/deadline expiry can
                # return this worker without permitting a late connection or
                # POST when a blocked lookup eventually finishes.
                bind_deadline_bounded_dns(
                    conn, cancel_event=cancellation_event)
                self._register_connection(
                    conn, deadline_monotonic, deadline_expired)
                connection_attempts += 1
                if first_attempt_unix is None:
                    first_attempt_unix = time.time()
                cap_connect_timeout(conn)
                t_conn0 = time.monotonic()
                conn.connect()
                remaining_s()
                connect_ms = (time.monotonic() - t_conn0) * 1000.0
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Request-Id": request_id,
                }
                tok_used = self.token
                if tok_used:
                    # Constructor validation covers the normal path. Recheck
                    # here as a defense against a caller mutating client.token.
                    tok_used = validate_bearer_token(
                        tok_used, source="active credential")
                    validate_bearer_transport(self.cfg.base_url)
                    headers["Authorization"] = f"Bearer {tok_used}"

                # Socket timeout setup is normally non-blocking, but keep it
                # ahead of the last cancellation check. This closes the race
                # where cancellation becomes visible while the connection is
                # being prepared for its POST.
                if is_cancelled():
                    return cancelled_result()
                cap_socket_timeout(conn)
                if is_cancelled():
                    return cancelled_result()

                if self.runtime_quota_guard is not None:
                    try:
                        quota_handle = self.runtime_quota_guard.reserve(
                            body=body,
                            message_count=len(messages),
                            max_tokens=int(max_tokens),
                            request_id=request_id,
                            attempt_ordinal=request_attempts + 1,
                            retry_trigger=(retry_reasons[-1]
                                           if retry_reasons else None),
                        )
                        quota_event = getattr(quota_handle, "event", None)
                        if not isinstance(quota_event, dict):
                            raise TypeError(
                                "quota admission handle has no event object")
                        quota_guard_events.append(quota_event)
                    except Exception as exc:
                        quota_guard_denied = True
                        return self._finish(
                            request_id, scheduled_s, dispatch_lag_ms,
                            last_send_unix, None, None, None, None, False,
                            "runtime quota admission failed closed before "
                            f"HTTP POST ({type(exc).__name__})",
                            StreamState(), intended, chars_sent,
                            len(retry_reasons), None, None, None, connect_ms,
                            first_send_unix, max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())
                    if quota_event.get("decision") != "admitted":
                        quota_guard_denied = True
                        # A denial created no provisional reservation. Keep
                        # its event evidence, but do not ask the attempt
                        # cleanup path to cancel a nonexistent admission.
                        quota_handle = None
                        stage = (
                            "before first HTTP POST" if request_attempts == 0
                            else "before retry; an earlier POST may have "
                                 "reached the provider")
                        return self._finish(
                            request_id, scheduled_s, dispatch_lag_ms,
                            last_send_unix, None, None, None, None, False,
                            f"runtime quota admission refused {stage}",
                            StreamState(), intended, chars_sent,
                            len(retry_reasons), None, None, None, connect_ms,
                            first_send_unix, max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())
                    if is_cancelled():
                        try:
                            self.runtime_quota_guard.cancel_before_post(
                                quota_handle, reason="request_cancelled")
                        except Exception:
                            # Keep the rich provisional event and exact zero
                            # POST count.  The guard has failed closed and its
                            # snapshot/invariant validation will make this run
                            # non-publishable.
                            quota_guard_denied = True
                        return cancelled_result()
                    try:
                        self.runtime_quota_guard.mark_post_may_have_started(
                            quota_handle)
                    except Exception as exc:
                        quota_guard_denied = True
                        return self._finish(
                            request_id, scheduled_s, dispatch_lag_ms,
                            last_send_unix, None, None, None, None, False,
                            "runtime quota guard failed closed before HTTP "
                            f"POST ({type(exc).__name__})",
                            StreamState(), intended, chars_sent,
                            len(retry_reasons), None, None, None, connect_ms,
                            first_send_unix, max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())
                    quota_post_marked = True
                    # Admission marking takes the guard lock and can briefly
                    # wait behind other workers.  Cancellation may become
                    # visible during that transition, after the earlier check
                    # but before this worker has invoked conn.request.  Recheck
                    # at the returned boundary so an operator stop cannot turn
                    # that lock wait into a late physical POST.  The marked
                    # reservation is retained conservatively by
                    # cancelled_result(): at this point the provider outcome is
                    # known to be unsent locally, but releasing after the guard's
                    # explicit may-have-started transition would weaken its
                    # fail-closed state machine.
                    if is_cancelled():
                        return cancelled_result()
                    # The same guard-lock interval is inside this worker's
                    # absolute deadline.  Do not start a POST after that budget
                    # elapsed merely because the socket watchdog raced while
                    # admission was being marked.
                    remaining_s()

                # The final-attempt clock begins immediately before the
                # blocking conn.request call. It therefore includes request
                # upload; it does not claim to begin when the last request
                # byte reaches the socket or provider.
                t_send = time.monotonic()
                t_send_unix = time.time()
                last_send_unix = t_send_unix
                if first_send_unix is None:
                    first_send_unix = t_send_unix
                    caller_send_ms = caller_elapsed(t_send)
                request_attempts += 1
                conn.request("POST", self.cfg.path, body=body, headers=headers)
                remaining_s()
                cap_socket_timeout(conn)
                resp = conn.getresponse()
                remaining_s()
                response_status = resp.status
                if quota_handle is not None:
                    try:
                        self.runtime_quota_guard.commit(
                            quota_handle, reason="response_headers_received")
                    except Exception as exc:
                        quota_guard_denied = True
                        failed_at = time.monotonic()
                        return self._finish(
                            request_id, scheduled_s, dispatch_lag_ms,
                            t_send_unix, None, None,
                            max((failed_at - t_send) * 1000.0, 0.0),
                            response_status, False,
                            "runtime quota guard failed closed after HTTP "
                            f"response headers ({type(exc).__name__})",
                            state, intended, chars_sent, len(retry_reasons),
                            None, None, None, connect_ms, first_send_unix,
                            max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())

                # A real HTTPResponse always exposes getheader(). Test
                # adapters without it remain usable, but a production 200
                # must prove that the body is an SSE stream before it can be
                # accepted as benchmark evidence.
                get_header = getattr(resp, "getheader", None)
                if resp.status == 200 and callable(get_header):
                    served_model_name = get_header("served-model-name")
                    if served_model_name is not None:
                        if not isinstance(served_model_name, str) \
                                or not served_model_name.strip():
                            state.errors.append(
                                "HTTP served-model-name response header must "
                                "be a non-empty string")
                        elif len(served_model_name) > \
                                _MAX_RESPONSE_IDENTITY_FIELD_CHARS:
                            state.errors.append(
                                "HTTP served-model-name response header "
                                "exceeded the 512-character safety limit")
                        else:
                            state.served_model_name = served_model_name.strip()
                    content_type = get_header("Content-Type")
                    media_type = (
                        content_type.split(";", 1)[0].strip().lower()
                        if isinstance(content_type, str) else "")
                    if media_type != "text/event-stream":
                        state.errors.append(
                            "HTTP 200 response Content-Type was not "
                            "text/event-stream")
                        failed_at = time.monotonic()
                        return self._finish(
                            request_id, scheduled_s, dispatch_lag_ms,
                            t_send_unix, None, None,
                            (failed_at - t_send) * 1000.0,
                            200, False, "stream protocol validation failed",
                            state, intended, chars_sent, len(retry_reasons),
                            None, None, None, connect_ms, first_send_unix,
                            max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())
                    state.response_content_type = media_type

                if resp.status == 400 and include_usage:
                    cap_socket_timeout(conn)
                    detail = resp.read(64 * 1024)
                    remaining_s()
                    if _stream_options_rejected(detail):
                        # This is a real second POST and is recorded as such.
                        with self._lock:
                            self._include_usage_supported = False
                        include_usage = False
                        retry_reasons.append("stream_options_rejected")
                        attempt -= 1
                        continue
                    return self._finish(
                        request_id, scheduled_s, dispatch_lag_ms,
                        t_send_unix, None, None, None, resp.status, False,
                        _safe_http_error(resp.status, detail), StreamState(),
                        intended, chars_sent, len(retry_reasons), None, None,
                        None, connect_ms, first_send_unix, max_tokens,
                        first_attempt_unix=first_attempt_unix,
                        connection_attempts=connection_attempts,
                        request_attempts=request_attempts,
                        retry_reasons=retry_reasons,
                        **caller_kwargs())

                if resp.status in (401, 403) and self._refresh:
                    cap_socket_timeout(conn)
                    detail = resp.read(64 * 1024)
                    remaining_s()
                    # keep the real reason. falling out of the retry loop
                    # with "exhausted retries" hides an auth problem, which
                    # is the most common thing to get wrong.
                    last_err = _safe_http_error(resp.status, detail)
                    try:
                        conn.close()
                    except (OSError, http.client.HTTPException):
                        pass
                    # this is a concurrent load generator, so when a token
                    # expires MANY requests fail at once. each of them must
                    # get a retry against the new token, and only the first
                    # of them should spend a refresh. comparing against the
                    # token this request actually used, rather than against
                    # the shared one, is what makes that true: a thread that
                    # arrives after someone else refreshed simply retries.
                    retry_auth = False
                    if not auth_retried and _credential_may_be_expired(
                            resp.status, detail):
                        retry_auth, refresh_error = \
                            self._refresh_after_rejection(
                                tok_used, deadline_monotonic)
                        if refresh_error:
                            last_err = refresh_error
                    if retry_auth:
                        auth_retried = True
                        retry_reasons.append("auth_token_refreshed")
                        attempt -= 1
                        continue
                    return self._finish(
                        request_id, scheduled_s, dispatch_lag_ms,
                        t_send_unix, None, None, None, resp.status, False,
                        last_err, StreamState(), intended, chars_sent,
                        len(retry_reasons), None, None, None, connect_ms,
                        first_send_unix, max_tokens,
                        first_attempt_unix=first_attempt_unix,
                        connection_attempts=connection_attempts,
                        request_attempts=request_attempts,
                        retry_reasons=retry_reasons,
                        **caller_kwargs())

                if resp.status != 200:
                    cap_socket_timeout(conn)
                    detail = resp.read(64 * 1024)
                    remaining_s()
                    return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                                        t_send_unix, None, None, None,
                                        resp.status, False,
                                        _safe_http_error(resp.status, detail),
                                        StreamState(), intended, chars_sent,
                                        len(retry_reasons), None, None, None,
                                        connect_ms, first_send_unix,
                                        max_tokens,
                                        first_attempt_unix=first_attempt_unix,
                                        connection_attempts=connection_attempts,
                                        request_attempts=request_attempts,
                                        retry_reasons=retry_reasons,
                                        **caller_kwargs())

                if include_usage:
                    with self._lock:
                        if self._include_usage_supported is None:
                            self._include_usage_supported = True

                last_content_t = None

                def timed_lines():
                    nonlocal ttfb_ms, caller_ttfb_ms
                    read1 = getattr(resp, "read1", None)
                    response_lines = None if callable(read1) else iter(resp)
                    stream_bytes = 0
                    while True:
                        cap_socket_timeout(conn)
                        if read1 is not None and callable(read1):
                            raw = read1(_STREAM_READ_CHUNK_BYTES)
                            if not raw:
                                return
                        else:
                            try:
                                raw = next(response_lines)
                            except StopIteration:
                                return
                        now = time.monotonic()
                        remaining_s(now)
                        if not isinstance(raw, (bytes, str)):
                            raise http.client.HTTPException(
                                "response stream yielded a non-byte chunk")
                        raw_size = (len(raw) if isinstance(raw, bytes)
                                    else len(raw.encode(
                                        "utf-8", errors="surrogatepass")))
                        stream_bytes += raw_size
                        if stream_bytes > _MAX_STREAM_BYTES:
                            state.errors.append(
                                "response stream exceeded the cumulative "
                                f"{_MAX_STREAM_BYTES}-byte safety limit")
                            return
                        if ttfb_ms is None:
                            ttfb_ms = (now - t_send) * 1000.0
                            caller_ttfb_ms = caller_elapsed(now)
                        # Iterator-only test adapters may return a large raw
                        # line. Slice it before the SSE parser sees it so its
                        # own physical-line cap applies incrementally too.
                        for start in range(0, len(raw),
                                           _STREAM_READ_CHUNK_BYTES):
                            yield raw[start:start + _STREAM_READ_CHUNK_BYTES]

                event_count = 0
                for event in iter_sse_events(timed_lines()):
                    event_count += 1
                    if event_count > _MAX_STREAM_EVENTS:
                        state.errors.append(
                            "response stream exceeded the cumulative "
                            f"{_MAX_STREAM_EVENTS}-event safety limit")
                        break
                    now = time.monotonic()
                    chunks_before = state.content_chunks
                    reasoning_before = state.saw_first_reasoning
                    visible_before = state.saw_first_visible
                    tool_before = state.saw_first_tool_call
                    first = update_state(state, event)
                    if len(state.errors) >= _MAX_STREAM_ERRORS:
                        state.errors[:] = state.errors[:_MAX_STREAM_ERRORS]
                        state.errors.append(
                            "additional stream validation errors omitted after "
                            f"the {_MAX_STREAM_ERRORS}-error safety limit")
                        break
                    if first and ttft_ms is None:
                        ttft_ms = (now - t_send) * 1000.0
                        caller_ttft_ms = caller_elapsed(now)
                    if state.saw_first_reasoning and not reasoning_before:
                        ttfr_ms = (now - t_send) * 1000.0
                        caller_ttfr_ms = caller_elapsed(now)
                    if state.saw_first_visible and not visible_before:
                        ttfv_ms = (now - t_send) * 1000.0
                        caller_ttfv_ms = caller_elapsed(now)
                    if state.saw_first_tool_call and not tool_before:
                        ttf_tool_call_ms = (now - t_send) * 1000.0
                        caller_ttf_tool_call_ms = caller_elapsed(now)
                    if state.content_chunks > chunks_before:
                        if last_content_t is not None:
                            gap = (now - last_content_t) * 1000.0
                            if interchunk_max is None or gap > interchunk_max:
                                interchunk_max = gap
                        last_content_t = now
                    if state.done:
                        break
                finished_stream_at = time.monotonic()
                remaining_s(finished_stream_at)
                e2e_ms = (finished_stream_at - t_send) * 1000.0
                finalize_tool_calls(state)
                has_output = (
                    state.saw_first_content or state.valid_tool_calls > 0)
                stream_complete = bool(state.done or state.finish_reason)
                if state.errors:
                    ok = False
                    err = "stream protocol validation failed"
                elif not stream_complete:
                    ok = False
                    err = (
                        "stream ended without [DONE] or a finish_reason")
                elif not has_output:
                    ok = False
                    err = "stream ended with no content or valid tool call"
                else:
                    ok = True
                    err = None
                return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                                    t_send_unix, ttfb_ms, ttft_ms, e2e_ms,
                                    200, ok, err, state, intended, chars_sent,
                                    len(retry_reasons), interchunk_max,
                                    ttfr_ms, ttfv_ms, connect_ms,
                                    first_send_unix, max_tokens,
                                    first_attempt_unix=first_attempt_unix,
                                    connection_attempts=connection_attempts,
                                    request_attempts=request_attempts,
                                    retry_reasons=retry_reasons,
                                    ttf_tool_call_ms=ttf_tool_call_ms,
                                    **caller_kwargs())

            except _RequestDeadlineExceeded:
                finished_at = time.monotonic()
                e2e_ms = (
                    max((finished_at - t_send) * 1000.0, 0.0)
                    if t_send is not None else None)
                return self._finish(
                    request_id, scheduled_s, dispatch_lag_ms,
                    t_send_unix, ttfb_ms, ttft_ms, e2e_ms,
                    response_status, False, timeout_error, state, intended,
                    chars_sent, len(retry_reasons), interchunk_max,
                    ttfr_ms, ttfv_ms, connect_ms, first_send_unix,
                    max_tokens,
                    first_attempt_unix=first_attempt_unix,
                    connection_attempts=connection_attempts,
                    request_attempts=request_attempts,
                    retry_reasons=retry_reasons,
                    ttf_tool_call_ms=ttf_tool_call_ms,
                    **caller_kwargs())
            except (OSError, http.client.HTTPException) as exc:
                if is_cancelled():
                    return cancelled_result()
                if deadline_expired.is_set() \
                        or time.monotonic() >= deadline_monotonic:
                    finished_at = time.monotonic()
                    e2e_ms = (
                        max((finished_at - t_send) * 1000.0, 0.0)
                        if t_send is not None else None)
                    return self._finish(
                        request_id, scheduled_s, dispatch_lag_ms,
                        t_send_unix, ttfb_ms, ttft_ms, e2e_ms,
                        response_status, False, timeout_error, state,
                        intended, chars_sent, len(retry_reasons),
                        interchunk_max, ttfr_ms, ttfv_ms, connect_ms,
                        first_send_unix, max_tokens,
                        first_attempt_unix=first_attempt_unix,
                        connection_attempts=connection_attempts,
                        request_attempts=request_attempts,
                        retry_reasons=retry_reasons,
                        ttf_tool_call_ms=ttf_tool_call_ms,
                        **caller_kwargs())
                last_err = f"transport failed: {type(exc).__name__}"
                if attempt <= self.cfg.max_retries:
                    retry_reasons.append(
                        "transport_error_after_post"
                        if request_attempts > posts_before_attempt else
                        "connection_error_before_post")
                    continue
                # The final failed physical attempt may already have returned
                # HTTP headers and partial SSE output. Preserve those facts;
                # replacing them with a blank state corrupts failure analysis.
                failed_at = time.monotonic()
                e2e_ms = (
                    max((failed_at - t_send) * 1000.0, 0.0)
                    if t_send is not None else None)
                finalize_tool_calls(state)
                return self._finish(
                    request_id, scheduled_s, dispatch_lag_ms,
                    t_send_unix, ttfb_ms, ttft_ms, e2e_ms,
                    response_status, False, last_err, state, intended,
                    chars_sent, len(retry_reasons), interchunk_max,
                    ttfr_ms, ttfv_ms, connect_ms, first_send_unix,
                    max_tokens,
                    first_attempt_unix=first_attempt_unix,
                    connection_attempts=connection_attempts,
                    request_attempts=request_attempts,
                    retry_reasons=retry_reasons,
                    ttf_tool_call_ms=ttf_tool_call_ms,
                    **caller_kwargs())
            finally:
                settle_quota_attempt()
                if conn is not None:
                    self._discard_connection(conn)
                    try:
                        conn.close()
                    except (OSError, http.client.HTTPException):
                        pass

        return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                            last_send_unix,
                            None, None, None, None, False,
                            last_err or "exhausted retries", StreamState(),
                            intended, chars_sent, len(retry_reasons),
                            None, None, None, None, first_send_unix,
                            max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons,
                            **caller_kwargs())

    @staticmethod
    def _finish(request_id, scheduled_s, dispatch_lag_ms, t_send_unix,
                ttfb_ms, ttft_ms, e2e_ms, status, ok, error, state,
                intended, chars_sent, retries,
                interchunk_max_ms=None,
                ttfr_ms=None, ttfv_ms=None, connect_ms=None,
                first_send_unix=None, max_tokens_requested=None, *,
                first_attempt_unix=None, connection_attempts=0,
                request_attempts=0, retry_reasons=None,
                ttf_tool_call_ms=None, scheduled_monotonic=None,
                queue_wait_ms=None, caller_ttfb_ms=None,
                caller_ttft_ms=None, caller_ttfr_ms=None,
                caller_ttfv_ms=None, caller_ttf_tool_call_ms=None,
                caller_send_ms=None,
                quota_guard_id=None, quota_guard_denied=False,
                quota_guard_events=None,
                worker_started_unix=None, worker_started_monotonic=None
                ) -> RequestResult:
        finished_monotonic = time.monotonic()
        finished_unix = (
            worker_started_unix
            + max(finished_monotonic - worker_started_monotonic, 0.0)
            if worker_started_unix is not None
            and worker_started_monotonic is not None
            else time.time())
        u = extract_usage(state.usage)
        stream_complete = bool(state.done or state.finish_reason)
        if ok and state.errors:
            ok = False
            error = error or "stream protocol validation failed"
        if ok and not stream_complete:
            ok = False
            error = error or (
                "stream ended without [DONE] or a finish_reason")
        distinct_parse_errors = list(dict.fromkeys(
            str(item)[:240] for item in state.errors))
        parse_error_details = distinct_parse_errors[:16]
        if len(distinct_parse_errors) > 16:
            parse_error_details.append(
                f"{len(distinct_parse_errors) - 16} additional distinct "
                "stream validation error(s) omitted")
        return RequestResult(
            request_id=request_id, scheduled_s=scheduled_s,
            dispatch_lag_ms=dispatch_lag_ms, t_send_unix=t_send_unix,
            ttfb_ms=ttfb_ms, ttft_ms=ttft_ms, ttfr_ms=ttfr_ms,
            ttfv_ms=ttfv_ms, e2e_ms=e2e_ms, status=status,
            ok=ok, error=error, content_chunks=state.content_chunks,
            stream_complete=stream_complete,
            visible_content_seen=bool(state.saw_first_visible),
            reasoning_seen=bool(state.saw_first_reasoning),
            truncated=(state.finish_reason == "length"),
            parse_errors=len(state.errors),
            parse_error_details=parse_error_details,
            max_tokens_requested=max_tokens_requested,
            interchunk_max_ms=interchunk_max_ms,
            finish_reason=state.finish_reason,
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
            cached_tokens=u["cached_tokens"],
            cached_tokens_source=u["cached_tokens_source"],
            intended_input_tokens=intended[0],
            intended_output_tokens=intended[1],
            intended_cache_fraction=intended[2],
            doc_id=intended[3] if len(intended) > 3 else -1,
            chars_sent=chars_sent, retries=retries,
            service_tier=state.service_tier,
            reasoning_tokens=u["reasoning_tokens"],
            reasoning_tokens_source=u["reasoning_tokens_source"],
            reasoning_chunks=state.reasoning_chunks,
            connect_ms=connect_ms,
            first_send_unix=first_send_unix,
            first_attempt_unix=first_attempt_unix,
            connection_attempts=connection_attempts,
            request_attempts=request_attempts,
            retry_reasons=list(retry_reasons or []),
            tool_call_seen=bool(state.saw_first_tool_call),
            tool_call_chunks=state.tool_call_chunks,
            ttf_tool_call_ms=ttf_tool_call_ms,
            valid_tool_calls=state.valid_tool_calls,
            refusal_seen=state.saw_refusal,
            refusal_chunks=state.refusal_chunks,
            response_content_type=state.response_content_type,
            served_model_name=state.served_model_name,
            response_model=state.response_model,
            response_object=state.response_object,
            response_id_sha256=(
                hashlib.sha256(state.response_id.encode("utf-8")).hexdigest()
                if state.response_id is not None else None),
            system_fingerprint=state.system_fingerprint,
            quota_guard_id=quota_guard_id,
            quota_guard_denied=bool(quota_guard_denied),
            quota_guard_events=copy.deepcopy(quota_guard_events or []),
            queue_wait_ms=queue_wait_ms,
            caller_ttfb_ms=caller_ttfb_ms,
            caller_ttft_ms=caller_ttft_ms,
            caller_ttfr_ms=caller_ttfr_ms,
            caller_ttfv_ms=caller_ttfv_ms,
            caller_ttf_tool_call_ms=caller_ttf_tool_call_ms,
            caller_send_ms=caller_send_ms,
            caller_e2e_ms=(
                max((finished_monotonic - scheduled_monotonic) * 1000.0, 0.0)
                if scheduled_monotonic is not None else None),
            finished_unix=finished_unix,
        )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
