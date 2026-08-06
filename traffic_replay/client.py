"""Blocking streaming client for OpenAI-compatible chat completions.

Standard library only (http.client), one connection per request, precise
monotonic timing. Concurrency is provided by the runner's thread pool; a
blocked socket read releases the GIL, so hundreds of in-flight requests are
fine, and the runner MEASURES client-side lateness rather than assuming
the client kept up (see runner.py / metrics.py).

Timing definitions, used consistently everywhere:
  t_send           just before the request is written to the socket
  ttfb_ms          first response line received (any SSE event)
  ttft_ms          first content delta received  <- the headline number
  e2e_ms           stream finished ([DONE] or final chunk)

Usage (prompt/completion/cached token counts) is read from the endpoint's
final usage block when present. stream_options.include_usage is requested
and automatically retried without it for endpoints that reject the field.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import ssl
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, asdict, field

from .sse import StreamState, parse_sse_line, update_state, extract_usage


@dataclass
class EndpointConfig:
    base_url: str                    # e.g. https://<workspace-host>
    path: str                        # e.g. /serving-endpoints/<name>/invocations
    auth_token_env: str = "DATABRICKS_TOKEN"
    auth_profile: str | None = None   # a ~/.databrickscfg profile name. takes
                                      # precedence over auth_token_env, and
                                      # handles OAuth profiles by asking the
                                      # Databricks CLI for a fresh token.
    model: str | None = None         # set for shared /chat/completions routes
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    temperature: float = 0.0
    max_retries: int = 1             # connection-level errors only
    extra_body: dict | None = None   # passthrough request params (see _body)


@dataclass
class RequestResult:
    request_id: str
    scheduled_s: float
    dispatch_lag_ms: float           # dispatcher lateness only. a full pool
                                     # queues, so this does NOT see client
                                     # saturation. metrics computes wire
                                     # lateness from first_send_unix.
    t_send_unix: float
    ttfb_ms: float | None
    ttft_ms: float | None            # first content of either kind (back compat)
    ttfr_ms: float | None            # first reasoning-channel delta, else None
    ttfv_ms: float | None            # first visible content delta, else None
    e2e_ms: float | None
    status: int | None
    ok: bool
    error: str | None
    content_chunks: int
    interchunk_max_ms: float | None   # widest gap between content chunks
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


_MAX_TOKEN_REFRESH = 5


class UnsafeBearerTransport(ValueError):
    """A bearer credential would cross an untrusted cleartext transport."""


def normalized_origin(value: str) -> tuple[str, str, int]:
    """Return a canonical HTTP(S) origin for credential binding.

    Host names are case-folded, IDNA-normalized, and stripped of a terminal
    dot. Explicit default ports compare equal to implicit ones. Userinfo is
    rejected because it makes security-sensitive URL review needlessly
    ambiguous (and is never needed for a serving endpoint).
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint base_url must be a non-empty URL")
    u = urllib.parse.urlsplit(value.strip())
    scheme = u.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("endpoint base_url must use an explicit http or https scheme")
    if u.username is not None or u.password is not None:
        raise ValueError("endpoint base_url must not contain userinfo")
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


class EndpointClient:
    def __init__(self, cfg: EndpointConfig, token: str | None,
                 refresh: "callable | None" = None):
        """`refresh` returns a fresh token, or None if it cannot.

        An OAuth token is minted once and a load test can outlive it. When
        it expires mid-run every remaining request comes back 401 or 403 and
        reads as an endpoint failure, which is both a wasted run and a
        misleading one. Measured for real: a 90 second run lost 171 of 281
        requests to `http 403: Invalid Token`.
        """
        self.cfg = cfg
        self.token = token
        self._refresh = refresh
        self._refreshed = 0
        self._lock = threading.Lock()
        self.scheme, self.host, self.port = normalized_origin(cfg.base_url)
        # A refresh callback means this is a bearer-auth flow even when the
        # initial token is absent or expired. Reject its transport before the
        # first unauthenticated probe rather than waiting until a token exists.
        if token or refresh is not None:
            validate_bearer_transport(cfg.base_url)
        self._ssl = ssl.create_default_context() if self.scheme == "https" else None
        self._include_usage_supported: bool | None = None  # learned

    def _connect(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.cfg.connect_timeout_s,
                context=self._ssl)
        return http.client.HTTPConnection(
            self.host, self.port, timeout=self.cfg.connect_timeout_s)

    def _body(self, messages: list[dict], max_tokens: int,
              include_usage: bool) -> bytes:
        # extra_body is user passthrough (top_p, stop, response_format, and
        # provider thinking control like reasoning_effort / thinking /
        # chat_template_kwargs). The harness owns the keys below: they are
        # popped first so nothing in extra_body can survive, then set from
        # their dedicated config, so a run stays measurable no matter what
        # the user put in extra_body.
        owned = ("messages", "max_tokens", "temperature", "stream",
                 "model", "stream_options")
        payload: dict = {k: v for k, v in (self.cfg.extra_body or {}).items()
                         if k not in owned}
        payload["messages"] = messages
        payload["max_tokens"] = int(max_tokens)
        payload["temperature"] = self.cfg.temperature
        payload["stream"] = True
        if self.cfg.model:
            payload["model"] = self.cfg.model
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
        return json.dumps(payload).encode()

    def send(self, messages: list[dict], max_tokens: int, request_id: str,
             scheduled_s: float, dispatch_lag_ms: float,
             intended: tuple[int, int, float, int],
             chars_sent: int) -> RequestResult:
        """One request, fully measured. Never raises; errors land in result."""
        attempt = 0
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

        while attempt <= self.cfg.max_retries:
            attempt += 1
            conn = None
            try:
                conn = self._connect()
                connection_attempts += 1
                if first_attempt_unix is None:
                    first_attempt_unix = time.time()
                t_conn0 = time.monotonic()
                conn.connect()
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
                    validate_bearer_transport(self.cfg.base_url)
                    headers["Authorization"] = f"Bearer {tok_used}"

                body = self._body(messages, max_tokens, include_usage)
                t_send = time.monotonic()
                t_send_unix = time.time()
                last_send_unix = t_send_unix
                if first_send_unix is None:
                    first_send_unix = t_send_unix
                request_attempts += 1
                conn.request("POST", self.cfg.path, body=body, headers=headers)
                conn.sock.settimeout(self.cfg.read_timeout_s)
                resp = conn.getresponse()

                if resp.status == 400 and include_usage \
                        and self._include_usage_supported is None:
                    # Endpoint may reject stream_options; learn and retry once
                    # without counting it against the retry budget.
                    resp.read()
                    self._include_usage_supported = False
                    include_usage = False
                    retry_reasons.append("stream_options_rejected")
                    attempt -= 1
                    continue

                if resp.status in (401, 403) and self._refresh:
                    detail = resp.read(2048).decode("utf-8", "replace")
                    # keep the real reason. falling out of the retry loop
                    # with "exhausted retries" hides an auth problem, which
                    # is the most common thing to get wrong.
                    last_err = f"http {resp.status}: {detail[:300]}"
                    conn.close()
                    # this is a concurrent load generator, so when a token
                    # expires MANY requests fail at once. each of them must
                    # get a retry against the new token, and only the first
                    # of them should spend a refresh. comparing against the
                    # token this request actually used, rather than against
                    # the shared one, is what makes that true: a thread that
                    # arrives after someone else refreshed simply retries.
                    with self._lock:
                        if self.token != tok_used:
                            retry_auth = True          # someone refreshed
                        elif self._refreshed < _MAX_TOKEN_REFRESH:
                            self._refreshed += 1
                            fresh = self._refresh()
                            if fresh and fresh != self.token:
                                try:
                                    validate_bearer_transport(self.cfg.base_url)
                                except UnsafeBearerTransport as exc:
                                    # Do not install the token: the next loop
                                    # must never get a chance to emit it.
                                    last_err = str(exc)
                                    retry_auth = False
                                else:
                                    self.token = fresh
                                    retry_auth = True
                            else:
                                retry_auth = False
                        else:
                            retry_auth = False
                    if retry_auth:
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
                        retry_reasons=retry_reasons)

                if resp.status != 200:
                    detail = resp.read(2048).decode("utf-8", "replace")
                    return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                                        t_send_unix, None, None, None,
                                        resp.status, False,
                                        f"http {resp.status}: {detail[:300]}",
                                        StreamState(), intended, chars_sent,
                                        len(retry_reasons), None, None, None,
                                        connect_ms, first_send_unix,
                                        max_tokens,
                                        first_attempt_unix=first_attempt_unix,
                                        connection_attempts=connection_attempts,
                                        request_attempts=request_attempts,
                                        retry_reasons=retry_reasons)

                if include_usage and self._include_usage_supported is None:
                    self._include_usage_supported = True

                state = StreamState()
                ttfb_ms = ttft_ms = ttfr_ms = ttfv_ms = None
                interchunk_max = None
                last_content_t = None
                for raw in resp:
                    now = time.monotonic()
                    if ttfb_ms is None:
                        ttfb_ms = (now - t_send) * 1000.0
                    event = parse_sse_line(raw)
                    if event is None:
                        continue
                    chunks_before = state.content_chunks
                    reasoning_before = state.saw_first_reasoning
                    visible_before = state.saw_first_visible
                    first = update_state(state, event)
                    if first and ttft_ms is None:
                        ttft_ms = (now - t_send) * 1000.0
                    if state.saw_first_reasoning and not reasoning_before:
                        ttfr_ms = (now - t_send) * 1000.0
                    if state.saw_first_visible and not visible_before:
                        ttfv_ms = (now - t_send) * 1000.0
                    if state.content_chunks > chunks_before:
                        if last_content_t is not None:
                            gap = (now - last_content_t) * 1000.0
                            if interchunk_max is None or gap > interchunk_max:
                                interchunk_max = gap
                        last_content_t = now
                    if state.done:
                        break
                e2e_ms = (time.monotonic() - t_send) * 1000.0
                ok = state.saw_first_content
                err = None if ok else "stream ended with no content delta"
                return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                                    t_send_unix, ttfb_ms, ttft_ms, e2e_ms,
                                    200, ok, err, state, intended, chars_sent,
                                    len(retry_reasons), interchunk_max,
                                    ttfr_ms, ttfv_ms, connect_ms,
                                    first_send_unix, max_tokens,
                                    first_attempt_unix=first_attempt_unix,
                                    connection_attempts=connection_attempts,
                                    request_attempts=request_attempts,
                                    retry_reasons=retry_reasons)

            except (OSError, http.client.HTTPException) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt <= self.cfg.max_retries:
                    retry_reasons.append("connection_error")
                continue
            finally:
                if conn is not None:
                    conn.close()

        return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                            last_send_unix if last_send_unix is not None
                            else (first_attempt_unix or time.time()),
                            None, None, None, None, False,
                            last_err or "exhausted retries", StreamState(),
                            intended, chars_sent, len(retry_reasons),
                            None, None, None, None, first_send_unix,
                            max_tokens,
                            first_attempt_unix=first_attempt_unix,
                            connection_attempts=connection_attempts,
                            request_attempts=request_attempts,
                            retry_reasons=retry_reasons)

    @staticmethod
    def _finish(request_id, scheduled_s, dispatch_lag_ms, t_send_unix,
                ttfb_ms, ttft_ms, e2e_ms, status, ok, error, state,
                intended, chars_sent, retries,
                interchunk_max_ms=None,
                ttfr_ms=None, ttfv_ms=None, connect_ms=None,
                first_send_unix=None, max_tokens_requested=None, *,
                first_attempt_unix=None, connection_attempts=0,
                request_attempts=0, retry_reasons=None
                ) -> RequestResult:
        u = extract_usage(state.usage)
        return RequestResult(
            request_id=request_id, scheduled_s=scheduled_s,
            dispatch_lag_ms=dispatch_lag_ms, t_send_unix=t_send_unix,
            ttfb_ms=ttfb_ms, ttft_ms=ttft_ms, ttfr_ms=ttfr_ms,
            ttfv_ms=ttfv_ms, e2e_ms=e2e_ms, status=status,
            ok=ok, error=error, content_chunks=state.content_chunks,
            stream_complete=bool(state.done or state.finish_reason),
            visible_content_seen=bool(state.saw_first_visible),
            reasoning_seen=bool(state.saw_first_reasoning),
            truncated=(state.finish_reason == "length"),
            parse_errors=len(state.errors),
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
            reasoning_tokens=u["reasoning_tokens"],
            reasoning_tokens_source=u["reasoning_tokens_source"],
            reasoning_chunks=state.reasoning_chunks,
            connect_ms=connect_ms,
            first_send_unix=first_send_unix,
            first_attempt_unix=first_attempt_unix,
            connection_attempts=connection_attempts,
            request_attempts=request_attempts,
            retry_reasons=list(retry_reasons or []),
        )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
