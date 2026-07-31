"""Blocking streaming client for OpenAI-compatible chat completions.

Standard library only (http.client), one connection per request, precise
monotonic timing. Concurrency is provided by the runner's thread pool; a
blocked socket read releases the GIL, so hundreds of in-flight requests are
fine, and the runner MEASURES client-side dispatch lag rather than assuming
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
import json
import ssl
import time
import urllib.parse
import uuid
from dataclasses import dataclass, asdict

from .sse import StreamState, parse_sse_line, update_state, extract_usage


@dataclass
class EndpointConfig:
    base_url: str                    # e.g. https://<workspace-host>
    path: str                        # e.g. /serving-endpoints/<name>/invocations
    auth_token_env: str = "DATABRICKS_TOKEN"
    model: str | None = None         # set for shared /chat/completions routes
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    temperature: float = 0.0
    max_retries: int = 1             # connection-level errors only


@dataclass
class RequestResult:
    request_id: str
    scheduled_s: float
    dispatch_lag_ms: float           # how late the client fired vs schedule
    t_send_unix: float
    ttfb_ms: float | None
    ttft_ms: float | None
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
    intended_cache_fraction: float
    doc_id: int                      # pooled document; -1 = no shared prefix
    chars_sent: int
    retries: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class EndpointClient:
    def __init__(self, cfg: EndpointConfig, token: str | None):
        self.cfg = cfg
        self.token = token
        u = urllib.parse.urlparse(cfg.base_url)
        self.scheme = u.scheme or "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.scheme == "https" else 80)
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
        payload: dict = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": self.cfg.temperature,
            "stream": True,
        }
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

        while attempt <= self.cfg.max_retries:
            attempt += 1
            conn = None
            try:
                conn = self._connect()
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Request-Id": request_id,
                }
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"

                body = self._body(messages, max_tokens, include_usage)
                t_send = time.monotonic()
                t_send_unix = time.time()
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
                    attempt -= 1
                    continue

                if resp.status != 200:
                    detail = resp.read(2048).decode("utf-8", "replace")
                    return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                                        t_send_unix, None, None, None,
                                        resp.status, False,
                                        f"http {resp.status}: {detail[:300]}",
                                        StreamState(), intended, chars_sent,
                                        attempt - 1)

                if include_usage and self._include_usage_supported is None:
                    self._include_usage_supported = True

                state = StreamState()
                ttfb_ms = ttft_ms = None
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
                    first = update_state(state, event)
                    if first and ttft_ms is None:
                        ttft_ms = (now - t_send) * 1000.0
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
                                    attempt - 1, interchunk_max)

            except (OSError, http.client.HTTPException) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                if conn is not None:
                    conn.close()

        return self._finish(request_id, scheduled_s, dispatch_lag_ms,
                            time.time(), None, None, None, None, False,
                            last_err or "exhausted retries", StreamState(),
                            intended, chars_sent, attempt - 1)

    @staticmethod
    def _finish(request_id, scheduled_s, dispatch_lag_ms, t_send_unix,
                ttfb_ms, ttft_ms, e2e_ms, status, ok, error, state,
                intended, chars_sent, retries,
                interchunk_max_ms=None) -> RequestResult:
        u = extract_usage(state.usage)
        return RequestResult(
            request_id=request_id, scheduled_s=scheduled_s,
            dispatch_lag_ms=dispatch_lag_ms, t_send_unix=t_send_unix,
            ttfb_ms=ttfb_ms, ttft_ms=ttft_ms, e2e_ms=e2e_ms, status=status,
            ok=ok, error=error, content_chunks=state.content_chunks,
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
        )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
