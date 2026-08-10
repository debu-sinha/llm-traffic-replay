"""Instrumented mock endpoint with a KNOWN latency model.

Purpose: validate the measurement path before pointing the harness at
anything real. The mock speaks OpenAI-compatible streaming chat completions
and, per request:

  * simulates a block-level prefix cache over the system message text
    (leading 256-character blocks, about 64 mock tokens, LRU capacity, TTL),
    so the pool's constructed
    cache structure is exercised end to end through real text;
  * sleeps a deterministic, parameterized latency:
        ttft_true_ms = ttft_base_ms
                     + ms_per_1k_uncached * (uncached_prompt_tokens / 1000)
        then per_token_ms between completion chunks;
  * reports usage with prompt_tokens, completion_tokens and
    prompt_tokens_details.cached_tokens at the mock's exact 4.0 chars/token;
  * appends its own server-side truth (actual sleeps, token counts) to a
    JSONL log keyed by X-Request-Id.

`python -m traffic_replay validate` runs the full pipeline against this
server and reports instrument error = client-measured minus server-truth.
"""
from __future__ import annotations

import json
import hashlib
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .json_input import loads_strict

MOCK_CPT = 4.0
BLOCK_CHARS = 256  # ~64 tokens per cache block, realistic page granularity

DEFAULTS = {
    "ttft_base_ms": 120.0,
    "ms_per_1k_uncached": 40.0,
    "per_token_ms": 4.0,
    "reasoning_tokens": 0,
    # emit the reasoning channel and then stop on "length" without ever
    # sending a visible delta. that is what a reasoning model does when the
    # token budget runs out mid-thought, and it is the shape that used to be
    # counted as a success.
    "reasoning_only": 0,
    "cache_capacity_chains": 4096,
    "cache_ttl_s": 900.0,
}


class _PrefixCache:
    """Chain-hash prefix cache: an entry per (doc-leading-blocks) chain."""

    def __init__(self, capacity: int, ttl_s: float):
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.store: OrderedDict[bytes, float] = OrderedDict()
        self.lock = threading.Lock()

    def match_and_insert(self, text: str) -> int:
        """Return matched leading chars already cached, then cache this text's
        chains. Thread-safe; called once per request."""
        now = time.monotonic()
        chains = []
        chain = b""
        n_full = len(text) // BLOCK_CHARS
        for i in range(n_full):
            block = text[i * BLOCK_CHARS:(i + 1) * BLOCK_CHARS]
            # Built-in hash() is salted per process, which made the validator
            # oracle change across interpreter launches. A content digest is
            # stable and models a chain-keyed prefix cache just as well.
            chain = hashlib.sha256(chain + block.encode("utf-8")).digest()
            chains.append(chain)
        matched_blocks = 0
        with self.lock:
            # expire
            while self.store:
                k, ts = next(iter(self.store.items()))
                if now - ts > self.ttl_s:
                    self.store.popitem(last=False)
                else:
                    break
            for i, ch in enumerate(chains):
                if ch in self.store:
                    matched_blocks = i + 1
                    self.store.move_to_end(ch)
                    self.store[ch] = now
                else:
                    break
            for ch in chains:
                self.store[ch] = now
                self.store.move_to_end(ch)
            while len(self.store) > self.capacity:
                self.store.popitem(last=False)
        return matched_blocks * BLOCK_CHARS


def make_handler(params: dict, cache: _PrefixCache, truth_path: Path,
                 truth_lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            t_recv = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("invalid content length")
                payload = loads_strict(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                msgs = payload.get("messages")
                if not isinstance(msgs, list) or not msgs \
                        or any(not isinstance(message, dict)
                               for message in msgs):
                    raise ValueError("messages must be a non-empty array")
                if any(not isinstance(message.get("role"), str)
                       or not isinstance(message.get("content"), str)
                       for message in msgs):
                    raise ValueError("mock messages need string role/content")
                max_tokens = payload.get("max_tokens", 32)
                if not isinstance(max_tokens, int) \
                        or isinstance(max_tokens, bool) or max_tokens <= 0:
                    raise ValueError("max_tokens must be a positive integer")
            except (OSError, TypeError, ValueError):
                self.send_error(400, "bad json")
                return

            rid = self.headers.get("X-Request-Id", "unknown")
            system_text = "".join(m.get("content", "") for m in msgs
                                  if m.get("role") == "system")
            all_text = "".join(m.get("content", "") for m in msgs)

            matched_chars = cache.match_and_insert(system_text) \
                if system_text else 0
            prompt_tokens = max(int(round(len(all_text) / MOCK_CPT)), 1)
            cached_tokens = min(int(round(matched_chars / MOCK_CPT)),
                                prompt_tokens)
            uncached = prompt_tokens - cached_tokens
            ttft_planned_ms = (params["ttft_base_ms"]
                               + params["ms_per_1k_uncached"] * uncached / 1000.0)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(obj: dict):
                data = f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"
                b = data.encode()
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            # role-only first chunk BEFORE the latency sleep, like real
            # servers that ack the stream early. TTFT must key on content,
            # not first byte; this is the trap the client must not fall into.
            emit({"choices": [{"delta": {"role": "assistant"},
                               "finish_reason": None}]})

            time.sleep(ttft_planned_ms / 1000.0)
            configured_reasoning = max(
                int(params.get("reasoning_tokens", 0)), 0)
            reasoning_only = bool(params.get("reasoning_only", 0))
            # max_tokens is a cap on all generated tokens, including hidden
            # reasoning. Preserve one visible token in ordinary mode; the
            # explicit reasoning-only mode is allowed to consume the cap.
            reasoning_n = min(
                configured_reasoning,
                max_tokens if reasoning_only else max(max_tokens - 1, 0))
            visible_n = 0 if reasoning_only else max_tokens - reasoning_n
            completion_tokens = reasoning_n + visible_n
            t_first_generated = None
            t_first_visible = None
            for i in range(reasoning_n):
                if i:
                    time.sleep(params["per_token_ms"] / 1000.0)
                if t_first_generated is None:
                    t_first_generated = time.monotonic()
                emit({"choices": [{"delta": {"reasoning_content": "hmm"},
                                   "finish_reason": None}]})
            if reasoning_n:
                time.sleep(params["per_token_ms"] / 1000.0)
            if reasoning_only:
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": reasoning_n,
                    "total_tokens": prompt_tokens + reasoning_n,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    "completion_tokens_details": {
                        "reasoning_tokens": reasoning_n},
                }
                emit({"choices": [{"delta": {}, "finish_reason": "length"}],
                      "usage": usage})
            else:
                t_first_visible = time.monotonic()
                if t_first_generated is None:
                    t_first_generated = t_first_visible
                emit({"choices": [{"delta": {"content": "The"},
                                   "finish_reason": None}]})
                for _ in range(visible_n - 1):
                    time.sleep(params["per_token_ms"] / 1000.0)
                    emit({"choices": [{"delta": {"content": " next"},
                                       "finish_reason": None}]})
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                }
                if reasoning_n:
                    usage["completion_tokens_details"] = {
                        "reasoning_tokens": reasoning_n}
                emit({"choices": [{"delta": {}, "finish_reason": "stop"}],
                      "usage": usage})
            t_done = time.monotonic()
            truth = {
                "request_id": rid,
                "ttft_true_ms": (
                    (t_first_generated - t_recv) * 1000.0
                    if t_first_generated is not None else None),
                "ttfr_true_ms": (
                    (t_first_generated - t_recv) * 1000.0
                    if reasoning_n and t_first_generated is not None else None),
                "ttfv_true_ms": (
                    (t_first_visible - t_recv) * 1000.0
                    if t_first_visible is not None else None),
                "e2e_true_ms": (t_done - t_recv) * 1000.0,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
            }
            with truth_lock:
                with truth_path.open("a") as f:
                    f.write(json.dumps(truth, separators=(",", ":")) + "\n")

            # Persist the oracle before telling the client the event stream is
            # done. Tests and validators may read it as soon as the client
            # returns; emitting [DONE] first created a real write-after-read
            # race on the reasoning-only path.
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def serve(port: int, truth_log: str | Path, **overrides) -> ThreadingHTTPServer:
    params = {**DEFAULTS, **overrides}
    truth_path = Path(truth_log)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text("")
    cache = _PrefixCache(params["cache_capacity_chains"], params["cache_ttl_s"])
    handler = make_handler(params, cache, truth_path, threading.Lock())
    class _QuietServer(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # client hangs up during shutdown etc.; not worth a traceback
            pass

    srv = _QuietServer(("127.0.0.1", port), handler)
    return srv


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="instrumented mock endpoint")
    ap.add_argument("--port", type=int, default=8808)
    ap.add_argument("--truth-log", default="results/mock_truth.jsonl")
    args = ap.parse_args()
    srv = serve(args.port, args.truth_log)
    print(f"mock listening on 127.0.0.1:{args.port}, "
          f"truth -> {args.truth_log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
