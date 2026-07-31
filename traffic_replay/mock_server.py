"""Instrumented mock endpoint with a KNOWN latency model.

Purpose: validate the measurement path before pointing the harness at
anything real. The mock speaks OpenAI-compatible streaming chat completions
and, per request:

  * simulates a block-level prefix cache over the system message text
    (leading 1 KiB blocks, LRU capacity, TTL), so the pool's constructed
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
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MOCK_CPT = 4.0
BLOCK_CHARS = 256  # ~64 tokens per cache block, realistic page granularity

DEFAULTS = {
    "ttft_base_ms": 120.0,
    "ms_per_1k_uncached": 40.0,
    "per_token_ms": 4.0,
    "reasoning_tokens": 0,
    "cache_capacity_chains": 4096,
    "cache_ttl_s": 900.0,
}


class _PrefixCache:
    """Chain-hash prefix cache: an entry per (doc-leading-blocks) chain."""

    def __init__(self, capacity: int, ttl_s: float):
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.store: OrderedDict[int, float] = OrderedDict()
        self.lock = threading.Lock()

    def match_and_insert(self, text: str) -> int:
        """Return matched leading chars already cached, then cache this text's
        chains. Thread-safe; called once per request."""
        now = time.monotonic()
        chains = []
        h = 0
        n_full = len(text) // BLOCK_CHARS
        for i in range(n_full):
            block = text[i * BLOCK_CHARS:(i + 1) * BLOCK_CHARS]
            h = hash((h, block))
            chains.append(h)
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
                payload = json.loads(self.rfile.read(length))
            except Exception:
                self.send_error(400, "bad json")
                return

            rid = self.headers.get("X-Request-Id", "unknown")
            msgs = payload.get("messages") or []
            system_text = "".join(m.get("content", "") for m in msgs
                                  if m.get("role") == "system")
            all_text = "".join(m.get("content", "") for m in msgs)
            max_tokens = int(payload.get("max_tokens", 32))

            matched_chars = cache.match_and_insert(system_text) \
                if system_text else 0
            prompt_tokens = max(int(round(len(all_text) / MOCK_CPT)), 1)
            cached_tokens = min(int(round(matched_chars / MOCK_CPT)),
                                prompt_tokens)
            uncached = prompt_tokens - cached_tokens
            completion_tokens = max_tokens

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
            reasoning_n = int(params.get("reasoning_tokens", 0))
            for i in range(reasoning_n):
                if i:
                    time.sleep(params["per_token_ms"] / 1000.0)
                emit({"choices": [{"delta": {"reasoning_content": "hmm"},
                                   "finish_reason": None}]})
            if reasoning_n:
                time.sleep(params["per_token_ms"] / 1000.0)
            t_first_content = time.monotonic()
            emit({"choices": [{"delta": {"content": "The"},
                               "finish_reason": None}]})
            for _ in range(completion_tokens - 1):
                time.sleep(params["per_token_ms"] / 1000.0)
                emit({"choices": [{"delta": {"content": " next"},
                                   "finish_reason": None}]})
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            }
            emit({"choices": [{"delta": {}, "finish_reason": "stop"}],
                  "usage": usage})
            t_done = time.monotonic()
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

            truth = {
                "request_id": rid,
                "ttft_true_ms": (t_first_content - t_recv) * 1000.0,
                "e2e_true_ms": (t_done - t_recv) * 1000.0,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
            }
            with truth_lock:
                with truth_path.open("a") as f:
                    f.write(json.dumps(truth, separators=(",", ":")) + "\n")

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
