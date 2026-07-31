"""Minimal, dependency-free Server-Sent Events parsing for OpenAI-style
streaming chat completions.

The client feeds raw lines; this module yields parsed events and extracts
the fields the harness measures: first content token, usage block, finish.
Kept separate from the HTTP layer so it is unit-testable against fixtures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class StreamState:
    saw_first_content: bool = False
    saw_first_visible: bool = False       # first visible content delta
    saw_first_reasoning: bool = False     # first reasoning-channel delta
    content_chunks: int = 0
    finish_reason: str | None = None
    usage: dict | None = None
    done: bool = False
    errors: list[str] = field(default_factory=list)


def parse_sse_line(line: bytes | str) -> dict | None:
    """Return the JSON payload of a `data:` line, {'__done__': True} for
    [DONE], or None for blanks/comments/other fields."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"__parse_error__": payload[:200]}


def update_state(state: StreamState, event: dict) -> bool:
    """Fold one event into state. Returns True if this event carries the
    FIRST content delta (the TTFT moment)."""
    if event.get("__done__"):
        state.done = True
        return False
    if "__parse_error__" in event:
        state.errors.append(event["__parse_error__"])
        return False

    first_content = False
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        visible = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if visible or reasoning:
            state.content_chunks += 1
            if not state.saw_first_content:
                state.saw_first_content = True
                first_content = True
        if reasoning and not state.saw_first_reasoning:
            state.saw_first_reasoning = True
        if visible and not state.saw_first_visible:
            state.saw_first_visible = True
        fr = choice.get("finish_reason")
        if fr:
            state.finish_reason = fr

    if event.get("usage"):
        state.usage = event["usage"]
    return first_content


# Known field paths for cached prompt tokens across providers. Checked in
# order; the first present wins. The report records WHICH path was found.
CACHED_TOKEN_PATHS = (
    ("prompt_tokens_details", "cached_tokens"),   # OpenAI-style
    ("prompt_cache_hit_tokens",),                 # DeepSeek-style
    ("cached_tokens",),                           # flat variants
    ("cache_read_input_tokens",),                 # Anthropic-style naming
)

# Reasoning (thinking) token counts, same convention.
REASONING_TOKEN_PATHS = (
    ("completion_tokens_details", "reasoning_tokens"),   # OpenAI o-series
    ("reasoning_tokens",),                               # flat variants
)


def _walk(usage: dict, paths) -> tuple[int | None, str | None]:
    """First present integer at any of `paths`, with its dotted source."""
    for path in paths:
        node = usage
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node and node[key] is not None:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            return int(node), ".".join(path)
    return None, None


def extract_usage(usage: dict | None) -> dict:
    """Normalize a usage block. Absent fields come back None, never guessed."""
    if not usage:
        return {"prompt_tokens": None, "completion_tokens": None,
                "cached_tokens": None, "cached_tokens_source": None,
                "reasoning_tokens": None, "reasoning_tokens_source": None}
    cached, cached_src = _walk(usage, CACHED_TOKEN_PATHS)
    reasoning, reasoning_src = _walk(usage, REASONING_TOKEN_PATHS)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached,
        "cached_tokens_source": cached_src,
        "reasoning_tokens": reasoning,
        "reasoning_tokens_source": reasoning_src,
    }
