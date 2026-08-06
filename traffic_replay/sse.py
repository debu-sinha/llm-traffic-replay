"""Minimal, dependency-free Server-Sent Events parsing for OpenAI-style
streaming chat completions.

The client feeds raw lines; this module yields parsed events and extracts
the fields the harness measures: first content token, usage block, finish.
Kept separate from the HTTP layer so it is unit-testable against fixtures.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass
class StreamState:
    saw_first_content: bool = False
    saw_first_visible: bool = False       # first visible content delta
    saw_first_reasoning: bool = False     # first reasoning-channel delta
    saw_first_tool_call: bool = False     # first tool/function-call delta
    content_chunks: int = 0
    reasoning_chunks: int = 0             # count of reasoning-channel deltas
    tool_call_chunks: int = 0
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
        event = json.loads(payload)
    except json.JSONDecodeError:
        return {"__parse_error__": payload[:200]}
    if not isinstance(event, dict):
        return {"__parse_error__":
                f"SSE data must be a JSON object, got "
                f"{type(event).__name__}: {payload[:160]}"}
    return event


def iter_sse_events(lines: Iterable[bytes | str]) -> Iterator[dict]:
    """Yield complete SSE ``data`` events from an iterable of raw lines.

    SSE permits an event to contain multiple ``data:`` fields. Their values
    are joined with newlines and dispatched by a blank line. OpenAI-compatible
    servers normally use one data field per event, but treating each physical
    line as a complete event corrupts otherwise valid multiline streams.

    Non-data fields and comments are ignored. A final unterminated event is
    dispatched at EOF, which is useful for defensive interoperability with
    servers that omit the last blank line.
    """
    data: list[str] = []

    def dispatch() -> dict | None:
        if not data:
            return None
        payload = "\n".join(data)
        data.clear()
        if payload.strip() == "[DONE]":
            return {"__done__": True}
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return {"__parse_error__": payload[:200]}
        if not isinstance(event, dict):
            return {"__parse_error__":
                    f"SSE data must be a JSON object, got "
                    f"{type(event).__name__}: {payload[:160]}"}
        return event

    for raw in lines:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace")
        else:
            line = raw
        line = line.rstrip("\r\n")
        if not line:
            event = dispatch()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data.append(value)

    event = dispatch()
    if event is not None:
        yield event


def _meaningful_text(value: object) -> bool:
    """Whether a provider content value contains user-visible text."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        for part in value:
            if isinstance(part, str) and part.strip():
                return True
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    return False


def _nonempty_delta(value: object) -> bool:
    """Whether a delta represents at least one emitted stream fragment."""
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, dict)):
        return bool(value)
    return False


def update_state(state: StreamState, event: object) -> bool:
    """Fold one event into state. Returns True if this event carries the
    FIRST content delta (the TTFT moment)."""
    if not isinstance(event, dict):
        state.errors.append(
            f"stream event must be an object, got {type(event).__name__}")
        return False
    if event.get("__done__"):
        state.done = True
        return False
    if "__parse_error__" in event:
        state.errors.append(event["__parse_error__"])
        return False

    choices = event.get("choices")
    if choices is None:
        choices = []
    elif not isinstance(choices, list):
        state.errors.append("stream event choices must be a list")
        choices = []

    first_content = False
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            state.errors.append(
                f"stream choice {index} must be an object, got "
                f"{type(choice).__name__}")
            continue
        delta = choice.get("delta")
        if delta is None:
            delta = {}
        elif not isinstance(delta, dict):
            state.errors.append(
                f"stream choice {index} delta must be an object")
            delta = {}
        visible = delta.get("content")
        reasoning = delta.get("reasoning_content")
        tool_call = delta.get("tool_calls") or delta.get("function_call")
        has_visible_delta = _nonempty_delta(visible)
        has_reasoning_delta = _nonempty_delta(reasoning)
        has_tool_call_delta = _nonempty_delta(tool_call)
        if has_visible_delta or has_reasoning_delta:
            state.content_chunks += 1
            if not state.saw_first_content:
                state.saw_first_content = True
                first_content = True
        if has_reasoning_delta:
            state.reasoning_chunks += 1
        if has_reasoning_delta and not state.saw_first_reasoning:
            state.saw_first_reasoning = True
        if _meaningful_text(visible) and not state.saw_first_visible:
            state.saw_first_visible = True
        if has_tool_call_delta:
            state.tool_call_chunks += 1
            state.saw_first_tool_call = True
        fr = choice.get("finish_reason")
        if isinstance(fr, str) and fr:
            state.finish_reason = fr

    usage = event.get("usage")
    if usage is not None:
        if isinstance(usage, dict):
            state.usage = usage
        else:
            state.errors.append("stream event usage must be an object")
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
        if (ok and isinstance(node, (int, float))
                and not isinstance(node, bool)
                and math.isfinite(float(node)) and node >= 0):
            return int(node), ".".join(path)
    return None, None


def _token_count(value: object) -> int | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and value >= 0):
        return int(value)
    return None


def extract_usage(usage: dict | None) -> dict:
    """Normalize a usage block. Absent fields come back None, never guessed."""
    if not isinstance(usage, dict) or not usage:
        return {"prompt_tokens": None, "completion_tokens": None,
                "cached_tokens": None, "cached_tokens_source": None,
                "reasoning_tokens": None, "reasoning_tokens_source": None}
    cached, cached_src = _walk(usage, CACHED_TOKEN_PATHS)
    reasoning, reasoning_src = _walk(usage, REASONING_TOKEN_PATHS)
    return {
        "prompt_tokens": _token_count(usage.get("prompt_tokens")),
        "completion_tokens": _token_count(usage.get("completion_tokens")),
        "cached_tokens": cached,
        "cached_tokens_source": cached_src,
        "reasoning_tokens": reasoning,
        "reasoning_tokens_source": reasoning_src,
    }
