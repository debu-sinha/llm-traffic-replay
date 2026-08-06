"""Minimal, dependency-free Server-Sent Events parsing for OpenAI-style
streaming chat completions.

The client feeds raw lines; this module yields parsed events and extracts
the fields the harness measures: first content token, usage block, finish.
Kept separate from the HTTP layer so it is unit-testable against fixtures.
"""
from __future__ import annotations

import codecs
import hashlib
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


def _safe_parse_error(kind: str, payload: str) -> dict:
    """Return diagnostic metadata without persisting streamed content."""
    encoded = payload.encode("utf-8", "replace")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return {"__parse_error__":
            f"{kind} (payload bytes={len(encoded)}, sha256={digest})"}


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
        return _safe_parse_error("invalid SSE JSON", payload)
    if not isinstance(event, dict):
        return _safe_parse_error(
            f"SSE data must be a JSON object, got {type(event).__name__}",
            payload)
    return event


def iter_sse_events(lines: Iterable[bytes | str],
                    max_event_chars: int = 4 * 1024 * 1024
                    ) -> Iterator[dict]:
    """Yield complete SSE ``data`` events from an iterable of raw lines.

    SSE permits an event to contain multiple ``data:`` fields. Their values
    are joined with newlines and dispatched by a blank line. OpenAI-compatible
    servers normally use one data field per event, but treating each physical
    line as a complete event corrupts otherwise valid multiline streams.

    Non-data fields and comments are ignored. A final unterminated event is
    dispatched at EOF, which is useful for defensive interoperability with
    servers that omit the last blank line.
    """
    if not isinstance(max_event_chars, int) or isinstance(max_event_chars, bool) \
            or max_event_chars <= 0:
        raise ValueError("max_event_chars must be a positive integer")

    data: list[str] = []
    data_chars = 0
    discard_event = False
    buffered = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    at_stream_start = True

    def dispatch() -> dict | None:
        nonlocal data_chars, discard_event
        if discard_event:
            data.clear()
            data_chars = 0
            discard_event = False
            return None
        if not data:
            return None
        payload = "\n".join(data)
        data.clear()
        data_chars = 0
        if len(payload) > max_event_chars:
            return {"__parse_error__":
                    f"SSE event exceeded {max_event_chars} characters"}
        if payload.strip() == "[DONE]":
            return {"__done__": True}
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return _safe_parse_error("invalid SSE JSON", payload)
        if not isinstance(event, dict):
            return _safe_parse_error(
                f"SSE data must be a JSON object, got {type(event).__name__}",
                payload)
        return event

    def consume_line(line: str) -> dict | None:
        nonlocal data_chars, discard_event
        if not line:
            return dispatch()
        if discard_event:
            return None
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if field != "data":
            return None
        if separator and value.startswith(" "):
            value = value[1:]
        added = len(value) + (1 if data else 0)
        if data_chars + added > max_event_chars:
            data.clear()
            data_chars = 0
            discard_event = True
            return {"__parse_error__":
                    f"SSE event exceeded {max_event_chars} characters"}
        data.append(value)
        data_chars += added
        return None

    def decoded_chunks() -> Iterator[str]:
        """Decode bytes incrementally so UTF-8 code points may cross chunks."""
        nonlocal decoder
        for raw in lines:
            if isinstance(raw, bytes):
                yield decoder.decode(raw, final=False)
            elif isinstance(raw, str):
                # Mixed byte/string streams are unusual, but flushing pending
                # byte state avoids joining half a code point to native text.
                pending = decoder.decode(b"", final=True)
                decoder = codecs.getincrementaldecoder("utf-8")(
                    errors="replace")
                yield pending + raw
            else:
                raise TypeError("SSE chunks must be bytes or strings")
        yield decoder.decode(b"", final=True)

    for text in decoded_chunks():
        if at_stream_start and text:
            text = text.removeprefix("\ufeff")
            at_stream_start = False
        buffered += text
        while True:
            lf = buffered.find("\n")
            cr = buffered.find("\r")
            indexes = [x for x in (lf, cr) if x >= 0]
            if not indexes:
                break
            end = min(indexes)
            # A terminal CR might be the first half of CRLF in the next
            # network chunk. Waiting preserves one logical blank separator.
            if buffered[end] == "\r" and end + 1 == len(buffered):
                break
            separator_len = (2 if buffered[end:end + 2] == "\r\n" else 1)
            line = buffered[:end]
            buffered = buffered[end + separator_len:]
            event = consume_line(line)
            if event is not None:
                yield event
        if len(buffered) > max_event_chars:
            yield {"__parse_error__":
                   f"SSE line exceeded {max_event_chars} characters"}
            buffered = ""
            data.clear()
            data_chars = 0
            discard_event = True

    if buffered:
        if buffered.endswith("\r"):
            buffered = buffered[:-1]
        event = consume_line(buffered)
        if event is not None:
            yield event
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
    if isinstance(value, dict):
        text = value.get("text")
        return isinstance(text, str) and bool(text.strip())
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
        if tool_call is not None and not isinstance(tool_call, (list, dict)):
            state.errors.append(
                f"stream choice {index} tool call must be an object or list")
            has_tool_call_delta = False
        else:
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
        elif fr is not None:
            state.errors.append(
                f"stream choice {index} finish_reason must be a string")

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
        parsed = _token_count(node) if ok else None
        if parsed is not None:
            return parsed, ".".join(path)
    return None, None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) \
            and value >= 0 and value.is_integer():
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
