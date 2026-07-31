"""Load real prompts for verbatim replay (prompts mode).

Some users do not have a statistical profile, they have the actual prompts
they test with. In prompts mode each of those prompts becomes a request,
replayed as-is. The harness measures the endpoint on the real text instead
of on synthetic text shaped to a profile.

Accepted inputs, by file extension:

  .jsonl : one JSON value per line, any of
             {"messages": [{"role": "user", "content": "..."}, ...]}
             {"prompt": "..."}        single user message
             {"text": "..."}          single user message
             "a bare json string"     single user message
  .txt   : one prompt per line, each a single user message (blanks skipped)
  .json  : a JSON array whose items use any of the per-line shapes above

Returns a list of message-lists, each ready to POST to a chat endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path


def _coerce(item) -> list[dict]:
    """Turn one loaded item into a chat messages list.

    Content must be a string. This harness replays text prompts, so a null
    or multimodal (list-of-parts) content fails at load with a line number
    rather than mis-counting sizes or crashing mid-run.
    """
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if isinstance(item, dict):
        if "messages" in item:
            msgs = item["messages"]
            if not isinstance(msgs, list) or not msgs:
                raise ValueError("'messages' must be a non-empty list")
            for m in msgs:
                if not (isinstance(m, dict)
                        and isinstance(m.get("role"), str)
                        and isinstance(m.get("content"), str)):
                    raise ValueError(
                        "each message needs a string 'role' and 'content'")
            return msgs
        # a single message given inline, with its role preserved
        if isinstance(item.get("role"), str) \
                and isinstance(item.get("content"), str):
            return [{"role": item["role"], "content": item["content"]}]
        for key in ("prompt", "text"):
            if isinstance(item.get(key), str):
                return [{"role": "user", "content": item[key]}]
        raise ValueError(
            "prompt object needs 'messages', 'prompt', 'text', or an inline "
            "role + string content")
    raise ValueError(f"unsupported prompt item type: {type(item).__name__}")


def load_prompts(path: str) -> list[list[dict]]:
    """Read a prompts file into a list of chat messages lists."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"prompts file not found: {path}")
    raw = p.read_text()
    prompts: list[list[dict]] = []
    if p.suffix == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(".json prompts file must be a JSON array")
        for item in data:
            prompts.append(_coerce(item))
    elif p.suffix == ".txt":
        for line in raw.splitlines():
            line = line.strip()
            if line:
                prompts.append([{"role": "user", "content": line}])
    else:  # .jsonl and anything else: one json value per line
        for ln, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {ln}: not valid JSON ({e})") from e
            prompts.append(_coerce(item))
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts
