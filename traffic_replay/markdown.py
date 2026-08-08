"""Small, dependency-free Markdown trust-boundary helpers."""
from __future__ import annotations

import re

from .artifacts import sanitize_display_text


# These characters can create inline Markdown even when the value is embedded
# after trusted report text. Block-only markers (``#``, ``-``, ``+``) are safe
# because line breaks are collapsed before escaping.
_MARKDOWN_PUNCTUATION = frozenset("\\`*_[]()!~")
def markdown_plain_text(value: object) -> str:
    """Render an untrusted value as readable text, never Markdown structure.

    This is intentionally narrower than a general Markdown serializer. It is
    for customer-controlled labels and notes embedded in trusted report
    structure. Newlines are collapsed, HTML is entity-escaped, table pipes
    become entities, Markdown punctuation is backslash-escaped, and bidi/C0
    controls are removed.
    """
    text = sanitize_display_text(value)
    pieces: list[str] = []
    pending_space = False
    for char in text:
        codepoint = ord(char)
        if char.isspace() or codepoint < 0x20:
            pending_space = bool(pieces)
            continue
        if pending_space:
            pieces.append(" ")
            pending_space = False
        if char == "&":
            pieces.append("&amp;")
        elif char == "<":
            pieces.append("&lt;")
        elif char == ">":
            pieces.append("&gt;")
        elif char == "|":
            # An entity is more portable than ``\\|`` inside GFM tables.
            pieces.append("&#124;")
        elif char in _MARKDOWN_PUNCTUATION:
            pieces.append("\\" + char)
        else:
            pieces.append(char)
    return re.sub(r" +", " ", "".join(pieces)).strip()
