"""Unambiguous, standards-compliant JSON parsing for all trust boundaries."""
from __future__ import annotations

import hashlib
import json
import math
import re


class StrictJSONError(ValueError):
    """A JSON failure whose message is safe to include in diagnostics."""


_SAFE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_SECRETISH_KEY = re.compile(
    r"(?i)(?:bearer|dapi[0-9a-z._-]{8,}|sk-[0-9a-z._-]{8,}|"
    r"xox[baprs]-|github_pat_|authorization\s*[:=]|token\s*[:=])")


def _duplicate_key_label(key: str) -> str:
    """Describe ordinary schema keys, but hash payload-like key material."""
    encoded = key.encode("utf-8", "surrogatepass")
    if _SAFE_KEY.fullmatch(key) and not _SECRETISH_KEY.search(key):
        return repr(key)
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"<redacted; bytes={len(encoded)}, sha256={digest}>"


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StrictJSONError(
                f"JSON contains duplicate key {_duplicate_key_label(key)}")
        value[key] = item
    return value


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise StrictJSONError("JSON contains a non-finite number")
    return value


def _reject_nonfinite_constant(_raw: str):
    raise StrictJSONError("JSON contains a non-finite number")


def loads_strict(value: str | bytes):
    """Parse UTF-8 JSON without duplicate keys or non-finite numbers.

    Python's default decoder accepts JavaScript constants such as ``NaN`` and
    turns an overflowing exponent such as ``1e999`` into infinity. Both are
    outside JSON and make comparisons fail open. Bytes are decoded explicitly
    so UTF-16 auto-detection and replacement decoding cannot enter evidence.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise StrictJSONError(
                f"JSON is not UTF-8 at byte offset {exc.start}") from exc
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicates,
            parse_float=_finite_float,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, StrictJSONError):
        raise
    except RecursionError as exc:
        raise StrictJSONError("JSON exceeds the safe nesting depth") from exc
    except ValueError as exc:
        # For example, Python's integer digit limit. Do not echo raw numeric
        # material from a customer-controlled document into logs.
        raise StrictJSONError("JSON contains an invalid numeric value") from exc


def json_error_detail(exc: BaseException) -> str:
    """Return one bounded diagnostic that never includes JSON payload values."""
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    if isinstance(exc, StrictJSONError):
        return str(exc)
    if isinstance(exc, UnicodeDecodeError):
        return f"JSON is not UTF-8 at byte offset {exc.start}"
    return f"invalid JSON ({type(exc).__name__})"
