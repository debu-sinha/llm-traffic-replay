#!/usr/bin/env python3
"""Build a content-free traffic profile from request-token logs.

The CLI consumes JSONL or CSV incrementally. It hashes the exact bytes read,
but retains only selected numeric token/cache values, aggregate counters, and
(in empirical-joint mode) deduplicated numeric triples. Prompt text and other
customer fields are never copied into the profile or diagnostics.

A token-count-only export is the safest input. Inputs are bounded even when a
caller accidentally supplies raw logs with very large prompt fields.

Usage:
  python3 scripts/profile_from_logs.py --input logs.jsonl --name agent_real
  python3 scripts/profile_from_logs.py --input logs.jsonl --name agent_real \
      --mode empirical-joint
  python3 scripts/profile_from_logs.py --input logs.csv \
      --out configs/profile_agent_real.json \
      --input-field prompt_tokens --output-field completion_tokens \
      --cached-field cached_tokens

Verify the result with:
  python3 -m traffic_replay sample --profile configs/profile_agent_real.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from array import array
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from traffic_replay.json_input import json_error_detail, loads_strict  # noqa: E402


# The defaults bound accidental ingestion while still accommodating a large
# token-only export. Every limit is overridable explicitly at the CLI.
DEFAULT_MAX_BYTES = 1_073_741_824       # 1 GiB
DEFAULT_MAX_LINE_BYTES = 8_388_608      # 8 MiB per physical line
DEFAULT_MAX_RECORD_BYTES = 16_777_216   # 16 MiB per logical CSV/JSONL record
DEFAULT_MAX_LINES = 20_000_000
DEFAULT_MAX_RECORDS = 10_000_000
DEFAULT_MAX_UNIQUE_TRIPLES = 1_000_000
MAX_EXACT_TOKEN_COUNT = (1 << 53) - 1


@dataclass(frozen=True)
class _InputLimits:
    max_bytes: int = DEFAULT_MAX_BYTES
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
    max_lines: int = DEFAULT_MAX_LINES
    max_records: int = DEFAULT_MAX_RECORDS
    max_unique_triples: int = DEFAULT_MAX_UNIQUE_TRIPLES

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class _SourceSummary:
    sha256: str
    byte_count: int


def _safe_path(path: Path) -> str:
    """Bound a user-controlled path before placing it in a diagnostic."""
    rendered = os.fspath(path).replace("\r", "?").replace("\n", "?")
    if len(rendered) > 512:
        rendered = rendered[:509] + "..."
    return rendered


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _open_regular_input(path: Path) -> tuple[BinaryIO, os.stat_result]:
    """Open one regular file descriptor without following a symlink.

    ``lstat`` rejects obvious special files before opening them (important for
    FIFOs). ``O_NOFOLLOW`` closes the check/open symlink race on platforms that
    expose it, and ``fstat`` makes the descriptor type authoritative.
    """
    label = _safe_path(path)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(
            f"{label}: cannot inspect input file ({exc.__class__.__name__})"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label}: input must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label}: input must be a regular file")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # If a regular file is replaced by a FIFO between lstat and open, this
    # keeps the safety check from blocking before fstat can reject it.
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"{label}: cannot safely open input file ({exc.__class__.__name__})"
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label}: input must be a regular file")
        if not _same_file(before, opened):
            raise ValueError(f"{label}: input changed while it was opened")
        return os.fdopen(fd, "rb", buffering=1024 * 1024), opened
    except BaseException:
        os.close(fd)
        raise


class _BoundedLines:
    """Incrementally read, bound, and hash exact physical input lines."""

    def __init__(self, path: Path, handle: BinaryIO,
                 opened: os.stat_result, limits: _InputLimits):
        self.path = path
        self.label = _safe_path(path)
        self.handle = handle
        self.opened = opened
        self.limits = limits
        self.byte_count = 0
        self.line_count = 0
        self._digest = hashlib.sha256()
        self._record_start: int | None = None
        self._saw_eof = False
        if opened.st_size > limits.max_bytes:
            raise ValueError(
                f"{self.label}: input size {opened.st_size} bytes exceeds "
                f"--max-bytes={limits.max_bytes}")

    def __iter__(self):
        return self

    def begin_record(self) -> None:
        if self._record_start is not None:
            raise AssertionError("nested logical input records")
        self._record_start = self.byte_count

    def end_record(self) -> None:
        self._record_start = None

    def __next__(self) -> bytes:
        if self._saw_eof:
            raise StopIteration

        total_remaining = self.limits.max_bytes - self.byte_count
        line_remaining = self.limits.max_line_bytes
        allowed = min(total_remaining, line_remaining)
        record_remaining = None
        if self._record_start is not None:
            record_used = self.byte_count - self._record_start
            record_remaining = self.limits.max_record_bytes - record_used
            allowed = min(allowed, record_remaining)

        # Reading one byte past the tightest bound proves an overrun without
        # ever buffering an unbounded customer-controlled line or CSV field.
        raw = self.handle.readline(max(0, allowed) + 1)
        if raw == b"":
            self._saw_eof = True
            raise StopIteration
        next_line = self.line_count + 1
        if self.line_count >= self.limits.max_lines:
            raise ValueError(
                f"{self.label}: physical line limit exceeded "
                f"(--max-lines={self.limits.max_lines})")
        if len(raw) > total_remaining:
            raise ValueError(
                f"{self.label}: input exceeds "
                f"--max-bytes={self.limits.max_bytes}")
        if len(raw) > self.limits.max_line_bytes:
            raise ValueError(
                f"{self.label}:{next_line}: physical line exceeds "
                f"--max-line-bytes={self.limits.max_line_bytes}")
        if record_remaining is not None and len(raw) > record_remaining:
            raise ValueError(
                f"{self.label}: logical record ending near line {next_line} "
                f"exceeds --max-record-bytes="
                f"{self.limits.max_record_bytes}")

        self.byte_count += len(raw)
        self.line_count = next_line
        self._digest.update(raw)
        return raw

    def finish(self) -> _SourceSummary:
        if not self._saw_eof:
            raise AssertionError("input parser did not consume the entire file")
        closed = os.fstat(self.handle.fileno())
        if not _same_file(self.opened, closed):
            raise ValueError(f"{self.label}: input changed while it was read")
        if (closed.st_size != self.opened.st_size
                or closed.st_mtime_ns != self.opened.st_mtime_ns
                or closed.st_ctime_ns != self.opened.st_ctime_ns
                or closed.st_size != self.byte_count):
            raise ValueError(f"{self.label}: input changed while it was read")
        return _SourceSummary(
            sha256=self._digest.hexdigest(), byte_count=self.byte_count)


def _numeric(value, field: str, record_number: int, *, integer=False,
             positive=False) -> float | int:
    if isinstance(value, bool):
        raise ValueError(
            f"record {record_number} field {field!r} must be numeric")
    if integer:
        if isinstance(value, int):
            number = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"record {record_number} field {field!r} must be finite")
            if not value.is_integer():
                raise ValueError(
                    f"record {record_number} field {field!r} must be an "
                    "integer count")
            if value > MAX_EXACT_TOKEN_COUNT:
                raise ValueError(
                    f"record {record_number} field {field!r} exceeds the "
                    f"exact token-count limit {MAX_EXACT_TOKEN_COUNT}")
            if value < 0 or (positive and value <= 0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(
                    f"record {record_number} field {field!r} must be "
                    f"{qualifier}")
            number = int(value)
        elif isinstance(value, str):
            try:
                decimal = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(
                    f"record {record_number} field {field!r} must be numeric"
                ) from exc
            if not decimal.is_finite():
                raise ValueError(
                    f"record {record_number} field {field!r} must be finite")
            if decimal != decimal.to_integral_value():
                raise ValueError(
                    f"record {record_number} field {field!r} must be an "
                    "integer count")
            if decimal > MAX_EXACT_TOKEN_COUNT:
                raise ValueError(
                    f"record {record_number} field {field!r} exceeds the "
                    f"exact token-count limit {MAX_EXACT_TOKEN_COUNT}")
            if decimal < 0 or (positive and decimal <= 0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(
                    f"record {record_number} field {field!r} must be "
                    f"{qualifier}")
            number = int(decimal)
        else:
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"record {record_number} field {field!r} must be numeric"
                ) from exc
            if not math.isfinite(converted):
                raise ValueError(
                    f"record {record_number} field {field!r} must be finite")
            if not converted.is_integer():
                raise ValueError(
                    f"record {record_number} field {field!r} must be an "
                    "integer count")
            if converted > MAX_EXACT_TOKEN_COUNT:
                raise ValueError(
                    f"record {record_number} field {field!r} exceeds the "
                    f"exact token-count limit {MAX_EXACT_TOKEN_COUNT}")
            if converted < 0 or (positive and converted <= 0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(
                    f"record {record_number} field {field!r} must be "
                    f"{qualifier}")
            number = int(converted)
        if number < 0 or (positive and number <= 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(
                f"record {record_number} field {field!r} must be {qualifier}")
        if number > MAX_EXACT_TOKEN_COUNT:
            raise ValueError(
                f"record {record_number} field {field!r} exceeds the exact "
                f"token-count limit {MAX_EXACT_TOKEN_COUNT}")
        return number
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"record {record_number} field {field!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(
            f"record {record_number} field {field!r} must be finite")
    if number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"record {record_number} field {field!r} must be {qualifier}")
    return number


def _missing(record: dict, field: str | None) -> bool:
    if field is None:
        return True
    value = record.get(field)
    return value is None or value == ""


def _validate_source_sha256(source_sha256: str | None) -> str | None:
    if source_sha256 is None:
        return None
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 \
            or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("source_sha256 must be 64 lowercase hexadecimal digits")
    return source_sha256


def _validate_source_byte_count(source_sha256: str | None,
                                source_byte_count: int | None) -> int | None:
    if source_byte_count is None:
        return None
    if source_sha256 is None:
        raise ValueError("source_byte_count requires source_sha256")
    if not isinstance(source_byte_count, int) \
            or isinstance(source_byte_count, bool) \
            or source_byte_count < 0:
        raise ValueError("source_byte_count must be a non-negative integer")
    return source_byte_count


def _weighted_anchor(rows: list[dict], field: str,
                     probability: float) -> float:
    ordered = sorted((row[field], int(row["weight"])) for row in rows)
    total = sum(weight for _, weight in ordered)
    rank = max(0, math.ceil(probability * total) - 1)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative > rank:
            return value
    raise AssertionError("empirical rows unexpectedly had no weighted anchor")


def _source_fields(source_sha256: str | None,
                   source_byte_count: int | None) -> dict:
    if source_sha256 is None:
        return {}
    source = {"digest_algorithm": "sha256", "sha256": source_sha256}
    if source_byte_count is not None:
        source["bytes"] = source_byte_count
    return {"source": source}


def _source_provenance(source_sha256: str | None,
                       source_byte_count: int | None) -> str:
    if source_sha256 is None:
        return ""
    byte_text = (f"; bytes: {source_byte_count}"
                 if source_byte_count is not None else "")
    return f" Source SHA-256: {source_sha256}{byte_text}."


class _ProfileAccumulator:
    """Retain numeric signals and counters, never complete source records."""

    def __init__(self, name: str, input_field: str, output_field: str,
                 cached_field: str, cache_fraction_field: str | None,
                 mode: str, *, max_unique_triples: int | None = None):
        self.name = name
        self.input_field = input_field
        self.output_field = output_field
        self.cached_field = cached_field
        self.cache_fraction_field = cache_fraction_field
        self.cache_field = cache_fraction_field or cached_field
        self.mode = mode
        self.max_unique_triples = max_unique_triples

        self.total = 0
        self.missing_input = 0
        self.missing_output = 0
        self.missing_cache = 0
        self.incomplete = 0
        # Compact native arrays avoid Python-object overhead for production
        # scale quantiles. They contain selected numeric signals only.
        self.inputs = array("d")
        self.outputs = array("d")
        self.cache_fractions = array("d")
        self.joint: Counter[tuple[int, int, float]] = Counter()

    def add(self, record: dict) -> None:
        if not isinstance(record, dict):
            raise ValueError(
                f"record {self.total + 1} must be an object")
        self.total += 1
        record_number = self.total
        no_input = _missing(record, self.input_field)
        no_output = _missing(record, self.output_field)
        no_cache = _missing(record, self.cache_field)
        self.missing_input += int(no_input)
        self.missing_output += int(no_output)
        self.missing_cache += int(no_cache)
        self.incomplete += int(no_input or no_output or no_cache)

        input_tokens = None if no_input else _numeric(
            record[self.input_field], self.input_field, record_number,
            integer=True, positive=True)
        output_tokens = None if no_output else _numeric(
            record[self.output_field], self.output_field, record_number,
            integer=True, positive=self.mode == "empirical-joint")

        cache_value = None
        # Quantile mode historically pairs cache signals only with rows that
        # have input tokens. Empirical mode validates every selected present
        # field, including incomplete rows, before dropping the incomplete
        # joint tuple.
        if (not no_cache and (input_tokens is not None
                              or self.mode == "empirical-joint")):
            if self.cache_fraction_field:
                cache_value = _numeric(
                    record[self.cache_fraction_field],
                    self.cache_fraction_field, record_number)
                if cache_value > 1:
                    raise ValueError(
                        f"record {record_number} field "
                        f"{self.cache_fraction_field!r} must be between 0 and 1")
            else:
                cached_tokens = _numeric(
                    record[self.cached_field], self.cached_field,
                    record_number, integer=True)
                if input_tokens is not None and cached_tokens > input_tokens:
                    raise ValueError(
                        f"record {record_number} field {self.cached_field!r} "
                        f"cannot exceed {self.input_field!r}")
                if input_tokens is not None:
                    cache_value = cached_tokens / input_tokens

        if self.mode == "quantiles":
            if input_tokens is not None:
                self.inputs.append(input_tokens)
            if output_tokens is not None:
                self.outputs.append(output_tokens)
            if cache_value is not None:
                self.cache_fractions.append(cache_value)
            return

        if no_input or no_output or no_cache:
            return
        assert input_tokens is not None
        assert output_tokens is not None
        assert cache_value is not None
        key = (int(input_tokens), int(output_tokens), cache_value)
        if (key not in self.joint and self.max_unique_triples is not None
                and len(self.joint) >= self.max_unique_triples):
            raise ValueError(
                "unique empirical triple limit exceeded "
                f"(--max-unique-triples={self.max_unique_triples})")
        self.joint[key] += 1

    def finish(self, *, source_sha256: str | None,
               source_byte_count: int | None) -> dict:
        if self.mode == "empirical-joint":
            return self._finish_empirical(
                source_sha256=source_sha256,
                source_byte_count=source_byte_count)
        return self._finish_quantiles(
            source_sha256=source_sha256,
            source_byte_count=source_byte_count)

    def _finish_quantiles(self, *, source_sha256: str | None,
                          source_byte_count: int | None) -> dict:
        inp = np.asarray(self.inputs, dtype=float)
        out = np.asarray(self.outputs, dtype=float)
        cf = np.asarray(self.cache_fractions, dtype=float)
        if inp.size == 0 or out.size == 0:
            raise SystemExit(
                f"no usable rows for {self.input_field!r} / "
                f"{self.output_field!r}")
        if cf.size == 0:
            raise SystemExit(
                "no rows with input tokens and a cache signal "
                f"({self.cached_field!r} or "
                f"{self.cache_fraction_field!r})")

        def qint(values):
            p50, p95 = np.percentile(values, [50, 95])
            return {
                "p50": int(round(p50)),
                "p95": int(round(p95)),
            }

        def qflt(values, ndigits=3):
            p50, p95 = np.percentile(values, [50, 95])
            return {
                "p50": round(float(p50), ndigits),
                "p95": round(float(p95), ndigits),
            }

        inp_q, out_q, cf_q = qint(inp), qint(out), qflt(cf)
        for label, value in (
                ("input_tokens", inp_q), ("output_tokens", out_q)):
            if not (value["p95"] >= value["p50"] > 0):
                raise ValueError(
                    f"{label} rounded to invalid quantiles {value}; at least "
                    "half the usable records must contain one or more tokens")
        if not (0 <= cf_q["p50"] <= cf_q["p95"] <= 1):
            raise ValueError(
                f"cache_fraction produced invalid quantiles {cf_q}")

        extraction = {
            "total_records": self.total,
            "usable_input_records": len(self.inputs),
            "dropped_input_records": self.missing_input,
            "usable_output_records": len(self.outputs),
            "dropped_output_records": self.missing_output,
            "usable_cache_records": len(self.cache_fractions),
            "dropped_cache_records": self.total - len(self.cache_fractions),
            "complete_joint_records": self.total - self.incomplete,
            "dropped_incomplete_joint_records": self.incomplete,
        }
        digest_text = _source_provenance(
            source_sha256, source_byte_count)
        result = {
            "name": self.name,
            "input_tokens": inp_q,
            "output_tokens": out_q,
            "cache_fraction": cf_q,
            "provenance": (
                f"Computed from {self.total} request records; "
                f"usable input={len(self.inputs)}, "
                f"output={len(self.outputs)}, "
                f"cache={len(self.cache_fractions)}.{digest_text}"),
            "label": (
                "Built from a real dataset. Verify the recovered quantiles "
                "with 'python3 -m traffic_replay sample --profile "
                "<this file>'."),
            "extraction": extraction,
        }
        result.update(_source_fields(source_sha256, source_byte_count))
        return result

    def _finish_empirical(self, *, source_sha256: str | None,
                          source_byte_count: int | None) -> dict:
        rows = [{
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_fraction": cache_fraction,
            "weight": weight,
        } for (input_tokens, output_tokens, cache_fraction), weight
            in sorted(self.joint.items())]
        self.joint.clear()
        if not rows:
            raise SystemExit(
                "no complete rows with positive input/output tokens and a "
                "cache signal for empirical-joint mode")

        def anchors(field: str) -> dict:
            p50 = _weighted_anchor(rows, field, 0.5)
            p95 = _weighted_anchor(rows, field, 0.95)
            if field != "cache_fraction":
                p50, p95 = int(p50), int(p95)
            return {"p50": p50, "p95": p95}

        extraction = {
            "total_records": self.total,
            "complete_joint_records": self.total - self.incomplete,
            "dropped_incomplete_joint_records": self.incomplete,
            "records_missing_input": self.missing_input,
            "records_missing_output": self.missing_output,
            "records_missing_cache": self.missing_cache,
            "unique_joint_rows": len(rows),
        }
        digest_text = _source_provenance(
            source_sha256, source_byte_count)
        result = {
            "schema_version": 2,
            "name": self.name,
            "input_tokens": anchors("input_tokens"),
            "output_tokens": anchors("output_tokens"),
            "cache_fraction": anchors("cache_fraction"),
            "sampling": {"mode": "empirical_joint", "rows": rows},
            "provenance": (
                "Content-free empirical distribution from "
                f"{extraction['complete_joint_records']} complete of "
                f"{extraction['total_records']} request records; "
                f"{extraction['dropped_incomplete_joint_records']} "
                f"incomplete records dropped.{digest_text}"),
            "label": (
                "Built from complete observed token/cache triples. Balanced "
                "weighted cycles preserve their combinations and "
                "frequencies."),
            "extraction": extraction,
        }
        result.update(_source_fields(source_sha256, source_byte_count))
        return result


def _validate_profile_arguments(name, input_field, output_field,
                                cached_field, cache_fraction_field,
                                mode) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be a non-empty string")
    for field_name, value in (
            ("input_field", input_field), ("output_field", output_field),
            ("cached_field", cached_field)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    if cache_fraction_field is not None and (
            not isinstance(cache_fraction_field, str)
            or not cache_fraction_field):
        raise ValueError(
            "cache_fraction_field must be a non-empty string or null")
    if mode not in {"quantiles", "empirical-joint"}:
        raise ValueError("mode must be 'quantiles' or 'empirical-joint'")


def build_profile(records, name, input_field, output_field,
                  cached_field, cache_fraction_field, *,
                  mode="quantiles", source_sha256=None,
                  source_byte_count=None):
    """Build a profile from an already-materialized, caller-owned record list.

    The production CLI does not use this compatibility API; it streams records
    directly into the same numeric-only accumulator.
    """
    if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects")
    _validate_profile_arguments(
        name, input_field, output_field, cached_field,
        cache_fraction_field, mode)
    source_sha256 = _validate_source_sha256(source_sha256)
    source_byte_count = _validate_source_byte_count(
        source_sha256, source_byte_count)
    accumulator = _ProfileAccumulator(
        name, input_field, output_field, cached_field,
        cache_fraction_field, mode)
    for record in records:
        accumulator.add(record)
    return accumulator.finish(
        source_sha256=source_sha256,
        source_byte_count=source_byte_count)


def _selected_csv_record(row: list[str], indices: dict[str, int]) -> dict:
    # Only the selected cells survive beyond this call. Short rows map missing
    # selected columns to None, matching csv.DictReader's former semantics.
    return {
        field: row[index] if index < len(row) else None
        for field, index in indices.items()
    }


def _consume_jsonl(lines: _BoundedLines, accumulator: _ProfileAccumulator,
                   limits: _InputLimits) -> None:
    while True:
        lines.begin_record()
        try:
            raw = next(lines)
        except StopIteration:
            break
        finally:
            lines.end_record()
        stripped = raw.strip()
        if not stripped:
            continue
        if accumulator.total >= limits.max_records:
            raise ValueError(
                f"{lines.label}: record limit exceeded "
                f"(--max-records={limits.max_records})")
        try:
            value = loads_strict(stripped)
        except ValueError as exc:
            raise ValueError(
                f"{lines.label}:{lines.line_count}: invalid JSON "
                f"({json_error_detail(exc)})") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"{lines.label}:{lines.line_count}: each record must be an "
                "object")
        wanted_fields = {
            accumulator.input_field,
            accumulator.output_field,
            accumulator.cache_field,
        }
        selected = {
            field: value[field] for field in wanted_fields if field in value
        }
        # Release the complete object and raw record before numeric validation,
        # so even a validation exception cannot retain arbitrary source fields.
        del value, wanted_fields, stripped, raw
        try:
            accumulator.add(selected)
        except ValueError as exc:
            raise ValueError(f"{lines.label}: {exc}") from exc


def _decode_csv_line(raw: bytes, *, first_line: bool,
                     lines: _BoundedLines) -> str:
    offset = lines.byte_count - len(raw)
    try:
        return raw.decode("utf-8-sig" if first_line else "utf-8",
                          errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{lines.label}: CSV is not UTF-8 at byte offset "
            f"{offset + exc.start}") from exc


class _DecodedCSVLines:
    def __init__(self, source: _BoundedLines):
        self.source = source
        self.first_line = True

    def __iter__(self):
        return self

    def __next__(self) -> str:
        raw = next(self.source)
        decoded = _decode_csv_line(
            raw, first_line=self.first_line, lines=self.source)
        self.first_line = False
        return decoded


def _consume_csv(lines: _BoundedLines, accumulator: _ProfileAccumulator,
                 limits: _InputLimits) -> None:
    decoded = _DecodedCSVLines(lines)
    reader = csv.reader(decoded, strict=True)
    old_field_limit = csv.field_size_limit()
    csv.field_size_limit(min(
        limits.max_record_bytes, sys.maxsize, (1 << 31) - 1))
    try:
        lines.begin_record()
        try:
            headers = next(reader)
        except StopIteration:
            lines.end_record()
            return
        finally:
            if lines._record_start is not None:
                lines.end_record()
        if any(not header.strip() for header in headers):
            raise ValueError(f"{lines.label}: CSV headers must be non-empty")
        if len(set(headers)) != len(headers):
            raise ValueError(f"{lines.label}: CSV headers must be unique")

        wanted_fields = {
            accumulator.input_field,
            accumulator.output_field,
            accumulator.cache_field,
        }
        selected_indices = {
            header: index for index, header in enumerate(headers)
            if header in wanted_fields
        }
        header_count = len(headers)
        # Only selected header names/indices are needed after intake setup.
        del headers, wanted_fields
        while True:
            lines.begin_record()
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise ValueError(
                    f"{lines.label}: malformed CSV near physical line "
                    f"{reader.line_num} ({exc.__class__.__name__})") from exc
            finally:
                lines.end_record()
            if not row:
                continue
            if len(row) > header_count:
                raise ValueError(
                    f"{lines.label}: CSV row has more values than headers")
            if accumulator.total >= limits.max_records:
                raise ValueError(
                    f"{lines.label}: record limit exceeded "
                    f"(--max-records={limits.max_records})")
            selected = _selected_csv_record(row, selected_indices)
            del row
            try:
                accumulator.add(selected)
            except ValueError as exc:
                raise ValueError(f"{lines.label}: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(
            f"{lines.label}: malformed CSV near physical line "
            f"{reader.line_num} ({exc.__class__.__name__})") from exc
    finally:
        csv.field_size_limit(old_field_limit)


def _profile_from_path(path: Path, name: str, input_field: str,
                       output_field: str, cached_field: str,
                       cache_fraction_field: str | None, *, mode: str,
                       limits: _InputLimits) -> dict:
    _validate_profile_arguments(
        name, input_field, output_field, cached_field,
        cache_fraction_field, mode)
    accumulator = _ProfileAccumulator(
        name, input_field, output_field, cached_field,
        cache_fraction_field, mode,
        max_unique_triples=limits.max_unique_triples)
    handle, opened = _open_regular_input(path)
    try:
        lines = _BoundedLines(path, handle, opened, limits)
        if path.suffix.lower() == ".csv":
            _consume_csv(lines, accumulator, limits)
        else:
            _consume_jsonl(lines, accumulator, limits)
        summary = lines.finish()
    finally:
        handle.close()
    if accumulator.total == 0:
        raise SystemExit(f"no records in {_safe_path(path)}")
    return accumulator.finish(
        source_sha256=summary.sha256,
        source_byte_count=summary.byte_count)


def _positive_cli_integer(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _validate_output_target(input_path: Path, output_path: Path) -> None:
    label = _safe_path(output_path)
    try:
        output_stat = os.lstat(output_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(
            f"{label}: cannot inspect output path "
            f"({exc.__class__.__name__})") from exc
    if stat.S_ISLNK(output_stat.st_mode):
        raise ValueError(f"{label}: output must not be a symbolic link")
    if not stat.S_ISREG(output_stat.st_mode):
        raise ValueError(f"{label}: output must be a regular file")
    try:
        input_stat = os.stat(input_path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"{_safe_path(input_path)}: cannot recheck input file "
            f"({exc.__class__.__name__})") from exc
    if _same_file(input_stat, output_stat):
        raise ValueError("--out must not overwrite the input log file")


def _write_profile(path: Path, text: str) -> None:
    """Atomically publish a private-by-default content-free profile."""
    parent = path.parent
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=parent, prefix=f".{path.name}.", suffix=".tmp", text=True)
    except OSError as exc:
        raise ValueError(
            f"{_safe_path(path)}: cannot create output "
            f"({exc.__class__.__name__})") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a content-free profile JSON from request logs")
    ap.add_argument("--input", required=True,
                    help="regular JSONL or CSV file of per-request records")
    ap.add_argument("--name", default="real_profile")
    ap.add_argument("--out", help="write here; default is stdout")
    ap.add_argument("--input-field", default="input_tokens")
    ap.add_argument("--output-field", default="output_tokens")
    ap.add_argument("--cached-field", default="cached_tokens",
                    help="cached prompt tokens; fraction = cached / input")
    ap.add_argument("--cache-fraction-field", default=None,
                    help="use a precomputed per-request fraction instead")
    ap.add_argument(
        "--mode", choices=("quantiles", "empirical-joint"),
        default="quantiles",
        help="quantiles keeps the legacy P50/P95 profile; empirical-joint "
             "preserves complete observed triples and their frequencies")
    ap.add_argument("--max-bytes", type=_positive_cli_integer,
                    default=DEFAULT_MAX_BYTES,
                    help=f"maximum input bytes (default: {DEFAULT_MAX_BYTES})")
    ap.add_argument(
        "--max-line-bytes", type=_positive_cli_integer,
        default=DEFAULT_MAX_LINE_BYTES,
        help="maximum bytes in one physical line "
             f"(default: {DEFAULT_MAX_LINE_BYTES})")
    ap.add_argument(
        "--max-record-bytes", type=_positive_cli_integer,
        default=DEFAULT_MAX_RECORD_BYTES,
        help="maximum bytes in one logical record "
             f"(default: {DEFAULT_MAX_RECORD_BYTES})")
    ap.add_argument("--max-lines", type=_positive_cli_integer,
                    default=DEFAULT_MAX_LINES,
                    help=f"maximum physical lines (default: {DEFAULT_MAX_LINES})")
    ap.add_argument(
        "--max-records", type=_positive_cli_integer,
        default=DEFAULT_MAX_RECORDS,
        help=f"maximum request records (default: {DEFAULT_MAX_RECORDS})")
    ap.add_argument(
        "--max-unique-triples", type=_positive_cli_integer,
        default=DEFAULT_MAX_UNIQUE_TRIPLES,
        help="maximum retained unique triples in empirical-joint mode "
             f"(default: {DEFAULT_MAX_UNIQUE_TRIPLES})")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.out) if args.out else None
    limits = _InputLimits(
        max_bytes=args.max_bytes,
        max_line_bytes=args.max_line_bytes,
        max_record_bytes=args.max_record_bytes,
        max_lines=args.max_lines,
        max_records=args.max_records,
        max_unique_triples=args.max_unique_triples)
    try:
        if output_path is not None:
            _validate_output_target(input_path, output_path)
        profile = _profile_from_path(
            input_path, args.name, args.input_field,
            args.output_field, args.cached_field,
            args.cache_fraction_field, mode=args.mode, limits=limits)
        text = json.dumps(profile, indent=2, allow_nan=False)
        if output_path is not None:
            _write_profile(output_path, text)
    except ValueError as exc:
        ap.error(str(exc))
    except OSError as exc:
        ap.error(f"file operation failed ({exc.__class__.__name__})")
    if output_path is not None:
        print(f"wrote {_safe_path(output_path)}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
