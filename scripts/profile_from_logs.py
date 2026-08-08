#!/usr/bin/env python3
"""Build a traffic profile from real request logs.

Reads per-request records (JSONL or CSV) and emits either a backward-
compatible P50/P95 profile or a schema-v2 empirical-joint profile. The joint
mode deduplicates complete numeric triples and stores only integer frequency
weights; it preserves observed combinations without copying arbitrary source
fields.

The parser necessarily reads each complete source record. Only the selected
numeric fields and aggregate extraction counts are emitted; prompt text and
all other arbitrary source fields are neither copied nor stored in the output.
A token-count-only export is still the safest input.

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
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from traffic_replay.json_input import json_error_detail, loads_strict  # noqa: E402


def _parse_records(path: Path, raw: bytes) -> list[dict]:
    """Parse the exact byte sequence whose provenance will be reported."""
    if path.suffix.lower() == ".csv":
        try:
            text = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path}: CSV is not UTF-8 at byte offset {exc.start}") \
                from exc
        reader = csv.DictReader(text.splitlines())
        headers = reader.fieldnames
        if headers is None:
            return []
        if any(header is None or not header.strip() for header in headers):
            raise ValueError(f"{path}: CSV headers must be non-empty")
        if len(set(headers)) != len(headers):
            raise ValueError(f"{path}: CSV headers must be unique")
        records = list(reader)
        if any(None in row for row in records):
            raise ValueError(f"{path}: CSV row has more values than headers")
        if any(not isinstance(row, dict) for row in records):
            raise ValueError(f"{path}: CSV rows must be objects")
        return records
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                value = loads_strict(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON "
                    f"({json_error_detail(exc)})") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each record must be an object")
            records.append(value)
    return records


def _load_records(path: Path) -> list[dict]:
    """Load records from one immutable-in-memory snapshot of ``path``."""
    return _parse_records(path, path.read_bytes())


def _numeric(value, field: str, record_number: int, *, integer=False,
             positive=False) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"record {record_number} field {field!r} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"record {record_number} field {field!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(
            f"record {record_number} field {field!r} must be finite")
    if integer and not number.is_integer():
        raise ValueError(
            f"record {record_number} field {field!r} must be an integer count")
    if number < 0 or (positive and number <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"record {record_number} field {field!r} must be {qualifier}")
    return number


def _column(records: list[dict], field: str, *,
            positive: bool = False) -> np.ndarray:
    values = []
    for record_number, r in enumerate(records, 1):
        v = r.get(field)
        if v is None or v == "":
            continue
        values.append(_numeric(
            v, field, record_number, integer=True, positive=positive))
    return np.asarray(values, dtype=float)


def _missing(record: dict, field: str | None) -> bool:
    if field is None:
        return True
    value = record.get(field)
    return value is None or value == ""


def _cache_fractions(records, input_field, cached_field,
                     cache_fraction_field) -> np.ndarray:
    """Per-record cache fraction, so input and cached always come from the
    same request even when some rows are missing a field."""
    out = []
    for record_number, r in enumerate(records, 1):
        iv = r.get(input_field)
        if iv is None or iv == "":
            continue
        input_tokens = _numeric(
            iv, input_field, record_number, integer=True, positive=True)
        if cache_fraction_field:
            cv = r.get(cache_fraction_field)
            if cv is None or cv == "":
                continue
            fraction = _numeric(cv, cache_fraction_field, record_number)
            if fraction > 1:
                raise ValueError(
                    f"record {record_number} field {cache_fraction_field!r} "
                    "must be between 0 and 1")
            out.append(fraction)
        else:
            cd = r.get(cached_field)
            if cd is None or cd == "":
                continue
            cached = _numeric(
                cd, cached_field, record_number, integer=True)
            if cached > input_tokens:
                raise ValueError(
                    f"record {record_number} field {cached_field!r} cannot "
                    f"exceed {input_field!r}")
            out.append(cached / input_tokens)
    return np.asarray(out, dtype=float)


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


def _extraction_counts(records: list[dict], input_field: str,
                       output_field: str, cached_field: str,
                       cache_fraction_field: str | None,
                       *, usable_input: int, usable_output: int,
                       usable_cache: int) -> dict:
    cache_field = cache_fraction_field or cached_field
    incomplete = sum(
        _missing(record, input_field)
        or _missing(record, output_field)
        or _missing(record, cache_field)
        for record in records)
    return {
        "total_records": len(records),
        "usable_input_records": usable_input,
        "dropped_input_records": len(records) - usable_input,
        "usable_output_records": usable_output,
        "dropped_output_records": len(records) - usable_output,
        "usable_cache_records": usable_cache,
        "dropped_cache_records": len(records) - usable_cache,
        "complete_joint_records": len(records) - incomplete,
        "dropped_incomplete_joint_records": incomplete,
    }


def _weighted_anchor(rows: list[dict], field: str,
                     probability: float) -> float:
    ordered = sorted((float(row[field]), int(row["weight"])) for row in rows)
    total = sum(weight for _, weight in ordered)
    rank = max(0, math.ceil(probability * total) - 1)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative > rank:
            return value
    raise AssertionError("empirical rows unexpectedly had no weighted anchor")


def _empirical_rows(records: list[dict], input_field: str,
                    output_field: str, cached_field: str,
                    cache_fraction_field: str | None) -> tuple[list[dict], dict]:
    cache_field = cache_fraction_field or cached_field
    counts: Counter[tuple[int, int, float]] = Counter()
    missing_input = missing_output = missing_cache = 0
    dropped = 0
    for record_number, record in enumerate(records, 1):
        no_input = _missing(record, input_field)
        no_output = _missing(record, output_field)
        no_cache = _missing(record, cache_field)
        missing_input += int(no_input)
        missing_output += int(no_output)
        missing_cache += int(no_cache)

        input_tokens = None if no_input else int(_numeric(
            record[input_field], input_field, record_number,
            integer=True, positive=True))
        output_tokens = None if no_output else int(_numeric(
            record[output_field], output_field, record_number,
            integer=True, positive=True))
        cache_value = None
        if not no_cache:
            if cache_fraction_field:
                cache_value = _numeric(
                    record[cache_fraction_field], cache_fraction_field,
                    record_number)
                if cache_value > 1.0:
                    raise ValueError(
                        f"record {record_number} field "
                        f"{cache_fraction_field!r} must be between 0 and 1")
            else:
                cached_tokens = _numeric(
                    record[cached_field], cached_field, record_number,
                    integer=True)
                if input_tokens is not None and cached_tokens > input_tokens:
                    raise ValueError(
                        f"record {record_number} field {cached_field!r} "
                        f"cannot exceed {input_field!r}")
                if input_tokens is not None:
                    cache_value = cached_tokens / input_tokens

        if no_input or no_output or no_cache:
            dropped += 1
            continue
        assert input_tokens is not None
        assert output_tokens is not None
        assert cache_value is not None
        counts[(input_tokens, output_tokens, cache_value)] += 1

    rows = [{
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_fraction": cache_fraction,
        "weight": weight,
    } for (input_tokens, output_tokens, cache_fraction), weight
        in sorted(counts.items())]
    extraction = {
        "total_records": len(records),
        "complete_joint_records": len(records) - dropped,
        "dropped_incomplete_joint_records": dropped,
        "records_missing_input": missing_input,
        "records_missing_output": missing_output,
        "records_missing_cache": missing_cache,
        "unique_joint_rows": len(rows),
    }
    return rows, extraction


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


def _build_empirical_profile(records: list[dict], name: str,
                             input_field: str, output_field: str,
                             cached_field: str,
                             cache_fraction_field: str | None,
                             source_sha256: str | None,
                             source_byte_count: int | None) -> dict:
    rows, extraction = _empirical_rows(
        records, input_field, output_field, cached_field,
        cache_fraction_field)
    if not rows:
        raise SystemExit(
            "no complete rows with positive input/output tokens and a cache "
            "signal for empirical-joint mode")

    def anchors(field: str) -> dict:
        p50 = _weighted_anchor(rows, field, 0.5)
        p95 = _weighted_anchor(rows, field, 0.95)
        if field != "cache_fraction":
            p50, p95 = int(p50), int(p95)
        return {"p50": p50, "p95": p95}

    digest_text = _source_provenance(source_sha256, source_byte_count)
    result = {
        "schema_version": 2,
        "name": name,
        "input_tokens": anchors("input_tokens"),
        "output_tokens": anchors("output_tokens"),
        "cache_fraction": anchors("cache_fraction"),
        "sampling": {"mode": "empirical_joint", "rows": rows},
        "provenance": (
            f"Content-free empirical distribution from "
            f"{extraction['complete_joint_records']} complete of "
            f"{extraction['total_records']} request records; "
            f"{extraction['dropped_incomplete_joint_records']} incomplete "
            f"records dropped.{digest_text}"),
        "label": (
            "Built from complete observed token/cache triples. Balanced "
            "weighted cycles preserve their combinations and frequencies."),
        "extraction": extraction,
    }
    result.update(_source_fields(source_sha256, source_byte_count))
    return result


def build_profile(records, name, input_field, output_field,
                  cached_field, cache_fraction_field, *,
                  mode="quantiles", source_sha256=None,
                  source_byte_count=None):
    if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be a non-empty string")
    for field_name, value in (
        ("input_field", input_field), ("output_field", output_field),
        ("cached_field", cached_field),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    if cache_fraction_field is not None and (
            not isinstance(cache_fraction_field, str)
            or not cache_fraction_field):
        raise ValueError(
            "cache_fraction_field must be a non-empty string or null")
    if mode not in {"quantiles", "empirical-joint"}:
        raise ValueError("mode must be 'quantiles' or 'empirical-joint'")
    source_sha256 = _validate_source_sha256(source_sha256)
    source_byte_count = _validate_source_byte_count(
        source_sha256, source_byte_count)
    if mode == "empirical-joint":
        return _build_empirical_profile(
            records, name, input_field, output_field, cached_field,
            cache_fraction_field, source_sha256, source_byte_count)

    inp = _column(records, input_field, positive=True)
    out = _column(records, output_field)
    if inp.size == 0 or out.size == 0:
        raise SystemExit(
            f"no usable rows for {input_field!r} / {output_field!r}")
    cf = _cache_fractions(records, input_field, cached_field,
                          cache_fraction_field)
    if cf.size == 0:
        raise SystemExit(
            "no rows with input tokens and a cache signal "
            f"({cached_field!r} or {cache_fraction_field!r})")

    def qint(a):
        return {"p50": int(round(np.percentile(a, 50))),
                "p95": int(round(np.percentile(a, 95)))}

    def qflt(a, ndigits=3):
        return {"p50": round(float(np.percentile(a, 50)), ndigits),
                "p95": round(float(np.percentile(a, 95)), ndigits)}

    # Constant and boundary distributions are legitimate and supported by the
    # sampler. Never move a customer's measured quantile merely to make a fit.
    inp_q, out_q, cf_q = qint(inp), qint(out), qflt(cf)
    for label, d in (("input_tokens", inp_q), ("output_tokens", out_q)):
        if not (d["p95"] >= d["p50"] > 0):
            raise ValueError(
                f"{label} rounded to invalid quantiles {d}; at least half "
                "the usable records must contain one or more tokens")
    if not (0.0 <= cf_q["p50"] <= cf_q["p95"] <= 1.0):
        raise ValueError(f"cache_fraction produced invalid quantiles {cf_q}")

    extraction = _extraction_counts(
        records, input_field, output_field, cached_field,
        cache_fraction_field, usable_input=len(inp), usable_output=len(out),
        usable_cache=len(cf))
    digest_text = _source_provenance(source_sha256, source_byte_count)
    result = {
        "name": name,
        "input_tokens": inp_q,
        "output_tokens": out_q,
        "cache_fraction": cf_q,
        "provenance": (
            f"Computed from {len(records)} request records; "
            f"usable input={len(inp)}, output={len(out)}, cache={len(cf)}."
            f"{digest_text}"),
        "label": "Built from a real dataset. Verify the recovered quantiles "
                 "with 'python3 -m traffic_replay sample --profile <this file>'.",
        "extraction": extraction,
    }
    result.update(_source_fields(source_sha256, source_byte_count))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a profile JSON from real request logs")
    ap.add_argument("--input", required=True,
                    help="JSONL or CSV of per-request records")
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
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    frozen_source = input_path.read_bytes()
    source_sha256 = hashlib.sha256(frozen_source).hexdigest()
    source_byte_count = len(frozen_source)
    records = _parse_records(input_path, frozen_source)
    if not records:
        raise SystemExit(f"no records in {args.input}")
    profile = build_profile(
        records, args.name, args.input_field, args.output_field,
        args.cached_field, args.cache_fraction_field,
        mode=args.mode, source_sha256=source_sha256,
        source_byte_count=source_byte_count)
    text = json.dumps(profile, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
