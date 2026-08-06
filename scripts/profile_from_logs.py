#!/usr/bin/env python3
"""Build a traffic profile from real request logs.

Reads per-request records (JSONL or CSV) and computes the P50/P95 quantiles
the harness needs: input tokens, output tokens, and prompt-cache-hit
fraction. Emits a profile JSON ready to drop into configs/ and point a run
config at with profile_path.

Only the distribution is extracted. No prompt text is read or stored, so a
log export with token counts is enough and no customer content moves.

Usage:
  python3 scripts/profile_from_logs.py --input logs.jsonl --name agent_real
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
import json
import math
import sys
from pathlib import Path

import numpy as np


def _load_records(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix.lower() == ".csv":
        records = list(csv.DictReader(text.splitlines()))
        if any(not isinstance(row, dict) for row in records):
            raise ValueError(f"{path}: CSV rows must be objects")
        return records
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each record must be an object")
            records.append(value)
    return records


def _numeric(value, field: str, record_number: int, *, integer=False,
             positive=False) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"record {record_number} field {field!r} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
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


def _column(records: list[dict], field: str, *, positive: bool = False) -> np.ndarray:
    values = []
    for record_number, r in enumerate(records, 1):
        v = r.get(field)
        if v is None or v == "":
            continue
        values.append(_numeric(
            v, field, record_number, integer=True, positive=positive))
    return np.asarray(values, dtype=float)


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


def build_profile(records, name, input_field, output_field,
                  cached_field, cache_fraction_field):
    if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be a non-empty string")
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

    return {
        "name": name,
        "input_tokens": inp_q,
        "output_tokens": out_q,
        "cache_fraction": cf_q,
        "provenance": (
            f"Computed from {len(records)} request records; "
            f"usable input={len(inp)}, output={len(out)}, cache={len(cf)}."),
        "label": "Built from a real dataset. Verify the recovered quantiles "
                 "with 'python3 -m traffic_replay sample --profile <this file>'.",
    }


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
    args = ap.parse_args(argv)

    records = _load_records(Path(args.input))
    if not records:
        raise SystemExit(f"no records in {args.input}")
    profile = build_profile(
        records, args.name, args.input_field, args.output_field,
        args.cached_field, args.cache_fraction_field)
    text = json.dumps(profile, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
