#!/usr/bin/env python3
"""Build a traffic profile from real request logs.

Reads per-request records (JSONL or CSV) and computes the P50/P95 quantiles
the harness needs: input tokens, output tokens, and prompt-cache-hit
fraction. Emits a profile JSON ready to drop into configs/ and point a run
config at with profile_path.

Only the distribution is extracted. No prompt text is read or stored, so a
log export with token counts is enough and no customer content moves.

Usage:
  python3 scripts/profile_from_logs.py --input logs.jsonl --name decagon_real
  python3 scripts/profile_from_logs.py --input logs.csv \
      --out configs/profile_decagon_real.json \
      --input-field prompt_tokens --output-field completion_tokens \
      --cached-field cached_tokens

Verify the result with:
  python3 -m traffic_replay sample --profile configs/profile_decagon_real.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load_records(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(text.splitlines()))
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _column(records: list[dict], field: str) -> np.ndarray:
    values = []
    for r in records:
        v = r.get(field)
        if v is None or v == "":
            continue
        values.append(float(v))
    return np.asarray(values, dtype=float)


def _cache_fractions(records, input_field, cached_field,
                     cache_fraction_field) -> np.ndarray:
    """Per-record cache fraction, so input and cached always come from the
    same request even when some rows are missing a field."""
    out = []
    for r in records:
        iv = r.get(input_field)
        if iv is None or iv == "" or float(iv) <= 0:
            continue
        if cache_fraction_field:
            cv = r.get(cache_fraction_field)
            if cv is None or cv == "":
                continue
            out.append(float(cv))
        else:
            cd = r.get(cached_field)
            if cd is None or cd == "":
                continue
            out.append(float(cd) / float(iv))
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def build_profile(records, name, input_field, output_field,
                  cached_field, cache_fraction_field):
    inp = _column(records, input_field)
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

    # Round first, THEN enforce the sampler's open-interval constraints on the
    # rounded values, so rounding cannot re-collapse a valid interval into one
    # profile.py rejects (sizes need p95 > p50 > 0; cache needs
    # 0 < p50 < p95 < 1).
    inp_q, out_q, cf_q = qint(inp), qint(out), qflt(cf)
    warnings = []
    for label, d in (("input_tokens", inp_q), ("output_tokens", out_q)):
        if d["p50"] < 1:
            d["p50"] = 1
        if d["p95"] <= d["p50"]:
            d["p95"] = d["p50"] + 1
            warnings.append(
                f"{label}: p95 <= p50 after rounding, bumped p95 to p50 + 1")
    if not (0.0 < cf_q["p50"] < cf_q["p95"] < 1.0):
        p50 = min(max(cf_q["p50"], 0.01), 0.98)
        p95 = min(max(cf_q["p95"], p50 + 0.01), 0.99)
        warnings.append(
            f"cache_fraction: clamped to p50={p50:.3f} p95={p95:.3f} "
            "(fits need 0 < p50 < p95 < 1)")
        cf_q = {"p50": p50, "p95": p95}

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    return {
        "name": name,
        "input_tokens": inp_q,
        "output_tokens": out_q,
        "cache_fraction": cf_q,
        "provenance": f"Computed from {len(records)} request records.",
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
    text = json.dumps(profile, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
