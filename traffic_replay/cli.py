"""Command line interface.

  python -m traffic_replay sample   --profile configs/profile_X.json
  python -m traffic_replay schedule --duration 300
  python -m traffic_replay validate            # full self-test vs bundled mock
  python -m traffic_replay run      --config configs/run_smoke.json
  python -m traffic_replay merge    OUT_DIR RUN_DIR1 RUN_DIR2 ...
  python -m traffic_replay compare  OUT_DIR RUN_DIR_A RUN_DIR_B ...
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path


def cmd_sample(args) -> int:
    from . import profile as prof
    p = prof.Profile.from_json(args.profile)
    d = prof.sample(p, args.n, seed=args.seed)
    print(json.dumps({"profile": p.name, "provenance": p.provenance,
                      "label": p.label,
                      "recovered": prof.quantile_report(d)}, indent=2))
    return 0


def cmd_schedule(args) -> int:
    from .schedule import make_schedule, schedule_report
    s = make_schedule(duration_s=args.duration, rate_scale=args.rate_scale)
    print(json.dumps(schedule_report(s), indent=2))
    return 0


def cmd_run(args) -> int:
    from .runner import RunConfig, run
    cfg = json.loads(Path(args.config).read_text())
    rc = RunConfig(**cfg)
    out = run(rc)
    print(json.dumps(out["summary"], indent=2)[:4000])
    print(f"\nopen in a browser: {out['out_dir']}/report.html")
    print(f"full outputs:      {out['out_dir']}")
    return 0


def cmd_validate(args) -> int:
    """Instrument self-test: run the whole pipeline against the bundled mock
    and report client-measured vs server-true latency error."""
    import numpy as np
    from .mock_server import serve
    from .runner import RunConfig, run

    port = args.port
    truth = Path(args.workdir) / "mock_truth.jsonl"
    srv = serve(port, truth)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        rc = RunConfig(
            profile_path=str(Path(__file__).parent.parent
                             / "configs" / "profile_validation_small.json"),
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "TRAFFIC_REPLAY_NO_TOKEN"},
            duration_s=args.duration, qps_base=6.0, qps_burst=18.0,
            qps_min=2.0, qps_max=30.0, rate_scale=1.0,
            max_concurrency=64, cpt=4.0, calibrate_n=8,
            out_dir=str(Path(args.workdir) / "results"),
            title="instrument validation vs bundled mock",
            label="VALIDATION RUN, mock endpoint, known latency model",
            max_output_tokens_cap=24,
        )
        out = run(rc, quiet=args.quiet)
    finally:
        srv.shutdown()

    # join client measurements to server truth
    truth_by_id = {}
    for line in truth.read_text().splitlines():
        rec = json.loads(line)
        truth_by_id[rec["request_id"]] = rec
    rows = []
    for line in (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") != "replay" or not r.get("ok"):
            continue
        tr = truth_by_id.get(r["request_id"])
        if tr and r.get("ttft_ms") is not None:
            rows.append((r["ttft_ms"], tr["ttft_true_ms"],
                         r["e2e_ms"], tr["e2e_true_ms"]))
    if not rows:
        print("VALIDATE: no joinable rows, FAIL")
        return 1
    a = np.array(rows)
    ttft_err = a[:, 0] - a[:, 1]
    e2e_err = a[:, 2] - a[:, 3]
    rep = {
        "joined_requests": len(rows),
        "ttft_error_ms": {"p50": float(np.percentile(ttft_err, 50)),
                          "p95": float(np.percentile(ttft_err, 95)),
                          "max": float(ttft_err.max())},
        "e2e_error_ms": {"p50": float(np.percentile(e2e_err, 50)),
                         "p95": float(np.percentile(e2e_err, 95))},
        "note": "error = client-measured minus server-true; includes real "
                "localhost network+parse overhead, so small positive is "
                "expected and honest",
    }
    print(json.dumps(rep, indent=2))
    ok = rep["ttft_error_ms"]["p95"] < args.tolerance_ms
    print(f"VALIDATE: {'PASS' if ok else 'FAIL'} "
          f"(ttft error p95 {rep['ttft_error_ms']['p95']:.1f} ms "
          f"vs tolerance {args.tolerance_ms} ms)")
    return 0 if ok else 1


def cmd_merge(args) -> int:
    from . import profile as prof
    from .aggregate import merge_runs
    acceptance = None
    if args.profile:
        acceptance = (prof.Profile.from_json(args.profile).extra or {}).get(
            "acceptance_targets")
    try:
        out = merge_runs(args.out, args.inputs, title=args.title,
                         acceptance=acceptance, force=args.force)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"merged -> {out}")
    return 0


def cmd_compare(args) -> int:
    from .aggregate import compare_runs
    try:
        out = compare_runs(args.out, args.inputs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {out}/comparison.md")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="traffic_replay")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw from a profile, print quantiles")
    s.add_argument("--profile", required=True)
    s.add_argument("--n", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("schedule", help="build a schedule, print its shape")
    s.add_argument("--duration", type=int, default=300)
    s.add_argument("--rate-scale", type=float, default=1.0)
    s.set_defaults(fn=cmd_schedule)

    s = sub.add_parser("run", help="replay against a real endpoint")
    s.add_argument("--config", required=True)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("validate", help="instrument self-test vs bundled mock")
    s.add_argument("--port", type=int, default=8808)
    s.add_argument("--duration", type=int, default=25)
    s.add_argument("--workdir", default="results/validation")
    s.add_argument("--tolerance-ms", type=float, default=60.0)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("merge", help="pool sharded run outputs into one")
    s.add_argument("out")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--profile", default=None,
                   help="profile whose acceptance_targets score the merge")
    s.add_argument("--title", default=None)
    s.add_argument("--force", action="store_true",
                   help="merge even if endpoint paths differ")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("compare", help="compare several runs side by side")
    s.add_argument("out")
    s.add_argument("inputs", nargs="+")
    s.set_defaults(fn=cmd_compare)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
