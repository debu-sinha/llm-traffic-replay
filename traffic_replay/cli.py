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
        # the run path stamps this; merge has to as well, or the scorecard
        # credits "the run configuration" for numbers out of the profile.
        if acceptance and "targets_are" not in acceptance:
            acceptance = {**acceptance, "targets_are": "this profile"}
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


def _pair(text, what):
    """Parse "10000" or "10000,24000" into a p50/p95 pair.

    A single value gets a p95 2.4x above it, which is roughly the spread of
    the agent traffic this was built for. Someone who knows their real p95
    passes both. Nobody should have to author a JSON file to say how big
    their prompts are.
    """
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        raise SystemExit(f"--{what} wants a number or two, got {text!r}")
    if not vals:
        raise SystemExit(f"--{what} is empty")
    if len(vals) > 2:
        raise SystemExit(f"--{what} takes p50 or p50,p95, got {text!r}")
    p50 = vals[0]
    frac = "rate" in what or "fraction" in what
    if len(vals) > 1:
        p95 = vals[1]
    elif frac:
        # a fraction has no room for a 2.4x tail. move it most of the way to
        # 1 instead, which is the shape a cache-reuse distribution actually
        # has, and keeps it a legal probability.
        p95 = p50 + (1.0 - p50) * 0.65
    else:
        p95 = p50 * 2.4
    if frac and not (0.0 <= p50 < p95 < 1.0):
        raise SystemExit(
            f"--{what} needs 0 <= p50 < p95 < 1, got {p50} and {p95}")
    if not frac and p95 <= p50:
        raise SystemExit(f"--{what} needs p95 above p50, got {p50} and {p95}")
    return {"p50": p50, "p95": p95}


def _preflight(cfg: dict) -> dict:
    """Send a couple of real requests and report what the endpoint does.

    This exists because the ways this tool produces a confidently wrong
    number are nearly all visible in two requests: auth that does not work,
    a model that spends its whole token budget reasoning, an endpoint that
    does not report usage, or one that does not report cached tokens. Better
    to find them in ten seconds than in a five minute run.
    """
    from .client import EndpointClient, EndpointConfig
    from .runner import _token
    from .textgen import TextMaterializer

    ecfg = EndpointConfig(**cfg["endpoint"])
    tok = _token(ecfg)
    client = EndpointClient(ecfg, tok)
    mat = TextMaterializer(cpt=4.0)
    ip = cfg["_input_tokens"]
    out: dict = {"auth": bool(tok)}
    rows = []
    for i in range(2):
        msgs = mat.messages(f"preflight{i}", i, int(ip["p50"]),
                            int(ip["p95"]), 200)
        res = client.send(msgs, 512, f"preflight-{i}", scheduled_s=0.0,
                          dispatch_lag_ms=0.0, intended=(0, 0, None, -1),
                          chars_sent=0)
        rows.append(res)
    ok = [r for r in rows if r.ok]
    out["reachable"] = len(ok)
    out["attempted"] = len(rows)
    if not ok:
        out["error"] = (rows[0].error or "no response")[:200]
        return out
    out["usage_reported"] = any(r.prompt_tokens for r in ok)
    out["cache_reported"] = any(r.cached_tokens is not None for r in ok)
    out["reasoning"] = any(r.reasoning_chunks for r in ok)
    out["visible"] = any(r.ttfv_ms is not None for r in ok)
    out["truncated"] = any(r.finish_reason == "length" for r in ok)
    return out


def cmd_benchmark(args) -> int:
    """One command from an endpoint URL to a report.

    The previous path was: author a profile JSON, run quickstart, edit the
    config, run it. Three of those four steps are things a person should not
    have to do to answer "does this endpoint meet my latency target".
    """
    from .runner import RunConfig, run

    path = args.endpoint
    if not path.startswith("/"):
        path = f"/serving-endpoints/{path}/invocations"
    ep: dict = {"base_url": args.host.rstrip("/"), "path": path}
    if args.auth_profile:
        ep["auth_profile"] = args.auth_profile
    else:
        ep["auth_token_env"] = args.token_env
    if args.model:
        ep["model"] = args.model
    if args.extra_body:
        try:
            ep["extra_body"] = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--extra-body is not valid JSON: {e}")

    cfg: dict = {
        "endpoint": ep,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "out_dir": args.out_dir,
        "title": args.title or f"{args.concurrency} concurrent, {args.endpoint}",
        "label": args.label or (
            "Describe the capacity this ran on. Shared pay-per-token is not "
            "a performance claim for a dedicated endpoint."),
    }

    inp = _pair(args.input_tokens, "input-tokens")
    outp = _pair(args.output_tokens, "output-tokens")
    # max_output_tokens_cap defaults to 512 and the per-request budget is the
    # smaller of it and the sampled value, so without this a run asking for
    # 2000 output tokens quietly got 512 and the preflight's advice to raise
    # --output-tokens did nothing.
    cfg["max_output_tokens_cap"] = max(int(outp["p95"] * 1.5), 512)
    if args.prompts:
        cfg["prompts_file"] = args.prompts
    elif args.profile:
        cfg["profile_path"] = args.profile
    else:
        prof = {
            "name": "from_command_line",
            "input_tokens": inp,
            "output_tokens": outp,
            "cache_fraction": _pair(args.cache_hit_rate, "cache-hit-rate"),
            "provenance": ("figures passed on the command line, not measured "
                           "from logs. build one from your own traffic with "
                           "scripts/profile_from_logs.py when you can."),
            "label": ("Traffic shape stated on the command line rather than "
                      "measured."),
        }
        pf = Path(args.out_dir) / "profile.json"
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps(prof, indent=2) + "\n")
        cfg["profile_path"] = str(pf)

    ttft = {q: v for q, v in (("p50", args.ttft_p50), ("p90", args.ttft_p90),
                              ("p95", args.ttft_p95), ("p99", args.ttft_p99))
            if v}
    ttfg = {q: v for q, v in (("p50", args.ttfg_p50), ("p90", args.ttfg_p90),
                              ("p95", args.ttfg_p95), ("p99", args.ttfg_p99))
            if v}
    if ttft or ttfg or args.success_rate:
        t: dict = {"targets_are": "yours, passed on the command line"}
        if ttft:
            t["ttft_ms"] = ttft
        if ttfg:
            t["ttfg_ms"] = ttfg
        if args.success_rate:
            t["success_rate"] = args.success_rate
        cfg["acceptance_targets"] = t

    cfg["_input_tokens"] = inp
    if not args.skip_preflight:
        print("[preflight] sending 2 requests to see what this endpoint does")
        pf_res = _preflight(cfg)
        if not pf_res.get("reachable"):
            print(f"[preflight] FAILED: {pf_res.get('error', 'no response')}")
            print("[preflight] check the host, the endpoint name and the "
                  "token before running a load test against it.")
            return 2
        print(f"[preflight] {pf_res['reachable']}/{pf_res['attempted']} "
              "responded")
        if not pf_res.get("usage_reported"):
            print("[preflight] WARNING: no token usage reported, so token "
                  "throughput and per-token cost will be blank")
        if not pf_res.get("cache_reported"):
            print("[preflight] note: no cached-token field, so achieved "
                  "cache cannot be reported and latency cannot be judged "
                  "against a cache target")
        if pf_res.get("reasoning"):
            print("[preflight] this is a REASONING model. it emits thinking "
                  "tokens before the answer, and they count against "
                  "max_tokens.")
            if not pf_res.get("visible"):
                print("[preflight] and it produced NO visible answer within "
                      "512 tokens. at your output budget it will produce "
                      "none either. raise --output-tokens, or turn reasoning "
                      "down with --extra-body, before trusting any latency "
                      "number from this endpoint.")
            if "ttft_definition" not in cfg:
                cfg["ttft_definition"] = "first_visible"
                print("[preflight] scoring TTFT on the first VISIBLE token, "
                      "which is what a user-facing SLA describes.")
    cfg.pop("_input_tokens", None)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    saved = Path(args.out_dir) / "run-config.json"
    saved.write_text(json.dumps(cfg, indent=2) + "\n")
    out = run(RunConfig(**cfg))
    print()
    print(f"config saved to {saved}, rerun it with:")
    print(f"  python3 -m traffic_replay run --config {saved}")
    return 0


def cmd_quickstart(args) -> int:
    """Write a run config from the few things a load test actually needs.

    Everything else has a default that works, or is derived at run time from
    the endpoint's measured service time. Nobody should have to compute an
    arrival rate to say "hold 30 in flight".
    """
    path = args.endpoint
    if not path.startswith("/"):
        path = f"/serving-endpoints/{path}/invocations"
    ep: dict = {"base_url": args.host.rstrip("/"), "path": path}
    if args.auth_profile:
        ep["auth_profile"] = args.auth_profile
    else:
        ep["auth_token_env"] = args.token_env
    if args.model:
        ep["model"] = args.model

    cfg: dict = {
        "profile_path": args.profile,
        "endpoint": ep,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "out_dir": args.out_dir,
        "title": args.title or f"{args.concurrency} concurrent, {args.endpoint}",
        "label": args.label or (
            "Describe the capacity this ran on. Shared pay-per-token is not a "
            "performance claim for a dedicated endpoint."),
    }
    if args.max_output_tokens:
        cfg["max_output_tokens_cap"] = args.max_output_tokens

    # SLA targets. the whole reason to run this is "do we meet ours", so it
    # has to be expressible here. without them the report falls back to the
    # profile's, which on a bundled profile are illustrative.
    ttft = {q: v for q, v in (("p50", args.ttft_p50), ("p90", args.ttft_p90),
                              ("p95", args.ttft_p95), ("p99", args.ttft_p99))
            if v}
    ttfg = {q: v for q, v in (("p50", args.ttfg_p50), ("p90", args.ttfg_p90),
                              ("p95", args.ttfg_p95), ("p99", args.ttfg_p99))
            if v}
    if ttft or ttfg or args.success_rate:
        targets: dict = {"targets_are": "yours, passed on the command line"}
        if ttft:
            targets["ttft_ms"] = ttft
        if ttfg:
            targets["ttfg_ms"] = ttfg
        if args.success_rate:
            targets["success_rate"] = args.success_rate
        cfg["acceptance_targets"] = targets

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {out}")
    print()
    print("run it with:")
    print(f"  python3 -m traffic_replay run --config {out}")
    print()
    print("the arrival rate and pool size are derived at run time from a short "
          "sizing pass, and printed before the replay starts.")
    if not args.auth_profile:
        print(f"export {args.token_env} first, or pass --auth-profile to read "
              "a ~/.databrickscfg profile instead.")
    if "acceptance_targets" not in cfg:
        print()
        print("no SLA targets given, so the scorecard will fall back to the "
              "profile's. pass --ttft-p95 and --ttfg-p95 (and the other "
              "quantiles) to score against yours.")
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

    s = sub.add_parser(
        "benchmark",
        help="one command: endpoint in, report out (start here)")
    s.add_argument("--host", required=True,
                   help="workspace URL, e.g. https://my-ws.cloud.databricks.com")
    s.add_argument("--endpoint", required=True,
                   help="endpoint name, or a full /serving-endpoints/... path")
    s.add_argument("--concurrency", type=int, default=10,
                   help="how many requests to hold in flight (default 10)")
    s.add_argument("--duration", type=int, default=300,
                   help="seconds. 300 gives five stability windows")
    s.add_argument("--input-tokens", default="10000",
                   help="prompt size as p50 or p50,p95. default 10000")
    s.add_argument("--output-tokens", default="200",
                   help="answer size as p50 or p50,p95. default 200")
    s.add_argument("--cache-hit-rate", default="0.3,0.7",
                   help="prompt-cache reuse as p50 or p50,p95, 0 to 1")
    s.add_argument("--prompts", default=None,
                   help="JSONL of your real prompts, instead of synthetic text")
    s.add_argument("--profile", default=None,
                   help="an existing profile JSON, instead of the flags above")
    s.add_argument("--auth-profile", default=None,
                   help="a ~/.databrickscfg profile name (PAT or OAuth)")
    s.add_argument("--token-env", default="DATABRICKS_TOKEN",
                   help="env var holding a bearer token, if not using a profile")
    s.add_argument("--model", default=None,
                   help="only for shared /chat/completions routes")
    s.add_argument("--extra-body", default=None,
                   help='JSON merged into each request, e.g. '
                        '\'{"reasoning_effort": "none"}\'')
    s.add_argument("--ttft-p50", type=float, default=None,
                   help="your TTFT target in ms. same for --ttft-p90/p95/p99")
    s.add_argument("--ttft-p90", type=float, default=None)
    s.add_argument("--ttft-p95", type=float, default=None)
    s.add_argument("--ttft-p99", type=float, default=None)
    s.add_argument("--ttfg-p50", type=float, default=None,
                   help="your full-generation target in ms")
    s.add_argument("--ttfg-p90", type=float, default=None)
    s.add_argument("--ttfg-p95", type=float, default=None)
    s.add_argument("--ttfg-p99", type=float, default=None)
    s.add_argument("--success-rate", type=float, default=None,
                   help="fraction 0-1, e.g. 0.99")
    s.add_argument("--out-dir", default="results/benchmark")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument("--skip-preflight", action="store_true",
                   help="skip the 2-request endpoint check. not recommended")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("quickstart",
                       help="write a run config from endpoint + concurrency")
    s.add_argument("--host", required=True,
                   help="workspace URL, e.g. https://my-ws.cloud.databricks.com")
    s.add_argument("--endpoint", required=True,
                   help="endpoint name, or a full /serving-endpoints/... path")
    s.add_argument("--profile", required=True,
                   help="traffic profile JSON describing your prompt shape")
    s.add_argument("--concurrency", type=int, required=True,
                   help="how many requests to hold in flight")
    s.add_argument("--duration", type=int, default=240,
                   help="seconds. 240 gives four stability windows")
    s.add_argument("--auth-profile", default=None,
                   help="a ~/.databrickscfg profile name (PAT or OAuth)")
    s.add_argument("--token-env", default="DATABRICKS_TOKEN",
                   help="env var holding a bearer token, if not using a profile")
    s.add_argument("--model", default=None,
                   help="only for shared /chat/completions routes")
    s.add_argument("--max-output-tokens", type=int, default=None)
    s.add_argument("--out-dir", default="results/quickstart")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument("--ttft-p50", type=float, default=None,
                   help="your TTFT target in ms. same for --ttft-p90/p95/p99")
    s.add_argument("--ttft-p90", type=float, default=None)
    s.add_argument("--ttft-p95", type=float, default=None)
    s.add_argument("--ttft-p99", type=float, default=None)
    s.add_argument("--ttfg-p50", type=float, default=None,
                   help="your full-generation target in ms")
    s.add_argument("--ttfg-p90", type=float, default=None)
    s.add_argument("--ttfg-p95", type=float, default=None)
    s.add_argument("--ttfg-p99", type=float, default=None)
    s.add_argument("--success-rate", type=float, default=None)
    s.add_argument("--out", default="configs/quickstart.json")
    s.set_defaults(fn=cmd_quickstart)

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
