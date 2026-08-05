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


_EXIT = {"ok": 0, "caution": 0, "miss": 1, "invalid": 2}


def _finish(out, fail_on: str = "miss", fmt: str = "text") -> int:
    """Print the result and turn the verdict into an exit code.

    Two things were wrong before. A run that missed every acceptance target
    exited 0, so the harness could not gate anything. And the default output
    was `json.dumps(summary)[:4000]`, which is a JSON document sliced mid
    structure, so the first thing a user saw was invalid JSON.
    """
    from .metrics import _verdict
    d = Path(out["out_dir"])
    if fmt == "json":
        print(json.dumps(out["summary"], indent=2))
    else:
        # report.md already says exactly this, and it is the artifact people
        # paste into email, so the terminal and the file cannot disagree.
        md = d / "report.md"
        if md.exists():
            print(md.read_text().rstrip())
    print()
    print(f"open in a browser: {d / 'report.html'}")
    print(f"full outputs:      {d}")

    kind, text = _verdict(out["summary"])
    code = _EXIT.get(kind, 0)
    if fail_on == "none":
        code = 0
    elif fail_on == "caution" and kind == "caution":
        code = 1
    print()
    print(f"{kind.upper()}: {text}")
    if code:
        print(f"exiting {code}. pass --fail-on none to always exit 0.")
    return code


def cmd_run(args) -> int:
    from .runner import RunConfig, run
    cfg = json.loads(Path(args.config).read_text())
    rc = RunConfig(**cfg)
    out = run(rc)
    return _finish(out, getattr(args, "fail_on", "miss"),
                   getattr(args, "format", "text"))


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
    import math
    if len(vals) > 2:
        raise SystemExit(f"--{what} takes p50 or p50,p95, got {text!r}")
    if any(not math.isfinite(v) for v in vals):
        raise SystemExit(f"--{what} needs finite numbers, got {text!r}")
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
    client = EndpointClient(ecfg, tok, refresh=lambda: _token(ecfg))
    mat = TextMaterializer(cpt=4.0)
    ip = cfg["_input_tokens"]
    # probe at the budget the run will actually use. probing at a fixed 512
    # and then stating what happens "at your output budget" was an
    # extrapolation presented as a measurement, in the one place a customer
    # decides whether to keep testing an endpoint.
    budget = int(cfg.get("max_output_tokens_cap") or 512)
    out: dict = {"auth": bool(tok), "budget": budget}
    rows = []
    for i in range(2):
        msgs = mat.messages(f"preflight{i}", i, int(ip["p50"]),
                            int(ip["p95"]), 200)
        res = client.send(msgs, budget, f"preflight-{i}", scheduled_s=0.0,
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


def _benchmark_config(args) -> dict:
    """Build a run config from the flags. Shared by benchmark and sweep, so
    the two cannot drift on how a profile or a target is interpreted."""
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

    # the per-request budget is min(sampled_output, max_output_tokens_cap),
    # and the cap defaults to 512, so a workload wanting more than that was
    # silently clipped. size the cap from whatever actually decides the
    # output distribution for THIS run, which is the given profile when one
    # was passed and the flags otherwise.
    _p95 = outp["p95"]
    if args.profile:
        try:
            _p95 = float(json.loads(Path(args.profile).read_text())
                         ["output_tokens"]["p95"])
        except Exception:
            pass
    if not args.prompts:
        cfg["max_output_tokens_cap"] = max(int(_p95 * 1.5), 512)

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
    return cfg


# the controls that turn reasoning down, in the order worth trying. every
# vendor spells this differently and several accept a flag and then ignore
# it, so the only way to know is to send one of each and look at what came
# back. measured: GLM-5.2 accepts reasoning_effort=none, Kimi K2.7 rejects
# that same value with "it is a thinking-only model" and wants minimal.
_REASONING_LEVERS = (
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=minimal", {"reasoning_effort": "minimal"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("thinking.type=disabled", {"thinking": {"type": "disabled"}}),
    ("enable_thinking=false", {"enable_thinking": False}),
)


def _probe_reasoning_levers(cfg: dict, budget: int) -> list[dict]:
    """Send one request per control and report what each one did.

    This runs only when the endpoint has already proven it produces no
    readable answer at the configured budget, which is a run the user cannot
    use. A handful of requests to turn "turn reasoning down somehow" into
    "use this exact flag" is a good trade at that point.

    The real prompt shape is used, not a short one. A one-line prompt gives
    a different and much rosier answer, which is a mistake worth not
    repeating.
    """
    import copy
    from .client import EndpointClient, EndpointConfig
    from .runner import _token
    from .textgen import TextMaterializer

    ip = cfg["_input_tokens"]
    mat = TextMaterializer(cpt=4.0)
    msgs = mat.messages("lever", 7, int(ip["p50"]), int(ip["p95"]), 200)
    out = []
    for name, extra in _REASONING_LEVERS:
        ec = copy.deepcopy(cfg["endpoint"])
        ec["extra_body"] = {**(ec.get("extra_body") or {}), **extra}
        ecfg = EndpointConfig(**ec)
        client = EndpointClient(ecfg, _token(ecfg))
        try:
            r = client.send(msgs, budget, f"lever-{name}", scheduled_s=0.0,
                            dispatch_lag_ms=0.0, intended=(0, 0, None, -1),
                            chars_sent=0)
        except Exception as e:  # never let a probe break the run
            out.append({"name": name, "extra": extra, "verdict": "error",
                        "detail": str(e)[:160]})
            continue
        if not r.ok:
            # a refusal is the most useful answer of all: it usually names
            # the reason, and it rules the flag out for good.
            out.append({"name": name, "extra": extra, "verdict": "rejected",
                        "detail": (r.error or "")[:220]})
        elif r.ttfv_ms is not None:
            out.append({"name": name, "extra": extra, "verdict": "works",
                        "detail": f"answered, finish {r.finish_reason}, "
                                  f"{r.completion_tokens} tokens"})
        else:
            out.append({"name": name, "extra": extra, "verdict": "ignored",
                        "detail": f"accepted, still no visible answer within "
                                  f"{budget} tokens"})
    return out


def _print_lever_report(levers: list[dict], budget: int) -> None:
    works = [x for x in levers if x["verdict"] == "works"]
    print("[preflight] trying the reasoning controls this endpoint might "
          "accept, one request each:")
    for x in levers:
        mark = {"works": "WORKS", "rejected": "rejected",
                "ignored": "ignored", "error": "error"}[x["verdict"]]
        print(f"[preflight]   {x['name']:24s} {mark:9s} {x['detail'][:96]}")
    if works:
        best = works[0]
        flag = json.dumps(best["extra"])
        print(f"[preflight] use this: --extra-body '{flag}'")
    else:
        print(f"[preflight] none of them produced an answer within {budget} "
              "tokens. this model needs a bigger output budget, or it is "
              "the wrong model for a budget this size. raise "
              "--output-tokens and re-run the preflight to find out which.")


def cmd_benchmark(args) -> int:
    """One command from an endpoint URL to a report.

    The previous path was: author a profile JSON, run quickstart, edit the
    config, run it. Three of those four steps are things a person should not
    have to do to answer "does this endpoint meet my latency target".
    """
    from .runner import RunConfig, run

    cfg = _benchmark_config(args)
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
                print(f"[preflight] and it produced NO visible answer within "
                      f"{pf_res['budget']} tokens, which is the budget this "
                      "run will use. raise --output-tokens, or turn "
                      "reasoning down, before trusting any latency number "
                      "from this endpoint.")
                if not args.no_lever_probe:
                    print()
                    _print_lever_report(
                        _probe_reasoning_levers(cfg, budget=512), 512)
                    print()
            if "ttft_definition" not in cfg:
                cfg["ttft_definition"] = "first_visible"
                print("[preflight] scoring TTFT on the first VISIBLE token, "
                      "which is what a user-facing SLA describes.")
    cfg.pop("_input_tokens", None)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    saved = Path(args.out_dir) / "run-config.json"
    saved.write_text(json.dumps(cfg, indent=2) + "\n")
    out = run(RunConfig(**cfg))
    code = _finish(out, getattr(args, "fail_on", "miss"),
                   getattr(args, "format", "text"))
    print()
    print(f"config saved to {saved}, rerun it with:")
    print(f"  python3 -m traffic_replay run --config {saved}")
    return code


def _rungs(spec: str) -> list[float]:
    """Parse "1:32" into a geometric ladder, or "2,5,10" into exactly those.

    Geometric rather than linear because the interesting region is
    multiplicative: the difference between 1 and 2 requests per second
    matters as much as the difference between 16 and 32, and a linear ladder
    spends most of its rungs past the knee.
    """
    spec = str(spec).strip()
    try:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            lo, hi = float(parts[0]), float(parts[1])
            n = int(parts[2]) if len(parts) == 3 else 6
            if not (0 < lo < hi) or n < 2:
                raise ValueError
            step = (hi / lo) ** (1.0 / (n - 1))
            return [round(lo * step ** i, 3) for i in range(n)]
        vals = [float(x) for x in spec.split(",") if x.strip()]
        if not vals or any(v <= 0 for v in vals):
            raise ValueError
        return sorted(vals)
    except ValueError:
        raise SystemExit(
            f"--rate wants lo:hi, lo:hi:rungs, or a comma list, got {spec!r}")


def cmd_sweep(args) -> int:
    """Climb a rate ladder and report the highest rung that stayed valid.

    The axis is arrival rate, not concurrency, and that is a deliberate
    choice rather than a convenience. An open-loop generator cannot hold a
    concurrency: Little's law says in-flight is arrival rate times service
    time, and service time rises under load, so fixing the rate means the
    concurrency moves. Every sweep in this category picks a concurrency axis
    because it is closed loop underneath, and pays for it with coordinated
    omission. We offer a rate, which is the thing we actually control, and
    report the concurrency each rung turned out to hold, which is the thing
    the customer wants to hear back.
    """
    import copy
    import time as _time
    from .metrics import _verdict
    from .runner import RunConfig, run

    rates = _rungs(args.rate)
    base = _benchmark_config(args)
    # the ladder sets its own rate on every rung, and _input_tokens is a
    # preflight-only key that RunConfig does not accept.
    base.pop("concurrency", None)
    base.pop("_input_tokens", None)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {len(rates)} rungs: "
          + ", ".join(f"{r:g}" for r in rates) + " requests/second")
    print(f"[sweep] {args.duration}s each"
          + (f", {args.cooldown}s cooldown between them"
             if args.cooldown else ""))
    print()

    rungs: list[dict] = []
    for i, rate in enumerate(rates):
        cfg = copy.deepcopy(base)
        cfg.update(qps_base=rate, qps_burst=rate, qps_min=rate,
                   qps_max=rate, rate_scale=1.0,
                   duration_s=args.duration,
                   out_dir=str(out_root / f"rate_{rate:g}"),
                   title=f"{rate:g} requests/second")
        # the pool has to be able to hold what the rate implies, or the
        # client becomes the bottleneck and measures itself.
        cfg["max_concurrency"] = max(64, int(rate * 30))
        print(f"[sweep] rung {i + 1}/{len(rates)}: {rate:g} rps")
        out = run(RunConfig(**cfg), quiet=False)
        kind, text = _verdict(out["summary"])
        s = out["summary"]
        cc = s.get("concurrency") or {}
        rungs.append({
            "rate": rate, "kind": kind, "text": text,
            "dir": out["out_dir"],
            "held": cc.get("in_flight_p50"),
            "achieved_rps": (s.get("arrivals") or {}).get(
                "achieved_qps_overall"),
            "err": s.get("error_rate"),
            "ttft_p50": (s.get("ttft_ms") or {}).get("p50"),
            "ttft_p95": (s.get("ttft_ms") or {}).get("p95"),
            "e2e_p50": (s.get("e2e_ms") or {}).get("p50"),
        })
        print(f"[sweep] rung {i + 1}: {kind.upper()} {text[:90]}")
        print()
        if kind in ("miss", "invalid") and not args.no_early_stop:
            print(f"[sweep] stopping: rung {i + 1} did not hold. pass "
                  "--no-early-stop to climb the whole ladder anyway.")
            break
        if args.cooldown and i + 1 < len(rates):
            _time.sleep(args.cooldown)

    return _sweep_report(rungs, out_root, args)


def _sweep_report(rungs: list[dict], out_root: Path, args) -> int:
    """One table, and one sentence naming the highest rung that held."""
    def _n(v, d=0):
        return "-" if v is None else f"{v:,.{d}f}"

    hdr = ("| rate asked | achieved | held | error | TTFT p50 | TTFT p95 "
           "| E2E p50 | verdict |")
    rows = [hdr, "|---|---|---|---|---|---|---|---|"]
    for r in rungs:
        rows.append(
            f"| {r['rate']:g} rps | {_n(r['achieved_rps'], 1)} | "
            f"{_n(r['held'])} | {(r['err'] or 0):.1%} | "
            f"{_n(r['ttft_p50'])} | {_n(r['ttft_p95'])} | "
            f"{_n(r['e2e_p50'])} | {r['kind'].upper()} |")

    # the ceiling is the highest rung that STAYED VALID, never the highest
    # one we managed to submit. every sweep in this category anchors on the
    # latter and reports a top rung its own error rate disqualifies.
    good = [r for r in rungs if r["kind"] in ("ok", "caution")]
    if good:
        best = good[-1]
        head = (f"Highest rate that held: {best['rate']:g} requests/second, "
                f"which carried about {_n(best['held'])} concurrent.")
        def _sentence(t: str) -> str:
            t = t.strip()
            return t if t.endswith(".") else t + "."

        if best["kind"] == "caution":
            head += " Read it with care: " + _sentence(best["text"])
        nxt = next((r for r in rungs if r["rate"] > best["rate"]), None)
        if nxt:
            head += (f" The next rung, {nxt['rate']:g} rps, "
                     f"{nxt['kind']}ed: " + _sentence(nxt["text"]))
        else:
            head += (" That was the top of the ladder, so the real ceiling "
                     "may be higher. Raise --rate to find it.")
    else:
        _t = rungs[0]["text"].strip()
        head = ("No rung held. The lowest rate tested "
                f"({rungs[0]['rate']:g} rps) already "
                f"{rungs[0]['kind']}ed: "
                + (_t if _t.endswith(".") else _t + "."))

    body = "\n".join([f"# Rate ladder: {args.endpoint}",
                      "", head, "",
                      "\n".join(rows), "",
                      "The axis is arrival rate because that is what an "
                      "open-loop generator controls. Concurrency is reported "
                      "as measured, not as asked for: in-flight is arrival "
                      "rate times service time, and service time rises under "
                      "load, so it is an outcome rather than an input.", "",
                      "Per-rung reports:", ""]
                     + [f"- {r['rate']:g} rps: `{r['dir']}/report.html`"
                        for r in rungs])
    path = out_root / "sweep.md"
    path.write_text(body + "\n")
    print()
    print(body)
    print()
    print(f"written to {path}")
    return 0 if good else 1


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
    s.add_argument("--no-lever-probe", action="store_true",
                   help="skip trying reasoning controls when the endpoint "
                        "produces no readable answer")
    s.add_argument("--fail-on", choices=("none", "miss", "caution"),
                   default="miss",
                   help="exit non-zero on this verdict or worse. miss=1, "
                        "invalid=2. use none to always exit 0")
    s.add_argument("--format", choices=("text", "json"), default="text",
                   help="text prints the report, json prints summary.json")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser(
        "sweep",
        help="climb a rate ladder and report the highest rate that held")
    s.add_argument("--host", required=True)
    s.add_argument("--endpoint", required=True)
    s.add_argument("--rate", default="1:32",
                   help="lo:hi, lo:hi:rungs, or a comma list. requests per "
                        "second. geometric by default, since the interesting "
                        "region is multiplicative")
    s.add_argument("--duration", type=int, default=120,
                   help="seconds per rung")
    s.add_argument("--cooldown", type=int, default=30,
                   help="seconds between rungs, so a sliding request quota "
                        "refills and each rung starts from the same place")
    s.add_argument("--no-early-stop", action="store_true",
                   help="climb every rung even after one fails")
    s.add_argument("--input-tokens", default="10000")
    s.add_argument("--output-tokens", default="200")
    s.add_argument("--cache-hit-rate", default="0.3,0.7")
    s.add_argument("--prompts", default=None)
    s.add_argument("--profile", default=None)
    s.add_argument("--auth-profile", default=None)
    s.add_argument("--token-env", default="DATABRICKS_TOKEN")
    s.add_argument("--model", default=None)
    s.add_argument("--extra-body", default=None)
    s.add_argument("--ttft-p50", type=float, default=None)
    s.add_argument("--ttft-p90", type=float, default=None)
    s.add_argument("--ttft-p95", type=float, default=None)
    s.add_argument("--ttft-p99", type=float, default=None)
    s.add_argument("--ttfg-p50", type=float, default=None)
    s.add_argument("--ttfg-p90", type=float, default=None)
    s.add_argument("--ttfg-p95", type=float, default=None)
    s.add_argument("--ttfg-p99", type=float, default=None)
    s.add_argument("--success-rate", type=float, default=None)
    s.add_argument("--out-dir", default="results/sweep")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument("--concurrency", type=int, default=None,
                   help=argparse.SUPPRESS)
    s.add_argument("--skip-preflight", action="store_true",
                   default=True, help=argparse.SUPPRESS)
    s.add_argument("--no-lever-probe", action="store_true",
                   default=True, help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_sweep)

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
    s.add_argument("--fail-on", choices=("none", "miss", "caution"),
                   default="miss",
                   help="exit non-zero on this verdict or worse. miss=1, "
                        "invalid=2. use none to always exit 0")
    s.add_argument("--format", choices=("text", "json"), default="text",
                   help="text prints the report, json prints summary.json")
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
