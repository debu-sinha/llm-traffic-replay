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
    kind, text = _verdict(out["summary"])
    # An unknown verdict is an invalid result, never a successful gate.
    code = _EXIT.get(kind, _EXIT["invalid"])
    if fail_on == "none":
        code = 0
    elif fail_on == "caution" and kind == "caution":
        code = 1

    if fmt == "json":
        # stdout is a single standards-compliant JSON document so automation
        # can parse it. Human navigation and verdict text belong to text mode.
        print(json.dumps(out["summary"], indent=2, allow_nan=False))
        return code
    else:
        # report.md already says exactly this, and it is the artifact people
        # paste into email, so the terminal and the file cannot disagree.
        md = d / "report.md"
        if md.exists():
            print(md.read_text().rstrip())
    print()
    print(f"open in a browser: {d / 'report.html'}")
    print(f"full outputs:      {d}")

    print()
    print(f"{kind.upper()}: {text}")
    if code:
        print(f"exiting {code}. pass --fail-on none to always exit 0.")
    return code


def cmd_run(args) -> int:
    from .json_input import loads_strict
    from .runner import RunConfig, run
    cfg = loads_strict(Path(args.config).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("run config JSON must be an object")
    if cfg.get("concurrency") is not None and cfg.get("sizing_concurrency") is None:
        print("warning: config field 'concurrency' is legacy; it is treated as "
              "'sizing_concurrency', which derives a fixed open-loop rate and "
              "does not hold concurrency.", file=sys.stderr)
    rc = RunConfig(**cfg)
    out = run(rc, quiet=getattr(args, "format", "text") == "json")
    return _finish(out, getattr(args, "fail_on", "miss"),
                   getattr(args, "format", "text"))


def _validation_error_stats(values) -> dict:
    """Signed and absolute measurement-oracle error percentiles."""
    import numpy as np
    absolute = np.abs(values)
    return {"p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "absolute_p95": float(np.percentile(absolute, 95)),
            "absolute_max": float(np.max(absolute))}


def _validation_passes(report: dict, tolerance_ms: float) -> bool:
    """Both TTFT and E2E clocks must agree with the oracle in magnitude."""
    return all(report[name]["absolute_p95"] <= tolerance_ms
               for name in ("ttft_error_ms", "e2e_error_ms"))


def cmd_validate(args) -> int:
    """Instrument self-test: run the whole pipeline against the bundled mock
    and report client-measured vs server-true latency error."""
    import numpy as np
    from importlib.resources import files
    from .mock_server import serve
    from .runner import RunConfig, run

    import math
    if isinstance(args.tolerance_ms, bool) \
            or not isinstance(args.tolerance_ms, (int, float)) \
            or not math.isfinite(float(args.tolerance_ms)) \
            or args.tolerance_ms <= 0:
        raise SystemExit("--tolerance-ms must be positive and finite")

    truth = Path(args.workdir) / "mock_truth.jsonl"
    srv = serve(args.port, truth)
    # Port zero asks the OS for a collision-free ephemeral port. The client
    # must use the assigned port, not literal port 0 (which means port 80 in
    # an HTTP URL parser).
    port = int(srv.server_address[1])
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        rc = RunConfig(
            profile_path=str(files("traffic_replay").joinpath(
                "data/profile_validation_small.json")),
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
        out = run(rc, quiet=(args.quiet or
                             getattr(args, "format", "text") == "json"))
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5.0)

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
        if getattr(args, "format", "text") == "json":
            print(json.dumps({"passed": False, "joined_requests": 0,
                              "error": "no joinable rows"},
                             allow_nan=False))
        else:
            print("VALIDATE: no joinable rows, FAIL")
        return 1
    a = np.array(rows)
    ttft_err = a[:, 0] - a[:, 1]
    e2e_err = a[:, 2] - a[:, 3]
    rep = {
        "joined_requests": len(rows),
        "ttft_error_ms": _validation_error_stats(ttft_err),
        "e2e_error_ms": _validation_error_stats(e2e_err),
        "tolerance_ms": float(args.tolerance_ms),
        "note": "error = client-measured minus server-true; includes real "
                "localhost network+parse overhead, so small positive is "
                "expected and honest",
    }
    # the verdict is the point of this command. dumping the full report
    # above it buried the answer under 16 lines of JSON, which is what a
    # first-time user meets on step one of the guide.
    ok = _validation_passes(rep, args.tolerance_ms)
    rep["passed"] = ok
    if getattr(args, "format", "text") == "json":
        print(json.dumps(rep, indent=2, allow_nan=False))
    else:
        print(f"VALIDATE: {'PASS' if ok else 'FAIL'} "
              f"(absolute error p95: TTFT "
              f"{rep['ttft_error_ms']['absolute_p95']:.1f} ms, E2E "
              f"{rep['e2e_error_ms']['absolute_p95']:.1f} ms; "
              f"tolerance {args.tolerance_ms:g} ms)")
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
    raw_parts = str(text).split(",")
    if any(not x.strip() for x in raw_parts):
        raise SystemExit(
            f"--{what} wants one number or a p50,p95 pair, got {text!r}")
    parts = [x.strip() for x in raw_parts]
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
        p95 = (p50 if p50 in (0.0, 1.0)
               else p50 + (1.0 - p50) * 0.65)
    else:
        p95 = p50 * 2.4
    if frac and not (0.0 <= p50 <= p95 <= 1.0):
        raise SystemExit(
            f"--{what} needs 0 <= p50 <= p95 <= 1, got {p50} and {p95}")
    if not frac and not (p95 >= p50 > 0):
        raise SystemExit(f"--{what} needs p95 above p50 (or equal for a "
                         f"constant) and p50 > 0, got {p50} and {p95}")
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
    from .runner import RunConfig, _representative_plans, _token

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    rc = RunConfig(**clean)
    ecfg = EndpointConfig(**rc.endpoint)
    tok = _token(ecfg)
    client = EndpointClient(ecfg, tok, refresh=lambda: _token(ecfg))
    plans = _representative_plans(rc)
    out: dict = {"auth": bool(tok),
                 "budgets": [p["max_output"] for p in plans],
                 "representatives": [p["representative"] for p in plans]}
    rows = []
    for plan in plans:
        try:
            res = client.send(
                plan["messages"], plan["max_output"], plan["request_id"],
                scheduled_s=0.0, dispatch_lag_ms=0.0,
                intended=plan["intended"], chars_sent=plan["chars"])
            rows.append(res)
        except Exception as exc:
            rows.append(exc)
    reached = [r for r in rows
               if not isinstance(r, Exception) and r.status == 200]
    out["reachable"] = len(reached)
    out["attempted"] = len(rows)
    if not reached:
        first = rows[0]
        out["error"] = ((str(first) if isinstance(first, Exception)
                         else first.error) or "no response")[:200]
        return out
    out["usage_reported"] = all(r.prompt_tokens is not None for r in reached)
    out["cache_reported"] = all(r.cached_tokens is not None for r in reached)
    out["reasoning"] = any(r.reasoning_seen or r.reasoning_chunks for r in reached)
    readable = [_answer_is_complete(r) for r in reached]
    out["readable"] = sum(readable)
    out["visible"] = (len(reached) == len(rows)
                      and all(r.visible_content_seen for r in reached))
    out["tool_call_answers"] = sum(
        1 for r in reached if getattr(r, "valid_tool_calls", 0))
    out["truncated"] = any(r.finish_reason == "length" for r in reached)
    failed_index = next((i for i, r in enumerate(rows)
                         if isinstance(r, Exception) or r.status != 200
                         or not _answer_is_complete(r)), None)
    if failed_index is not None:
        out["failed_probe_index"] = failed_index
    budget_index = failed_index if failed_index is not None else len(plans) - 1
    out["budget"] = plans[budget_index]["max_output"]
    return out


def _benchmark_config(args) -> dict:
    """Build a run config from the flags. Shared by benchmark and sweep, so
    the two cannot drift on how a profile or a target is interpreted."""
    if args.prompts and args.profile:
        raise SystemExit("set --prompts or --profile, not both")
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
            from .json_input import loads_strict
            ep["extra_body"] = loads_strict(args.extra_body)
        except (json.JSONDecodeError, ValueError) as e:
            raise SystemExit(f"--extra-body is not valid JSON: {e}")
        if not isinstance(ep["extra_body"], dict):
            raise SystemExit("--extra-body must be a JSON object")
        try:
            from .client import validate_extra_body_safety
            validate_extra_body_safety(ep["extra_body"])
        except ValueError as exc:
            raise SystemExit(f"invalid --extra-body: {exc}") from exc

    sizing = getattr(args, "sizing_concurrency", None)
    legacy = (getattr(args, "legacy_concurrency", None)
              if hasattr(args, "legacy_concurrency")
              else getattr(args, "concurrency", None))
    if sizing is not None and legacy is not None:
        raise SystemExit("use --sizing-concurrency or legacy --concurrency, not both")
    if legacy is not None:
        print("warning: --concurrency is now --sizing-concurrency. it derives "
              "one fixed open-loop rate; it does not hold concurrency.",
              file=sys.stderr)
        sizing = legacy
    if sizing is None and getattr(args, "cmd", "benchmark") == "benchmark":
        sizing = 10

    default_title = (f"open-loop rate sized from {sizing} concurrent, "
                     f"{args.endpoint}" if sizing is not None
                     else f"fixed-rate workload, {args.endpoint}")
    cfg: dict = {
        "endpoint": ep,
        "sizing_concurrency": sizing,
        "duration_s": args.duration,
        "out_dir": args.out_dir,
        "title": args.title or default_title,
        "label": args.label or (
            "Describe the capacity this ran on. Shared pay-per-token is not "
            "a performance claim for a dedicated endpoint."),
    }
    if getattr(args, "max_concurrency", None) is not None:
        cfg["max_concurrency"] = args.max_concurrency
    if getattr(args, "max_pending_requests", None) is not None:
        cfg["max_pending_requests"] = args.max_pending_requests

    inp = _pair(args.input_tokens, "input-tokens")
    outp = _pair(args.output_tokens, "output-tokens")
    if args.prompts:
        try:
            from .prompts import load_prompts
            load_prompts(args.prompts)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid --prompts {args.prompts!r}: {exc}")
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
            from .profile import Profile
            _p95 = float(Profile.from_json(args.profile).output_tokens["p95"])
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            raise SystemExit(f"invalid --profile {args.profile!r}: {exc}")
    # Keep enough headroom above p95 that the cap is a safety guard rather
    # than the distribution itself. There is deliberately no hidden 512-token
    # floor: preflight and replay must use the workload's configured budget.
    import math
    cfg["max_output_tokens_cap"] = max(1, int(math.ceil(_p95 * 1.5)))

    ttft = {q: v for q, v in (("p50", args.ttft_p50), ("p90", args.ttft_p90),
                              ("p95", args.ttft_p95), ("p99", args.ttft_p99))
            if v is not None}
    ttfg = {q: v for q, v in (("p50", args.ttfg_p50), ("p90", args.ttfg_p90),
                              ("p95", args.ttfg_p95), ("p99", args.ttfg_p99))
            if v is not None}
    for name, targets in (("ttft", ttft), ("ttfg", ttfg)):
        if any(not math.isfinite(float(v)) or float(v) <= 0
               for v in targets.values()):
            raise SystemExit(f"--{name} targets must be positive and finite")
    if args.success_rate is not None and (
            not math.isfinite(float(args.success_rate))
            or not (0 < args.success_rate <= 1)):
        raise SystemExit("--success-rate must be in (0, 1]")
    if ttft or ttfg or args.success_rate is not None:
        t: dict = {"targets_are": "yours, passed on the command line"}
        if ttft:
            t["ttft_ms"] = ttft
        if ttfg:
            t["ttfg_ms"] = ttfg
        if args.success_rate is not None:
            t["success_rate"] = args.success_rate
        cfg["acceptance_targets"] = t
    return cfg


def _json_object_arg(value: str) -> dict:
    """Parse one finite JSON object before any endpoint traffic is sent."""
    try:
        from .json_input import loads_strict
        parsed = loads_strict(value)
        if not isinstance(parsed, dict):
            raise ValueError("value is not an object")
        json.dumps(parsed, allow_nan=False)
        from .client import validate_extra_body_safety
        validate_extra_body_safety(parsed)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected a finite JSON object, got {value!r}: {exc}") from exc
    return parsed


def _probe_label(extra: dict, position: int) -> str:
    """Give a candidate a stable label without echoing request-body values."""
    keys = ",".join(sorted(str(key) for key in extra))
    return f"candidate {position} ({keys[:72] or 'empty object'})"


def _safe_probe_detail(value: object) -> str:
    from .artifacts import redact_secrets
    return str(redact_secrets(str(value)))


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _answer_is_complete(result) -> bool:
    """A completed answer may be visible text or valid structured tool use."""
    return bool(result.stream_complete and not result.parse_errors
                and (result.visible_content_seen
                     or (getattr(result, "valid_tool_calls", 0) or 0) > 0))


def _probe_reasoning_levers(cfg: dict, budget: int,
                            candidates: list[dict],
                            probe_index: int = 1) -> list[dict]:
    """Send one request per user-supplied control and report what each did.

    This runs only when the endpoint has already proven it produces no
    readable answer at the configured budget. The harness does not guess
    provider fields or values: candidates must come from the target's current
    documentation or an explicitly authorized experiment.

    The real prompt shape is used, not a short one. A one-line prompt gives
    a different and much rosier answer, which is a mistake worth not
    repeating.
    """
    import copy
    from .client import EndpointClient, EndpointConfig
    from .runner import RunConfig, _representative_plans, _token

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    rc = RunConfig(**clean)
    plans = _representative_plans(rc)
    plan = plans[min(max(probe_index, 0), len(plans) - 1)]
    # The caller passes the exact failed budget. Keep it explicit so a future
    # refactor cannot reintroduce a probe-only 512-token floor.
    budget = int(budget)
    out = []
    for position, extra in enumerate(candidates, start=1):
        name = _probe_label(extra, position)
        ec = copy.deepcopy(cfg["endpoint"])
        ec["extra_body"] = _deep_merge(ec.get("extra_body") or {}, extra)
        ecfg = EndpointConfig(**ec)
        client = EndpointClient(ecfg, _token(ecfg),
                                refresh=lambda: _token(ecfg))
        try:
            r = client.send(
                plan["messages"], budget, f"lever-{name}", scheduled_s=0.0,
                dispatch_lag_ms=0.0, intended=plan["intended"],
                chars_sent=plan["chars"])
        except Exception as e:  # never let a probe break the run
            out.append({"name": name, "extra": extra, "verdict": "error",
                        "detail": _safe_probe_detail(e)[:160]})
            continue
        if r.status != 200:
            # a refusal is the most useful answer of all: it usually names
            # the reason, and it rules the flag out for good.
            out.append({"name": name, "extra": extra, "verdict": "rejected",
                        "detail": _safe_probe_detail(r.error or "")[:220]})
        elif _answer_is_complete(r):
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
    print("[preflight] trying the supplied reasoning-control candidates, "
          "one request each:")
    for x in levers:
        mark = {"works": "WORKS", "rejected": "rejected",
                "ignored": "ignored", "error": "error"}[x["verdict"]]
        print(f"[preflight]   {x['name']:24s} {mark:9s} {x['detail'][:96]}")
    if works:
        best = works[0]
        from .artifacts import redact_secrets
        flag = json.dumps(redact_secrets(best["extra"]))
        print(f"[preflight] use this: --extra-body '{flag}'")
    else:
        print(f"[preflight] none of the supplied candidates produced an "
              f"answer within {budget} "
              "tokens. this model needs a bigger output budget, or it is "
              "the wrong model for a budget this size. raise "
              "--output-tokens and re-run the preflight to find out which.")


def _refuse(levers: list[dict], args) -> int:
    """Stop before a run we have already shown will produce nothing.

    Found by following our own guide as a new user: the preflight said the
    model could not answer at the configured budget, printed the exact flag
    that fixes it, and then ran the full five minute test anyway. It came
    back INVALID with 1,872 requests and zero readable answers. Knowing the
    answer and spending the money anyway is the worst of both.
    """
    works = [x for x in levers if x["verdict"] == "works"]
    print("[preflight] STOPPING before the load starts. this run would have "
          "produced no readable answers, so it would cost you time and "
          "tokens for a verdict we can already give you.")
    print()
    if works:
        from .artifacts import redact_secrets
        flag = json.dumps(redact_secrets(works[0]["extra"]))
        print("  re-run with the control that worked:")
        print()
        print(f"    --extra-body '{flag}'")
    elif levers:
        print("  no supplied reasoning-control candidate helped at this budget.")
        print("  verify the exact model/provider contract, raise --output-tokens,")
        print("  or choose a model that fits this output budget.")
    else:
        print("  no reasoning controls were probed. configure a control documented")
        print("  by this exact model/provider with --extra-body, or explicitly test")
        print("  candidates with --probe-extra-body. alternatively, raise")
        print("  --output-tokens or choose a model that fits this budget.")
    print()
    print("  or pass --force to run it anyway and see the INVALID report.")
    return 3


def _check_preflight(cfg: dict, args) -> int | None:
    """Run the shared benchmark/sweep gate; return an exit code on refusal."""
    print("[preflight] sending 2 representative workload requests")
    pf_res = _preflight(cfg)
    if pf_res.get("reachable") != pf_res.get("attempted"):
        print(f"[preflight] FAILED: {pf_res.get('reachable', 0)}/"
              f"{pf_res.get('attempted', 2)} reached HTTP 200: "
              f"{pf_res.get('error', 'one or more requests failed')}")
        print("[preflight] check the host, endpoint, token and workload "
              "before running a load test.")
        return 2
    print(f"[preflight] {pf_res['reachable']}/{pf_res['attempted']} "
          "reached HTTP 200 at effective budgets "
          + ", ".join(str(x) for x in pf_res["budgets"]))
    if not pf_res.get("usage_reported"):
        print("[preflight] WARNING: at least one response reported no token "
              "usage, so throughput and per-token cost may be incomplete")
    if not pf_res.get("cache_reported"):
        print("[preflight] note: at least one response had no cached-token "
              "field, so achieved cache coverage may be incomplete")
    if pf_res.get("reasoning"):
        print("[preflight] this endpoint emitted reasoning-channel content; "
              "those tokens count against max_tokens.")
        if "ttft_definition" not in cfg:
            cfg["ttft_definition"] = "first_visible"
            print("[preflight] scoring TTFT on the first VISIBLE token.")

    if pf_res.get("readable") != pf_res.get("attempted"):
        print(f"[preflight] only {pf_res.get('readable', 0)}/"
              f"{pf_res['attempted']} produced a valid completed answer. "
              "This gate accepts visible content or a structurally valid tool "
              "call, plus clean stream completion.")
        levers: list[dict] = []
        candidates = list(getattr(args, "probe_extra_body", None) or [])
        if candidates:
            print()
            levers = _probe_reasoning_levers(
                cfg, budget=pf_res["budget"], candidates=candidates,
                probe_index=pf_res.get("failed_probe_index", 1))
            _print_lever_report(levers, pf_res["budget"])
            print()
        else:
            print("[preflight] no provider controls were guessed. pass a "
                  "model-documented control with --extra-body, or opt in to "
                  "specific candidates with --probe-extra-body.")
        if not getattr(args, "force", False):
            return _refuse(levers, args)
    return None


def cmd_benchmark(args) -> int:
    """One command from an endpoint URL to a report.

    The previous path was: author a profile JSON, run quickstart, edit the
    config, run it. Three of those four steps are things a person should not
    have to do to answer "does this endpoint meet my latency target".
    """
    from .runner import RunConfig, run

    cfg = _benchmark_config(args)
    json_mode = getattr(args, "format", "text") == "json"
    if not args.skip_preflight:
        if json_mode:
            import contextlib
            with contextlib.redirect_stdout(sys.stderr):
                refused = _check_preflight(cfg, args)
        else:
            refused = _check_preflight(cfg, args)
        if refused is not None:
            if json_mode:
                print(json.dumps({"passed": False,
                                  "stage": "preflight",
                                  "exit_code": refused},
                                 allow_nan=False))
            return refused

    # Validate the final (possibly preflight-adjusted) configuration before
    # writing a rerun file or starting the measured workload.
    rc = RunConfig(**cfg)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    saved = Path(args.out_dir) / "run-config.json"
    saved.write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
    out = run(rc, quiet=json_mode)
    code = _finish(out, getattr(args, "fail_on", "miss"),
                   getattr(args, "format", "text"))
    stream = sys.stderr if json_mode else sys.stdout
    print(file=stream)
    print(f"config saved to {saved}, rerun it with:", file=stream)
    print(f"  python3 -m traffic_replay run --config {saved}", file=stream)
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
        import math
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            lo, hi = float(parts[0]), float(parts[1])
            n = int(parts[2]) if len(parts) == 3 else 6
            if not (math.isfinite(lo) and math.isfinite(hi)
                    and 0 < lo < hi) or n < 2:
                raise ValueError
            step = (hi / lo) ** (1.0 / (n - 1))
            vals = [round(lo * step ** i, 3) for i in range(n)]
            if len(set(vals)) != len(vals):
                raise ValueError
            return vals
        raw = spec.split(",")
        if any(not x.strip() for x in raw):
            raise ValueError
        vals = [float(x) for x in raw]
        if (not vals or any(not math.isfinite(v) or v <= 0 for v in vals)
                or len(set(vals)) != len(vals)):
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

    if args.duration <= 0:
        raise SystemExit("--duration must be a positive number of seconds")
    if args.cooldown < 0:
        raise SystemExit("--cooldown cannot be negative")
    rates = _rungs(args.rate)
    base = _benchmark_config(args)
    # The ladder controls arrival rate directly. It must not also run the
    # unloaded concurrency-sizing pass, which would overwrite every rung.
    base.pop("sizing_concurrency", None)
    # Validate the complete fixed-rate config before persisting sweep configs.
    # extra_body is reproducibility evidence, not a credential channel.
    RunConfig(**base)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_preflight:
        refused = _check_preflight(base, args)
        if refused is not None:
            return refused
    (out_root / "sweep-base-config.json").write_text(
        json.dumps(base, indent=2) + "\n")

    nominal = (len(rates) * args.duration
               + max(0, len(rates) - 1) * args.cooldown)
    print(f"[sweep] {len(rates)} rungs: "
          + ", ".join(f"{r:g}" for r in rates) + " requests/second")
    print(f"[sweep] {args.duration}s of offered load each"
          + (f", {args.cooldown}s cooldown between them"
             if args.cooldown else ""))
    print(f"[sweep] nominal scheduled time {nominal}s if every rung runs; "
          "preflight, calibration, response drain and report writing are extra")
    print()

    rungs: list[dict] = []
    sweep_started = _time.monotonic()
    for i, rate in enumerate(rates):
        cfg = copy.deepcopy(base)
        cfg.update(qps_base=rate, qps_burst=rate, qps_min=rate,
                   qps_max=rate, rate_scale=1.0,
                   duration_s=args.duration,
                   out_dir=str(out_root / f"rate_{rate:g}"),
                   title=(args.title or f"{args.endpoint} rate sweep")
                         + f" @ {rate:g} requests/second")
        rung_rc = RunConfig(**cfg)
        rung_root = Path(cfg["out_dir"])
        rung_root.mkdir(parents=True, exist_ok=True)
        (rung_root / "run-config.json").write_text(
            json.dumps(cfg, indent=2) + "\n")
        print(f"[sweep] rung {i + 1}/{len(rates)}: {rate:g} rps")
        rung_started = _time.monotonic()
        try:
            out = run(rung_rc, quiet=False)
            kind, verdict_text = _verdict(out["summary"])
            s = out["summary"]
            out_dir = out["out_dir"]
        except Exception as exc:
            kind = "invalid"
            verdict_text = (f"rung failed before a report: "
                            f"{type(exc).__name__}: {exc}")
            s = {}
            out_dir = str(rung_root)
        cc = s.get("concurrency") or {}
        rungs.append({
            "rate": rate, "kind": kind, "text": verdict_text,
            "dir": out_dir,
            "held": cc.get("in_flight_p50"),
            "achieved_rps": (s.get("arrivals") or {}).get(
                "achieved_qps_overall"),
            "err": s.get("error_rate"),
            "ttft_p50": (s.get("ttft_ms") or {}).get("p50"),
            "ttft_p95": (s.get("ttft_ms") or {}).get("p95"),
            "e2e_p50": (s.get("e2e_ms") or {}).get("p50"),
            "wall_s": _time.monotonic() - rung_started,
        })
        print(f"[sweep] rung {i + 1}: {kind.upper()} {verdict_text[:90]}")
        print()
        if kind != "ok" and not args.no_early_stop:
            print(f"[sweep] stopping: rung {i + 1} was not an unqualified OK. pass "
                  "--no-early-stop to climb the whole ladder anyway.")
            break
        if args.cooldown and i + 1 < len(rates):
            _time.sleep(args.cooldown)

    args._sweep_wall_s = _time.monotonic() - sweep_started
    return _sweep_report(rungs, out_root, args)


def _sweep_report(rungs: list[dict], out_root: Path, args) -> int:
    """One table, and one sentence naming the highest rung that held."""
    def _n(v, d=0):
        return "-" if v is None else f"{v:,.{d}f}"

    def _pct(v):
        return "-" if v is None else f"{v:.1%}"

    hdr = ("| rate asked | achieved | held | error | TTFT p50 | TTFT p95 "
           "| E2E p50 | verdict |")
    rows = [hdr, "|---|---|---|---|---|---|---|---|"]
    for r in rungs:
        rows.append(
            f"| {r['rate']:g} rps | {_n(r['achieved_rps'], 1)} | "
            f"{_n(r['held'])} | {_pct(r['err'])} | "
            f"{_n(r['ttft_p50'])} | {_n(r['ttft_p95'])} | "
            f"{_n(r['e2e_p50'])} | {r['kind'].upper()} |")

    # the ceiling is the highest rung that STAYED VALID, never the highest
    # one we managed to submit. every sweep in this category anchors on the
    # latter and reports a top rung its own error rate disqualifies.
    # A caution explicitly says evidence was undermined (sample size,
    # delivery, cache, etc.). It is not a proven capacity point.
    good = [r for r in rungs if r["kind"] == "ok"]
    if good:
        best = good[-1]
        head = (f"Highest rate that held: {best['rate']:g} requests/second, "
                f"which carried about {_n(best['held'])} concurrent.")
        def _sentence(t: str) -> str:
            t = t.strip()
            return t if t.endswith(".") else t + "."

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
                      (f"Sweep command wall time: "
                       f"{getattr(args, '_sweep_wall_s', 0.0):.1f}s. Per-rung "
                       "wall time includes setup and response drain; the "
                       "configured duration is offered-load schedule time."),
                      "",
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

    sizing = getattr(args, "sizing_concurrency", None)
    legacy = getattr(args, "legacy_concurrency", None)
    if legacy is not None:
        print("warning: --concurrency is now --sizing-concurrency. it derives "
              "one fixed open-loop rate; it does not hold concurrency.",
              file=sys.stderr)
        sizing = legacy
    cfg: dict = {
        "profile_path": args.profile,
        "endpoint": ep,
        "sizing_concurrency": sizing,
        "duration_s": args.duration,
        "out_dir": args.out_dir,
        "title": args.title or (
            f"open-loop rate sized from {sizing} concurrent, {args.endpoint}"),
        "label": args.label or (
            "Describe the capacity this ran on. Shared pay-per-token is not a "
            "performance claim for a dedicated endpoint."),
    }
    if args.max_output_tokens is not None:
        cfg["max_output_tokens_cap"] = args.max_output_tokens

    # SLA targets. the whole reason to run this is "do we meet ours", so it
    # has to be expressible here. without them the report falls back to the
    # profile's, which on a bundled profile are illustrative.
    ttft = {q: v for q, v in (("p50", args.ttft_p50), ("p90", args.ttft_p90),
                              ("p95", args.ttft_p95), ("p99", args.ttft_p99))
            if v is not None}
    ttfg = {q: v for q, v in (("p50", args.ttfg_p50), ("p90", args.ttfg_p90),
                              ("p95", args.ttfg_p95), ("p99", args.ttfg_p99))
            if v is not None}
    if ttft or ttfg or args.success_rate is not None:
        targets: dict = {"targets_are": "yours, passed on the command line"}
        if ttft:
            targets["ttft_ms"] = ttft
        if ttfg:
            targets["ttfg_ms"] = ttfg
        if args.success_rate is not None:
            targets["success_rate"] = args.success_rate
        cfg["acceptance_targets"] = targets

    # A config generator must not happily write a file that the runner will
    # reject. Validate the endpoint, profile, workload controls, and policy
    # before touching the requested output path.
    try:
        from .client import EndpointConfig
        from .config_validation import validate_acceptance_targets
        from .profile import Profile
        from .runner import RunConfig
        EndpointConfig(**ep)
        Profile.from_json(args.profile)
        validate_acceptance_targets(cfg.get("acceptance_targets"))
        RunConfig(**cfg)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid quickstart configuration: {exc}") from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")
    print()
    print("run it with:")
    print(f"  python3 -m traffic_replay run --config {out}")
    print()
    print("a fixed open-loop arrival rate and pool size are derived at run "
          "time from a short unloaded sizing pass. concurrency is measured, "
          "not held.")
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
    s.add_argument("--sizing-concurrency", type=int, default=None,
                   help="unloaded concurrency used to derive one fixed "
                        "open-loop rate (default 10); it is not held")
    s.add_argument("--concurrency", dest="legacy_concurrency", type=int,
                   default=None, help=argparse.SUPPRESS)
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
                        '\'{"reasoning_effort": "low"}\' when the target '
                        'documents that control')
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
    s.add_argument("--max-concurrency", type=int, default=None,
                   help="worker bound; sizing derives it when omitted")
    s.add_argument("--max-pending-requests", type=int, default=None,
                   help="bound on running plus queued client requests")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument("--skip-preflight", action="store_true",
                   help="skip the 2-request endpoint check. not recommended")
    s.add_argument(
        "--probe-extra-body", action="append", type=_json_object_arg,
        default=[], metavar="JSON",
        help="after a no-answer preflight, explicitly test this documented "
             "reasoning-control JSON object; repeat for multiple candidates")
    s.add_argument("--force", action="store_true",
                   help="run even when the preflight has shown the run will "
                        "produce no readable answers")
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
    s.add_argument("--max-concurrency", type=int, default=256,
                   help="fixed worker bound reused unchanged at every rung")
    s.add_argument("--max-pending-requests", type=int, default=None,
                   help="bound on running plus queued client requests")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument("--skip-preflight", action="store_true",
                   help="skip the representative endpoint gate")
    s.add_argument(
        "--probe-extra-body", action="append", type=_json_object_arg,
        default=[], metavar="JSON",
        help="after a no-answer preflight, explicitly test this documented "
             "reasoning-control JSON object; repeat for multiple candidates")
    s.add_argument("--force", action="store_true",
                   help="run despite a preflight with no readable answer")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("quickstart",
                       help="write a run config from endpoint + concurrency")
    s.add_argument("--host", required=True,
                   help="workspace URL, e.g. https://my-ws.cloud.databricks.com")
    s.add_argument("--endpoint", required=True,
                   help="endpoint name, or a full /serving-endpoints/... path")
    s.add_argument("--profile", required=True,
                   help="traffic profile JSON describing your prompt shape")
    cg = s.add_mutually_exclusive_group(required=True)
    cg.add_argument("--sizing-concurrency", type=int,
                    help="unloaded concurrency used to derive a fixed rate")
    cg.add_argument("--concurrency", dest="legacy_concurrency", type=int,
                    help=argparse.SUPPRESS)
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
    s.add_argument("--port", type=int, default=0,
                   help="mock-server port; 0 asks the OS for a free port")
    s.add_argument("--duration", type=int, default=25)
    s.add_argument("--workdir", default="results/validation")
    s.add_argument("--tolerance-ms", type=float, default=60.0)
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--format", choices=("text", "json"), default="text",
                   help="json prints the full comparison report")
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
