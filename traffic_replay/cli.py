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
import os
import sys
import threading
import time
import uuid
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


def cmd_adapters(args) -> int:
    """List the versioned wire contracts available in this installation."""
    from .adapters import list_endpoint_adapters

    adapters = list_endpoint_adapters()
    if args.format == "json":
        print(json.dumps(
            {"schema_version": "endpoint-adapter-catalog/v1",
             "adapters": adapters},
            indent=2,
            allow_nan=False,
        ))
        return 0
    for adapter in adapters:
        media = ", ".join(adapter["accepted_response_media_types"])
        print(
            f"{adapter['adapter_id']}: {adapter['response_mode']}; "
            f"response {media}; usage {adapter['usage_request_mode']}")
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
    from .quota_planner import QuotaPlanError
    from .runner import RunConfig, run
    cfg = loads_strict(Path(args.config).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("run config JSON must be an object")
    if cfg.get("concurrency") is not None and cfg.get("sizing_concurrency") is None:
        print("warning: config field 'concurrency' is legacy; it is treated as "
              "'sizing_concurrency', which derives a fixed open-loop rate and "
              "does not hold concurrency.", file=sys.stderr)
    rc = RunConfig(**cfg)
    json_mode = getattr(args, "format", "text") == "json"
    try:
        out = run(rc, quiet=json_mode)
    except QuotaPlanError as exc:
        if json_mode:
            print(json.dumps({
                "passed": False,
                "stage": "quota_plan",
                "exit_code": 3,
                "quota_plan": exc.plan,
            }, indent=2, allow_nan=False))
        return 3
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
    from .json_input import json_error_detail, loads_strict
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
    def strict_rows(path: Path):
        for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
            if not line.strip():
                raise ValueError(
                    f"{path.name}:{line_number}: blank JSONL record")
            try:
                value = loads_strict(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON "
                    f"({json_error_detail(exc)})") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path.name}:{line_number}: JSONL record must be an "
                    "object")
            yield value

    truth_by_id = {}
    for rec in strict_rows(truth):
        truth_by_id[rec["request_id"]] = rec
    rows = []
    for r in strict_rows(Path(out["out_dir"]) / "requests.jsonl"):
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
    print(f"wrote {out}/comparison.html (primary report)")
    print(f"wrote {out}/comparison.md (text alternative)")
    return 0


def _pair(text, what):
    """Parse "10000" or "10000,24000" into a p50/p95 pair.

    A single value uses an explicitly disclosed convenience heuristic.  It is
    not a measured tail, target, or provider recommendation; production work
    must pass both values or use a measured profile with provenance.
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
        # A fraction has no room for the numeric 2.4x convenience spread. Use
        # the separately disclosed bounded heuristic; this is an assumption,
        # not an empirical claim about cache-reuse distributions.
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


def _preflight(cfg: dict, *, representative_plans=None,
               runtime_quota_guard=None, row_sink=None) -> dict:
    """Send a couple of real requests and report what the endpoint does.

    This exists because the ways this tool produces a confidently wrong
    number are nearly all visible in two requests: auth that does not work,
    a model that spends its whole token budget reasoning, an endpoint that
    does not report usage, or one that does not report cached tokens. Better
    to find them in ten seconds than in a five minute run.
    """
    from .client import EndpointClient, EndpointConfig
    from .artifacts import redact_secrets
    from .runner import (
        RunConfig,
        _attach_setup_request_binding,
        _annotate_result,
        _client_payload_hash,
        _exception_result,
        _representative_plans,
        _rescope_representative_plans,
        _token,
    )

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    rc = RunConfig(**clean)
    ecfg = EndpointConfig(**rc.endpoint)
    tok = _token(ecfg)
    client_kwargs = {"refresh": lambda: _token(ecfg)}
    if runtime_quota_guard is not None:
        client_kwargs["runtime_quota_guard"] = runtime_quota_guard
    client = EndpointClient(ecfg, tok, **client_kwargs)
    plans = (_rescope_representative_plans(
        _representative_plans(rc), f"direct-preflight-{uuid.uuid4().hex}")
        if representative_plans is None else list(representative_plans))
    if not plans:
        raise ValueError("preflight needs at least one representative plan")
    out: dict = {"auth": bool(tok),
                 "budgets": [p["max_output"] for p in plans],
                 "representatives": [p["representative"] for p in plans]}
    rows = []
    request_rows = []
    for position, plan in enumerate(plans, start=1):
        body_hash = _client_payload_hash(
            client, ecfg, plan["messages"], plan["max_output"])
        try:
            res = client.send(
                plan["messages"], plan["max_output"], plan["request_id"],
                scheduled_s=0.0, dispatch_lag_ms=0.0,
                intended=plan["intended"], chars_sent=plan["chars"])
            rows.append(res)
            request_row = _annotate_result(
                res, "preflight", plan, body_hash)
        except Exception as exc:
            rows.append(exc)
            request_row = _exception_result(
                plan["request_id"], "preflight", plan, body_hash,
                "preflight request outcome unknown: "
                f"{type(exc).__name__}: {redact_secrets(str(exc))}",
                endpoint_adapter=ecfg.adapter)
            # The exception boundary cannot prove whether a POST reached the
            # provider.  Unknown is materially different from zero for quota
            # accounting.
            request_row["request_attempts"] = None
            request_row["connection_attempts"] = None
        _attach_setup_request_binding(
            request_row, endpoint=rc.endpoint,
            max_output_tokens_cap=rc.max_output_tokens_cap,
            plan=plan, phase="preflight", position=position)
        request_rows.append(request_row)
        if callable(row_sink):
            row_sink(request_row)
    out["_request_rows"] = request_rows
    transport_contract = getattr(client, "transport_contract", None)
    out["transport"] = (
        transport_contract() if callable(transport_contract) else None)
    out["include_usage_support_state"] = (
        out["transport"].get("include_usage_support_state")
        if isinstance(out["transport"], dict) else None)
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
    failed_indices = [
        i for i, r in enumerate(rows)
        if isinstance(r, Exception) or r.status != 200
        or not _answer_is_complete(r)
    ]
    # Probe the largest failing representative. Probing the first failure can
    # incorrectly label a control "ignored" simply because the smaller p50
    # output budget was exhausted. A successful discovery is still followed
    # by a full preflight rerun, which must pass every representative before
    # measured load can start.
    failed_index = (max(
        failed_indices,
        key=lambda index: int(plans[index]["max_output"]),
    ) if failed_indices else None)
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
    ep: dict = {
        "base_url": args.host.rstrip("/"),
        "path": path,
        # Keep the historic 0.0 wire behavior when older programmatic callers
        # construct a Namespace without the new flag.  ``None`` is persisted
        # explicitly so reruns preserve omission rather than falling back to
        # EndpointConfig's numeric default.
        "temperature": getattr(args, "temperature", 0.0),
    }
    endpoint_adapter = getattr(args, "endpoint_adapter", None)
    if endpoint_adapter:
        ep["adapter"] = endpoint_adapter
    if args.auth_profile:
        ep["auth_profile"] = args.auth_profile
    else:
        ep["auth_token_env"] = args.token_env
    if args.model:
        ep["model"] = args.model
    production_policy = getattr(
        args, "production_connection_policy", None)
    if production_policy is not None:
        ep["production_connection_policy"] = production_policy
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

    fixed_rate = getattr(args, "fixed_rate", None)
    exact_requests = getattr(args, "requests", None)
    if exact_requests is not None and (
            isinstance(exact_requests, bool) or not isinstance(exact_requests, int)
            or exact_requests <= 0):
        raise SystemExit("--requests must be a positive integer")
    if exact_requests is not None:
        from .schedule import MAX_EXACT_ANALYSIS_REQUEST_ROWS
        if exact_requests > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
            raise SystemExit(
                f"--requests cannot exceed the exact-analysis limit of "
                f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}")
    if fixed_rate is not None:
        import math
        if isinstance(fixed_rate, bool) or not math.isfinite(fixed_rate) \
                or fixed_rate <= 0:
            raise SystemExit("--fixed-rate must be positive and finite")
    sizing = getattr(args, "sizing_concurrency", None)
    legacy = (getattr(args, "legacy_concurrency", None)
              if hasattr(args, "legacy_concurrency")
              else getattr(args, "concurrency", None))
    if sizing is not None and legacy is not None:
        raise SystemExit("use --sizing-concurrency or legacy --concurrency, not both")
    if fixed_rate is not None and (sizing is not None or legacy is not None):
        raise SystemExit(
            "use --fixed-rate or --sizing-concurrency, not both")
    if legacy is not None:
        print("warning: --concurrency is now --sizing-concurrency. it derives "
              "one fixed open-loop rate; it does not hold concurrency.",
              file=sys.stderr)
        sizing = legacy
    if exact_requests is not None and (
            fixed_rate is not None or sizing is not None or legacy is not None):
        raise SystemExit(
            "use --requests or --fixed-rate/--sizing-concurrency, not both")
    if sizing is None and fixed_rate is None and exact_requests is None \
            and getattr(args, "cmd", "benchmark") == "benchmark":
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
    calibration_requests = getattr(args, "calibrate_requests", None)
    if calibration_requests is not None:
        cfg["calibrate_n"] = calibration_requests
    ttft_definition = getattr(args, "ttft_definition", None)
    if ttft_definition is not None:
        cfg["ttft_definition"] = ttft_definition
    if fixed_rate is not None:
        cfg.update(
            qps_base=fixed_rate, qps_burst=fixed_rate,
            qps_min=fixed_rate, qps_max=fixed_rate, rate_scale=1.0)
    if exact_requests is not None:
        # Midpoint spacing avoids an arrival exactly at the duration boundary
        # and gives the high-level benchmark an exact, deterministic count.
        timestamps = "".join(
            f"{(index + 0.5) * args.duration / exact_requests:.12f}\n"
            for index in range(exact_requests))
        from .immutable_config import write_immutable_text
        cfg["timestamps_file"] = str(write_immutable_text(
            args.out_dir, "timestamps", timestamps))
        cfg["title"] = args.title or (
            f"exact {exact_requests}-request open-loop workload, "
            f"{args.endpoint}")
    if getattr(args, "max_concurrency", None) is not None:
        cfg["max_concurrency"] = args.max_concurrency
    if getattr(args, "max_pending_requests", None) is not None:
        cfg["max_pending_requests"] = args.max_pending_requests

    inp = _pair(args.input_tokens, "input-tokens")
    outp = _pair(args.output_tokens, "output-tokens")
    if args.prompts:
        # The bounded no-follow snapshot in _freeze_and_prevalidate_cli_config
        # is the first read. An eager pathlib read here could block on a FIFO
        # or parse a different file view from the one eventually replayed.
        cfg["prompts_file"] = args.prompts
    elif args.profile:
        cfg["profile_path"] = args.profile
    else:
        prof = {
            "name": "from_command_line",
            "input_tokens": inp,
            "output_tokens": outp,
            "cache_fraction": _pair(
                getattr(args, "cache_fraction", None)
                or getattr(args, "cache_hit_rate", "0.3,0.7"),
                "cache-fraction"),
            "provenance": ("figures passed on the command line, not measured "
                           "from logs. build one from your own traffic with "
                           "scripts/profile_from_logs.py when you can."),
            "label": ("Traffic shape stated on the command line rather than "
                      "measured."),
        }
        from .immutable_config import publish_legacy_copy, write_immutable_json
        pf = write_immutable_json(args.out_dir, "profile", prof)
        if getattr(args, "cmd", "benchmark") == "benchmark":
            publish_legacy_copy(pf, Path(args.out_dir) / "profile.json")
        cfg["profile_path"] = str(pf)

    # the per-request budget is min(sampled_output, max_output_tokens_cap),
    # and the cap defaults to 512, so a workload wanting more than that was
    # silently clipped. size the cap from whatever actually decides the
    # output distribution for THIS run, which is the given profile when one
    # was passed and the flags otherwise. A supplied profile is intentionally
    # not read here; the cap is replaced from its private bounded snapshot in
    # _freeze_and_prevalidate_cli_config.
    _p95 = outp["p95"]
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
            or not (0 < args.success_rate < 1)):
        raise SystemExit("--success-rate must be in (0, 1); finite evidence "
                         "cannot statistically demonstrate 100% reliability")
    if ttft or ttfg or args.success_rate is not None:
        t: dict = {"targets_are": "yours, passed on the command line"}
        if ttft:
            t["ttft_ms"] = ttft
        if ttfg:
            t["ttfg_ms"] = ttfg
        if args.success_rate is not None:
            t["success_rate"] = args.success_rate
        cfg["acceptance_targets"] = t
    rate_limits_path = getattr(args, "rate_limits_file", None)
    if rate_limits_path:
        from .config_validation import validate_rate_limits
        from .json_input import loads_strict
        try:
            raw = Path(rate_limits_path).read_text(encoding="utf-8")
            rate_limits = loads_strict(raw)
            if not isinstance(rate_limits, dict):
                raise ValueError("top-level value must be an object")
            validate_rate_limits(rate_limits)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"invalid --rate-limits file {rate_limits_path!r}: {exc}") \
                from exc
        cfg["rate_limits"] = rate_limits
    if sizing is not None:
        # Two representative preflight requests are always constructed. Each
        # explicit candidate can add one paid reasoning-control probe if the
        # representatives are unreadable; reserve the whole possible setup
        # population before generating the replay ceiling.
        setup_rows = (0 if getattr(args, "skip_preflight", False)
                      else 2 + len(
                          getattr(args, "probe_extra_body", None) or []))
        _apply_cli_sizing_resource_ceiling(cfg, setup_rows=setup_rows)
    return cfg


def _quota_setup_plans(cfg: dict, args, *, representative_plans=None) \
        -> list[dict]:
    """Conservatively enumerate CLI traffic that can precede replay."""
    if getattr(args, "skip_preflight", False):
        return []
    if representative_plans is None:
        from .runner import RunConfig, _representative_plans
        rc = RunConfig(**cfg)
        representatives = _representative_plans(rc)
    else:
        representatives = list(representative_plans)
    if not representatives:
        raise ValueError("quota setup needs representative workload plans")
    plans = list(representatives)
    probes = _validated_probe_candidates_for_endpoint(
        cfg, getattr(args, "probe_extra_body", None) or [])
    args.probe_extra_body = probes
    if probes:
        # Which representative fails is observable only after traffic. Use
        # the largest offered envelope for each explicitly requested probe,
        # with the exact merged candidate body that the probe will submit.
        def demand(plan):
            intended = plan.get("intended") or (0,)
            return (int(intended[0] or 0), int(plan.get("max_output") or 0))

        import copy
        largest = max(representatives, key=demand)
        base_extra = cfg.get("endpoint", {}).get("extra_body") or {}
        for probe in probes:
            candidate = copy.deepcopy(largest)
            merged_extra = _deep_merge(
                copy.deepcopy(base_extra), copy.deepcopy(probe))
            if cfg.get("rate_limits") is not None \
                    and "service_tier" in merged_extra \
                    and merged_extra["service_tier"] != "default":
                raise ValueError(
                    "quota-aware reasoning probes require service_tier to "
                    "be absent or the exact string 'default'; another tier "
                    "needs its own verified rate-limit scope")
            candidate["_quota_extra_body"] = merged_extra
            plans.append(candidate)
    return plans


def _apply_cli_sizing_resource_ceiling(cfg: dict, *, setup_rows: int) -> None:
    """Give generated sizing configs a QPS ceiling that fits exact analysis.

    The runner still counts the concrete seeded Poisson schedule and refuses
    any overage before credentials or traffic. This prevents the unrelated
    500-QPS dataclass default from invalidating the normal sizing workflow.
    """
    from .runner import RunConfig, sizing_probe_row_count
    from .schedule import conservative_sizing_qps_ceiling

    rc = RunConfig(**cfg)
    if rc.sizing_concurrency is None:
        return
    ceiling = conservative_sizing_qps_ceiling(
        rc.duration_s,
        calibration_rows=rc.calibrate_n,
        sizing_rows=sizing_probe_row_count(rc.calibrate_n),
        setup_rows=setup_rows,
        context="generated CLI sizing run",
    )
    cfg.update(
        qps_base=ceiling,
        qps_burst=ceiling,
        qps_min=ceiling,
        qps_max=ceiling,
        rate_scale=1.0,
    )


def _quota_gate(cfg: dict, args, *, rates: list[float] | None = None,
                prevalidated=None, prevalidated_rungs=None) \
        -> int | None:
    """Refuse a known-unsafe paid workload before CLI preflight traffic."""
    args._quota_endpoint_metadata = None
    if cfg.get("rate_limits") is None:
        args._quota_plan = None
        args._runtime_quota_guard = None
        return None
    from .quota_planner import (
        bind_quota_plan_to_endpoint,
        plan_run_quota,
        plan_sweep_quota,
        render_quota_plan,
    )
    from .runner import RunConfig, _token
    from .client import EndpointConfig
    from .endpoint_meta import (
        fetch_endpoint_metadata,
        rate_limit_endpoint_binding,
    )

    if prevalidated is not None and prevalidated_rungs is not None:
        raise ValueError(
            "quota gate accepts one run or sweep prevalidation, not both")
    representatives = None
    if prevalidated is not None:
        representatives = prevalidated.representative_plans
    elif prevalidated_rungs:
        representatives = prevalidated_rungs[0].representative_plans
    try:
        setup = _quota_setup_plans(
            cfg, args, representative_plans=representatives)
    except ValueError as exc:
        plan = {
            "plan_kind": "sweep" if rates is not None else "run",
            "status": "refused",
            "may_start": False,
            "refusal_stage": "rate_limit_scope_validation",
            "refusal_reasons": [str(exc)],
        }
        args._quota_plan = plan
        args._runtime_quota_guard = None
        print("[quota-plan] REFUSED before endpoint traffic: " + str(exc))
        return 3
    try:
        if rates is None:
            rc = (prevalidated.rc if prevalidated is not None
                  else RunConfig(**cfg))
            plan = plan_run_quota(
                rc, setup_plans=setup, prevalidated=prevalidated)
        else:
            plan = plan_sweep_quota(
                cfg, rates, duration_s=args.duration,
                cooldown_s=args.cooldown, setup_plans=setup,
                prevalidated_rungs=prevalidated_rungs)
    except ValueError as exc:
        # Invalid/empty schedules are a user-facing refusal, not a Python
        # traceback. This path is deliberately before token lookup or any
        # endpoint request.
        plan = {
            "plan_kind": "sweep" if rates is not None else "run",
            "status": "refused",
            "may_start": False,
            "refusal_stage": "schedule_validation",
            "refusal_reasons": [str(exc)],
        }
        args._quota_plan = plan
        print("[quota-plan] REFUSED before endpoint traffic: " + str(exc))
        return 3
    # A failed schedule plan needs no credential or network access.  A passing
    # plan still cannot authorize paid POSTs until the control-plane endpoint
    # document binds this route to the configured P2T model shape.
    if plan is not None and plan.get("may_start"):
        endpoint = EndpointConfig(**cfg["endpoint"])
        metadata = fetch_endpoint_metadata(
            endpoint.base_url, endpoint.path, _token(endpoint), timeout=5.0)
        args._quota_endpoint_metadata = metadata
        binding = rate_limit_endpoint_binding(
            cfg["rate_limits"], metadata, endpoint.path)
        plan = bind_quota_plan_to_endpoint(plan, binding)
    args._quota_plan = plan
    args._runtime_quota_guard = None
    if plan is not None and plan.get("may_start"):
        from .quota_planner import (
            RuntimeQuotaGuard,
            runtime_quota_scope_material,
        )
        args._runtime_quota_guard = RuntimeQuotaGuard(
            cfg["rate_limits"],
            shard_index=int(cfg.get("shard_index", 0)),
            shard_total=int(cfg.get("shard_total", 1)),
            scope_material=runtime_quota_scope_material(
                cfg["rate_limits"], cfg["endpoint"]))
    if plan is not None:
        print(render_quota_plan(plan))
    return None if plan is None or plan.get("may_start") else 3


def _json_object_arg(value: str) -> dict:
    """Parse one bounded, persistence-safe probe object before traffic."""
    try:
        from .json_input import loads_strict
        parsed = loads_strict(value)
        parsed = _validated_probe_candidate(parsed, position=1)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        # The candidate may itself be a credential. Never echo the raw CLI
        # token in an argparse diagnostic, even though no artifact exists yet.
        try:
            input_bytes = len(str(value).encode("utf-8", errors="replace"))
        except Exception:
            input_bytes = -1
        raise argparse.ArgumentTypeError(
            "expected a bounded, credential-free JSON object "
            f"(input bytes={input_bytes}): {_safe_probe_detail(exc)}") \
            from exc
    return parsed


_PROBE_EVIDENCE_SCHEMA = "reasoning-control-probe-evidence/v1"
_PROBE_MAX_CANDIDATES = 16
_PROBE_MAX_CANONICAL_BYTES = 16 * 1024
_PROBE_MAX_DEPTH = 8
_PROBE_MAX_NODES = 256
_PROBE_MAX_OBJECT_KEYS = 64
_PROBE_MAX_ARRAY_ITEMS = 64
_PROBE_MAX_KEY_BYTES = 128
_PROBE_MAX_STRING_BYTES = 4096
_PROBE_REQUEST_ID_MAX_BYTES = 256
_PROBE_MAX_PHYSICAL_BODY_HASHES = 5
_PROBE_DISPOSITIONS = {"accepted", "rejected", "unknown"}
_PROBE_EVIDENCE_METHODS = {
    "single_request_behavior_observation",
    "request_validation_response",
    "candidate_rejection_unverified_http_failure",
    "non_validation_http_failure",
    "transport_outcome_unknown",
}


def _probe_candidate_canonical_json(value: dict) -> str:
    """Canonical UTF-8 JSON used by the v1 candidate digest."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"))


def _probe_candidate_sha256(value: dict) -> str:
    import hashlib
    return hashlib.sha256(
        _probe_candidate_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_probe_json_shape(value: object) -> None:
    """Bound a control object independently of provider payload limits."""
    import math

    nodes = 0

    def walk(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _PROBE_MAX_NODES:
            raise ValueError(
                f"probe candidate exceeds {_PROBE_MAX_NODES} JSON nodes")
        if depth > _PROBE_MAX_DEPTH:
            raise ValueError(
                f"probe candidate exceeds depth {_PROBE_MAX_DEPTH}")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            if not -(2 ** 63) <= item <= 2 ** 63 - 1:
                raise ValueError("probe candidate integer exceeds int64")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("probe candidate numbers must be finite")
            return
        if isinstance(item, str):
            if len(item.encode("utf-8")) > _PROBE_MAX_STRING_BYTES:
                raise ValueError(
                    "probe candidate string exceeds "
                    f"{_PROBE_MAX_STRING_BYTES} UTF-8 bytes")
            if any(ord(char) < 0x20 or ord(char) == 0x7f for char in item):
                raise ValueError(
                    "probe candidate strings must not contain controls")
            return
        if isinstance(item, list):
            if len(item) > _PROBE_MAX_ARRAY_ITEMS:
                raise ValueError(
                    "probe candidate array exceeds "
                    f"{_PROBE_MAX_ARRAY_ITEMS} items")
            for child in item:
                walk(child, depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > _PROBE_MAX_OBJECT_KEYS:
                raise ValueError(
                    "probe candidate object exceeds "
                    f"{_PROBE_MAX_OBJECT_KEYS} keys")
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        "probe candidate keys must be non-empty strings")
                if len(key.encode("utf-8")) > _PROBE_MAX_KEY_BYTES:
                    raise ValueError(
                        "probe candidate key exceeds "
                        f"{_PROBE_MAX_KEY_BYTES} UTF-8 bytes")
                if any(ord(char) < 0x20 or ord(char) == 0x7f
                       for char in key):
                    raise ValueError(
                        "probe candidate keys must not contain controls")
                walk(child, depth + 1)
            return
        raise ValueError("probe candidate contains a non-JSON value")

    walk(value, 0)


def _validated_probe_candidate(value: object, *, position: int) -> dict:
    """Return an exact safe copy; never redact-and-send a changed control."""
    import copy
    from .artifacts import redact_secrets
    from .client import validate_extra_body_safety

    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"probe candidate {position} must be a non-empty JSON object")
    # Credential rejection precedes diagnostics that might otherwise reveal a
    # secret-looking key or value. A redacted mutation is never sent.
    validate_extra_body_safety(value)
    safe = redact_secrets(value)
    if safe != value:
        raise ValueError(
            f"probe candidate {position} contains secret-like material")
    _validate_probe_json_shape(value)
    raw = _probe_candidate_canonical_json(value).encode("utf-8")
    if len(raw) > _PROBE_MAX_CANONICAL_BYTES:
        raise ValueError(
            f"probe candidate {position} exceeds "
            f"{_PROBE_MAX_CANONICAL_BYTES} canonical UTF-8 bytes")
    return copy.deepcopy(value)


def _validated_probe_candidates(value: object) -> list[dict]:
    import copy

    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("reasoning-control probe candidates must be a list")
    if len(value) > _PROBE_MAX_CANDIDATES:
        raise ValueError(
            f"at most {_PROBE_MAX_CANDIDATES} probe candidates are allowed")
    out = [
        _validated_probe_candidate(candidate, position=position)
        for position, candidate in enumerate(value, start=1)
    ]
    seen: set[str] = set()
    for candidate in out:
        digest = _probe_candidate_sha256(candidate)
        if digest in seen:
            raise ValueError("duplicate reasoning-control probe candidate")
        seen.add(digest)
    return copy.deepcopy(out)


def _validated_probe_candidates_for_endpoint(
        cfg: dict, candidates: object) -> list[dict]:
    """Validate every merged wire body before any candidate is submitted."""
    import copy
    from .client import EndpointConfig

    out = _validated_probe_candidates(candidates)
    endpoint = cfg.get("endpoint") if isinstance(cfg, dict) else None
    if out and not isinstance(endpoint, dict):
        raise ValueError("probe candidates require an endpoint configuration")
    for candidate in out:
        merged = copy.deepcopy(endpoint)
        merged["extra_body"] = _deep_merge(
            merged.get("extra_body") or {}, copy.deepcopy(candidate))
        EndpointConfig(**merged)
    return out


def _finite_temperature_arg(value: str) -> float:
    """Parse an explicit numeric temperature without accepting JSON NaN/Inf."""
    import math

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            f"temperature must be a finite number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            f"temperature must be a finite number, got {value!r}")
    return parsed


def _calibration_request_count_arg(value: str) -> int:
    """Parse the bounded number of unloaded calibration requests."""
    from .runner import _MAX_CALIBRATION_REQUESTS

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "calibration requests must be a non-negative integer") from exc
    if parsed < 0 or parsed > _MAX_CALIBRATION_REQUESTS:
        raise argparse.ArgumentTypeError(
            "calibration requests must be between 0 and "
            f"{_MAX_CALIBRATION_REQUESTS}")
    return parsed


def _add_temperature_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same sampling-field contract to every high-level command."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--temperature", type=_finite_temperature_arg, default=0.0,
        metavar="FINITE_FLOAT",
        help="send this numeric sampling temperature (default: 0.0)")
    group.add_argument(
        "--omit-temperature", dest="temperature", action="store_const",
        const=None, default=argparse.SUPPRESS,
        help="omit temperature from every request; this is distinct from "
             "sending numeric 0.0")


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


def _validate_probe_evidence_envelope(value: object, *, row: dict | None = None) \
        -> dict:
    """Validate the exact metadata-only v1 envelope and optional row links."""
    import copy
    import re

    required = {
        "schema_version", "candidate_index", "candidate_redacted",
        "candidate_canonical_sha256", "disposition", "evidence_method",
        "effective_status", "effective_value", "request_id",
        "logical_request_body_sha256", "physical_request_body_sha256s",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "reasoning-control probe evidence has unknown or missing fields")
    if value.get("schema_version") != _PROBE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported reasoning-control probe evidence schema")
    index = value.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) \
            or not 1 <= index <= _PROBE_MAX_CANDIDATES:
        raise ValueError("probe evidence candidate_index is invalid")
    candidate = _validated_probe_candidate(
        value.get("candidate_redacted"), position=index)
    expected_digest = _probe_candidate_sha256(candidate)
    digest = value.get("candidate_canonical_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) \
            or digest != expected_digest:
        raise ValueError("probe evidence candidate SHA-256 is invalid")
    disposition = value.get("disposition")
    method = value.get("evidence_method")
    if disposition not in _PROBE_DISPOSITIONS \
            or method not in _PROBE_EVIDENCE_METHODS:
        raise ValueError("probe evidence disposition or method is invalid")
    allowed_methods = {
        "accepted": {"single_request_behavior_observation"},
        "rejected": {"request_validation_response"},
        "unknown": {
            "candidate_rejection_unverified_http_failure",
            "non_validation_http_failure", "transport_outcome_unknown"},
    }
    if method not in allowed_methods[disposition]:
        raise ValueError(
            "probe evidence method disagrees with its disposition")
    expected_effective = (
        "not_applied_request_rejected"
        if disposition == "rejected" else "unknown")
    if value.get("effective_status") != expected_effective \
            or value.get("effective_value") is not None:
        raise ValueError(
            "probe evidence must not infer an effective control value")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id \
            or len(request_id.encode("utf-8")) > _PROBE_REQUEST_ID_MAX_BYTES \
            or any(ord(char) < 0x21 or ord(char) > 0x7e
                   for char in request_id):
        raise ValueError("probe evidence request_id is invalid")
    logical = value.get("logical_request_body_sha256")
    if not isinstance(logical, str) \
            or not re.fullmatch(r"[0-9a-f]{64}", logical):
        raise ValueError("probe evidence logical body SHA-256 is invalid")
    physical = value.get("physical_request_body_sha256s")
    if not isinstance(physical, list) \
            or len(physical) > _PROBE_MAX_PHYSICAL_BODY_HASHES \
            or any(not isinstance(item, str)
                   or not re.fullmatch(r"[0-9a-f]{64}", item)
                   for item in physical):
        raise ValueError("probe evidence physical body SHA-256 list is invalid")
    if row is not None:
        if not isinstance(row, dict) or row.get("phase") != "probe" \
                or row.get("request_id") != request_id \
                or row.get("request_body_sha256") != logical \
                or row.get("physical_request_body_sha256s", []) != physical:
            raise ValueError(
                "probe evidence request/body links disagree with its row")
        attempts = row.get("request_attempts")
        if attempts is not None and (
                isinstance(attempts, bool) or not isinstance(attempts, int)
                or attempts < 0 or len(physical) != attempts):
            raise ValueError(
                "probe evidence physical hashes disagree with attempts")
    return copy.deepcopy(value)


def _probe_evidence_envelope(*, candidate: dict, position: int,
                             disposition: str, evidence_method: str,
                             row: dict) -> dict:
    """Create a content-free envelope; HTTP acceptance never proves effect."""
    safe_candidate = _validated_probe_candidate(candidate, position=position)
    effective_status = (
        "not_applied_request_rejected"
        if disposition == "rejected" else "unknown")
    envelope = {
        "schema_version": _PROBE_EVIDENCE_SCHEMA,
        "candidate_index": position,
        "candidate_redacted": safe_candidate,
        "candidate_canonical_sha256": _probe_candidate_sha256(safe_candidate),
        "disposition": disposition,
        "evidence_method": evidence_method,
        "effective_status": effective_status,
        "effective_value": None,
        "request_id": row.get("request_id"),
        "logical_request_body_sha256": row.get("request_body_sha256"),
        "physical_request_body_sha256s": list(
            row.get("physical_request_body_sha256s") or []),
    }
    return _validate_probe_evidence_envelope(envelope, row=row)


def _answer_is_complete(result) -> bool:
    """A completed non-refusal may be visible text or valid tool use."""
    return bool(not getattr(result, "refusal_seen", False)
                and result.stream_complete and not result.parse_errors
                and (result.visible_content_seen
                     or (getattr(result, "valid_tool_calls", 0) or 0) > 0))


def _probe_reasoning_levers(cfg: dict, budget: int,
                            candidates: list[dict],
                            probe_index: int = 1, *,
                            representative_plans=None,
                            runtime_quota_guard=None,
                            row_sink=None) -> list[dict]:
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
    from .runner import (
        RunConfig,
        _attach_setup_request_binding,
        _annotate_result,
        _client_payload_hash,
        _exception_result,
        _representative_plans,
        _stable_request_id,
        _token,
    )

    # Validate the whole candidate population, including each merged endpoint
    # body, before constructing the first client. One unsafe later candidate
    # must not allow earlier candidates to emit paid traffic.
    candidates = _validated_probe_candidates_for_endpoint(cfg, candidates)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    rc = RunConfig(**clean)
    plans = (_representative_plans(rc) if representative_plans is None
             else list(representative_plans))
    if not plans:
        raise ValueError("reasoning probe needs a representative workload")
    plan = plans[min(max(probe_index, 0), len(plans) - 1)]
    # The caller passes the exact failed budget. Keep it explicit so a future
    # refactor cannot reintroduce a probe-only 512-token floor.
    budget = int(budget)
    if budget <= 0:
        raise ValueError("reasoning probe budget must be positive")
    out = []
    execution_scope = plan.get("execution_scope_id") \
        or f"direct-probe-{uuid.uuid4().hex}"
    bound_plan = copy.deepcopy(plan)
    bound_plan["max_output"] = budget

    def record(*, row: dict, candidate: dict, position: int,
               name: str, verdict: str, disposition: str,
               evidence_method: str, confidence: str, detail: str) -> None:
        envelope = _probe_evidence_envelope(
            candidate=candidate, position=position,
            disposition=disposition, evidence_method=evidence_method,
            row=row)
        row["reasoning_control_probe"] = envelope
        # The durable sink sees the complete envelope, never an intermediate
        # row whose HTTP status has not yet been classified.
        if callable(row_sink):
            row_sink(row)
        out.append({
            "name": name,
            "extra": candidate,
            "verdict": verdict,
            "disposition": disposition,
            "effective_status": envelope["effective_status"],
            "effective_value": envelope["effective_value"],
            "evidence_method": evidence_method,
            "confidence": confidence,
            "detail": detail,
            "evidence": envelope,
            "_request_row": row,
        })

    for position, extra in enumerate(candidates, start=1):
        name = _probe_label(extra, position)
        ec = copy.deepcopy(cfg["endpoint"])
        ec["extra_body"] = _deep_merge(ec.get("extra_body") or {}, extra)
        ecfg = EndpointConfig(**ec)
        client_kwargs = {"refresh": lambda: _token(ecfg)}
        if runtime_quota_guard is not None:
            client_kwargs["runtime_quota_guard"] = runtime_quota_guard
        client = EndpointClient(ecfg, _token(ecfg), **client_kwargs)
        request_id = _stable_request_id(
            execution_scope, position, "probe")
        body_hash = _client_payload_hash(
            client, ecfg, plan["messages"], budget)
        try:
            r = client.send(
                plan["messages"], budget, request_id, scheduled_s=0.0,
                dispatch_lag_ms=0.0, intended=plan["intended"],
                chars_sent=plan["chars"], probe_candidate=extra)
        except Exception as e:  # never let a probe break the run
            row = _exception_result(
                request_id, "probe", plan, body_hash,
                "reasoning-control probe outcome unknown: "
                f"{type(e).__name__}: {_safe_probe_detail(e)}",
                endpoint_adapter=ecfg.adapter)
            row["request_attempts"] = None
            row["connection_attempts"] = None
            _attach_setup_request_binding(
                row, endpoint=ec,
                max_output_tokens_cap=rc.max_output_tokens_cap,
                plan=bound_plan, phase="probe", position=position)
            record(
                row=row, candidate=extra, position=position, name=name,
                verdict="error", disposition="unknown",
                evidence_method="transport_outcome_unknown",
                confidence="none", detail=_safe_probe_detail(e)[:160])
            continue
        row = _annotate_result(r, "probe", plan, body_hash)
        _attach_setup_request_binding(
            row, endpoint=ec,
            max_output_tokens_cap=rc.max_output_tokens_cap,
            plan=bound_plan, phase="probe", position=position)
        if r.status != 200:
            # Status alone cannot identify which request field failed.  The
            # adapter saw the bounded response body in memory and may mark a
            # rejection only when provider wording names this exact candidate
            # field/path.  No response text is retained in the result row.
            rejected = getattr(r, "probe_candidate_rejected", None) is True
            validation_status = r.status in {400, 422}
            record(
                row=row, candidate=extra, position=position, name=name,
                verdict="rejected" if rejected else "error",
                disposition="rejected" if rejected else "unknown",
                evidence_method=(
                    "request_validation_response" if rejected
                    else "candidate_rejection_unverified_http_failure"
                    if validation_status else "non_validation_http_failure"),
                confidence="high" if rejected else "none",
                detail=_safe_probe_detail(r.error or "")[:220])
        elif _answer_is_complete(r):
            reasoning = ("reasoning observed" if
                         (r.reasoning_seen or r.reasoning_chunks)
                         else "no reasoning observed")
            record(
                row=row, candidate=extra, position=position, name=name,
                verdict="works", disposition="accepted",
                evidence_method="single_request_behavior_observation",
                confidence="unknown_effect",
                detail=f"answered, finish {r.finish_reason}, "
                       f"{r.completion_tokens} tokens, {reasoning}")
        else:
            record(
                row=row, candidate=extra, position=position, name=name,
                verdict="unknown", disposition="accepted",
                evidence_method="single_request_behavior_observation",
                confidence="unknown_effect",
                detail=f"accepted, still no visible answer within "
                       f"{budget} tokens; effect not demonstrated")
    return out


def _print_lever_report(levers: list[dict], budget: int) -> None:
    works = [x for x in levers if x["verdict"] == "works"]
    print("[preflight] trying the supplied reasoning-control candidates, "
          "one request each:")
    for x in levers:
        mark = {"works": "ANSWERED", "rejected": "rejected",
                "unknown": "unknown", "error": "error"}[x["verdict"]]
        print(f"[preflight]   {x['name']:24s} {mark:9s} {x['detail'][:96]}")
    if works:
        best = works[0]
        from .artifacts import redact_secrets
        flag = json.dumps(redact_secrets(best["extra"]))
        print("[preflight] a completed answer proves this candidate is worth "
              "a full preflight; it does not prove the provider applied the "
              "candidate or disabled reasoning.")
        print(f"[preflight] test this: --extra-body '{flag}'")
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
        print("  re-run with the candidate that produced an answer:")
        print()
        print(f"    --extra-body '{flag}'")
        print()
        print("  one completed probe is not proof that the provider applied "
              "the candidate or disabled reasoning; require the complete "
              "two-representative preflight and inspect reasoning evidence.")
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


def _claim_setup_traffic_evidence(cfg: dict, args, *, command: str):
    """Claim a crash-visible journal before CLI inference setup traffic.

    Preflight and reasoning probes happen before the measured runner owns a
    run directory.  They are still paid physical POSTs.  A separate sealed
    setup artifact prevents a normal preflight refusal (or a process crash)
    from making that traffic disappear from the evidence chain.
    """
    import copy
    import uuid
    from datetime import datetime, timezone

    from .artifacts import (
        RunArtifacts, snapshot_source_state,
        strict_json_dumps,
    )
    from .runner import (
        RunConfig, _effective_config, _prepare_prior_request_rows,
        _resolved_workload_id,
    )

    clean = {key: value for key, value in cfg.items()
             if not key.startswith("_")}
    rc = RunConfig(**copy.deepcopy(clean))
    inputs = copy.deepcopy(rc.input_expectations or {})
    now = time.time()
    logical_run_id = f"setup-{uuid.uuid4().hex}"
    execution_id = f"execution-{uuid.uuid4().hex}"
    artifact_id = f"artifact-{uuid.uuid4().hex}"
    guard = getattr(args, "_runtime_quota_guard", None)
    baseline = guard.snapshot() if guard is not None else None
    source = snapshot_source_state(Path(__file__).parent)
    start = {
        "start_schema_version": 1,
        "status": "setup-traffic-writing",
        "artifact_kind": "command_setup_traffic",
        "command": command,
        "run_started_at_unix": now,
        "run_started_at_utc": datetime.fromtimestamp(
            now, timezone.utc).isoformat(),
        "logical_run_id": logical_run_id,
        "workload_id": _resolved_workload_id(rc, inputs),
        "execution_id": execution_id,
        "artifact_id": artifact_id,
        "effective_config": _effective_config(rc, rc),
        "inputs": inputs,
        "source": source,
        "runtime_quota_guard": copy.deepcopy(baseline),
        "runtime_quota_guard_baseline": copy.deepcopy(baseline),
        "note": (
            "preflight/probe request evidence only; this artifact is not a "
            "latency, SLA, throughput, or capacity result"),
    }
    requested_root = Path(rc.out_dir)
    setup_root = requested_root.parent / (
        requested_root.name + "-setup-traffic")
    requested = setup_root / time.strftime(
        "%Y%m%d-%H%M%S", time.localtime(now))
    artifact = RunArtifacts.claim(
        requested, start, sync_every_rows=1, artifact_id=artifact_id)
    state = {
        "artifact": artifact,
        "baseline": baseline,
        "digests": {},
        "config": clean,
        "command": command,
        "sealed": False,
    }

    def sink(row: dict) -> None:
        if state["sealed"]:
            raise RuntimeError("setup traffic evidence is already sealed")
        if not isinstance(row, dict):
            raise ValueError("setup request evidence row must be an object")
        phase = row.get("phase")
        request_id = row.get("request_id")
        if phase not in {"preflight", "probe"} \
                or not isinstance(request_id, str) or not request_id:
            raise ValueError(
                "setup request evidence needs preflight/probe phase and a "
                "request_id")
        envelope = row.get("reasoning_control_probe")
        if phase == "probe":
            validated = _validate_probe_evidence_envelope(
                envelope, row=row)
            if envelope != validated:
                raise ValueError("probe row evidence is not canonical")
        elif envelope is not None:
            raise ValueError(
                "preflight row cannot carry reasoning-control probe evidence")
        safe = _prepare_prior_request_rows([row])[0]
        encoded = strict_json_dumps(safe)
        import hashlib
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        key = (phase, request_id)
        previous = state["digests"].get(key)
        if previous is not None:
            if previous != digest:
                raise ValueError(
                    "conflicting setup request evidence for "
                    f"{phase}/{request_id}")
            return
        artifact.append(safe)
        artifact.sync()
        state["digests"][key] = digest

    state["sink"] = sink
    args._setup_traffic_state = state
    args._setup_request_sink = sink
    args._setup_traffic_artifact_path = str(artifact.path)
    args._setup_traffic_execution_id = execution_id
    return state


def _seal_setup_traffic_evidence(cfg: dict, args, *, outcome: str,
                                 exit_code: int | None) -> Path:
    """Seal the already-claimed setup journal as an auditable non-result."""
    import copy
    import hashlib

    from .client import EndpointConfig
    from .metrics import summarize, write_outputs

    state = getattr(args, "_setup_traffic_state", None)
    if not isinstance(state, dict) or state.get("sealed"):
        raise RuntimeError("no open setup traffic evidence artifact")
    artifact = state["artifact"]
    sink = state["sink"]
    # Test doubles and third-party wrappers may return rows without invoking
    # the streaming sink. Reconcile them before sealing; production rows are
    # deduplicated by phase/request_id plus canonical digest.
    for row in list(getattr(args, "_preflight_request_rows", []) or []):
        sink(row)
    guard = getattr(args, "_runtime_quota_guard", None)
    final_guard = guard.snapshot() if guard is not None else None
    preflight_gate = copy.deepcopy(getattr(args, "_preflight_evidence", None))
    if not isinstance(preflight_gate, dict) \
            or preflight_gate.get("outcome") != outcome:
        raise ValueError("setup outcome disagrees with preflight gate evidence")
    artifact.update_start(
        status="setup-traffic-sealing",
        setup_outcome=outcome,
        setup_exit_code=exit_code,
        preflight_gate=copy.deepcopy(preflight_gate),
        durable_request_rows=artifact.row_count,
        runtime_quota_guard=copy.deepcopy(final_guard))
    rows = list(artifact.read_rows())
    endpoint = EndpointConfig(**cfg["endpoint"])
    empty_vector_sha256 = hashlib.sha256(b"").hexdigest()
    schedule_identity = {
        "encoding": "float64-le-seconds-from-run-start",
        "global_timestamps_sha256": empty_vector_sha256,
        "global_count": 0,
        "global_min_s": None,
        "global_max_s": None,
        "shard_timestamps_sha256": empty_vector_sha256,
        "shard_count": 0,
        "shard_min_s": None,
        "shard_max_s": None,
    }
    index_identity = {
        "encoding": "int64-le",
        "global_indices_sha256": empty_vector_sha256,
        "count": 0,
        "min": None,
        "max": None,
        "global_count": 0,
        "shard_index": int(cfg.get("shard_index", 0)),
        "shard_total": int(cfg.get("shard_total", 1)),
        "partition": (
            "unsharded" if int(cfg.get("shard_total", 1)) == 1
            else "round_robin_modulo"),
    }
    title = str(cfg.get("title") or "benchmark") + \
        " - setup traffic evidence"
    run_meta = {
        "title": title,
        "label": (
            "CLI preflight/probe evidence only. Do not use this artifact as "
            "a performance, SLA, throughput, or endpoint-capacity result."),
        "artifact_kind": "command_setup_traffic",
        "setup_outcome": outcome,
        "preflight_gate": copy.deepcopy(preflight_gate),
        "endpoint_path": endpoint.path,
        "endpoint_base_url": endpoint.base_url,
        "endpoint_model": endpoint.model,
        "endpoint_metadata": getattr(
            args, "_quota_endpoint_metadata", None),
        "endpoint_metadata_stability": "not_applicable_setup_only",
        "transport": getattr(args, "_preflight_transport", None),
        "input_mode": (
            "prompts" if cfg.get("prompts_file") else "profile"),
        "profile_path": (
            Path(cfg["profile_path"]).name
            if cfg.get("profile_path") else None),
        "prompts_file": (
            Path(cfg["prompts_file"]).name
            if cfg.get("prompts_file") else None),
        "ttft_definition": cfg.get("ttft_definition", "first_content"),
        "schedule_identity": schedule_identity,
        "index_identity": index_identity,
        "shard": (f"{int(cfg.get('shard_index', 0)) + 1}/"
                  f"{int(cfg.get('shard_total', 1))}"),
        "runtime_quota_guard_baseline": copy.deepcopy(state["baseline"]),
        "runtime_quota_guard": copy.deepcopy(final_guard),
    }
    summary = summarize(
        [], schedule_meta={
            "requests": 0,
            "seconds": 0,
            "source": "command setup traffic only; no replay schedule",
            "rate_min": 0.0,
            "rate_p50": 0.0,
            "rate_p95": 0.0,
            "rate_max": 0.0,
        }, run_meta=run_meta,
        ttft_definition=cfg.get("ttft_definition", "first_content"),
        rate_limits=cfg.get("rate_limits"),
        rate_limit_results=rows)
    summary["setup_traffic"] = {
        "artifact_kind": "command_setup_traffic",
        "outcome": outcome,
        "exit_code": exit_code,
        "request_rows": len(rows),
        "preflight_gate": copy.deepcopy(preflight_gate),
        "performance_result": False,
        "sla_result": False,
        "capacity_result": False,
        "note": (
            "these rows are attached once to the measured run's complete "
            "request population when the command proceeds past the setup "
            "gate, including an explicitly forced diagnostic run"),
    }
    out = write_outputs(
        None, summary, artifact.path, title, artifact_run=artifact,
        start_provenance=artifact.start_provenance)
    state["sealed"] = True
    args._setup_request_sink = None
    args._setup_traffic_artifact_path = str(out)
    return out


def _setup_artifact_reference(path: str | Path, preflight_gate: dict) -> dict:
    """Return the portable ID/digest link retained by measured artifacts."""
    import hashlib
    from .json_input import loads_strict
    from .runner import (
        _SETUP_ARTIFACT_REFERENCE_SCHEMA,
        _validated_setup_artifact_reference,
    )

    manifest_path = Path(path) / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = loads_strict(raw)
    if not isinstance(manifest, dict):
        raise ValueError("setup manifest must contain an object")
    reference = {
        "schema_version": _SETUP_ARTIFACT_REFERENCE_SCHEMA,
        "artifact_id": manifest.get("artifact_id"),
        "execution_id": manifest.get("execution_id"),
        "workload_id": manifest.get("workload_id"),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "preflight_binding_sha256": preflight_gate.get("binding_sha256"),
    }
    return _validated_setup_artifact_reference(reference, preflight_gate)


def _abort_setup_traffic_evidence(args, error: BaseException) -> None:
    state = getattr(args, "_setup_traffic_state", None)
    if not isinstance(state, dict) or state.get("sealed"):
        return
    state["artifact"].abort(error)
    args._setup_request_sink = None


def _check_preflight(cfg: dict, args, *, representative_plans=None) \
        -> int | None:
    """Run the shared benchmark/sweep gate; return an exit code on refusal."""
    from .runner import _rescope_representative_plans

    candidates = _validated_probe_candidates_for_endpoint(
        cfg, getattr(args, "probe_extra_body", None) or [])
    args.probe_extra_body = candidates
    scope = getattr(args, "_setup_traffic_execution_id", None) \
        or f"direct-preflight-{uuid.uuid4().hex}"
    if representative_plans is not None:
        representative_plans = _rescope_representative_plans(
            list(representative_plans), scope)
        args._preflight_representative_plans = representative_plans
    print("[preflight] sending 2 representative workload requests")
    guard = getattr(args, "_runtime_quota_guard", None)
    preflight_kwargs = {}
    if representative_plans is not None:
        preflight_kwargs["representative_plans"] = representative_plans
    if guard is not None:
        preflight_kwargs["runtime_quota_guard"] = guard
    setup_sink = getattr(args, "_setup_request_sink", None)
    if callable(setup_sink):
        preflight_kwargs["row_sink"] = setup_sink
    pf_res = _preflight(cfg, **preflight_kwargs)
    args._preflight_transport = pf_res.get("transport")
    args._preflight_request_rows = list(pf_res.get("_request_rows") or [])
    args._preflight_evidence = {
        "skipped": False,
        "attempted": int(pf_res.get("attempted", 0) or 0),
        "reachable": int(pf_res.get("reachable", 0) or 0),
        "readable": int(pf_res.get("readable", 0) or 0),
        "reasoning_probe_requests": 0,
    }
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
    if pf_res.get("include_usage_support_state") is False \
            and cfg.get("endpoint", {}).get("include_usage", True):
        # The preflight already paid for and recorded the provider's explicit
        # rejection plus fallback POST. Freeze the learned capability into the
        # measured config so concurrent replay workers do not rediscover it
        # during the benchmark and distort both load and retries.
        cfg["endpoint"]["include_usage"] = False
        print("[preflight] endpoint rejected stream_options.include_usage; "
              "the measured config is frozen with include_usage=false")
    if not pf_res.get("cache_reported"):
        print("[preflight] note: at least one response had no cached-token "
              "field, so achieved cache coverage may be incomplete")
    if pf_res.get("reasoning"):
        print("[preflight] this endpoint emitted reasoning-channel content; "
              "whether reasoning consumes the offered output budget is "
              "provider-, route-, and revision-specific. Inspect reported "
              "usage, finish_reason, and visible-answer completion instead "
              "of assuming a universal accounting rule.")
        if "ttft_definition" not in cfg:
            cfg["ttft_definition"] = "first_visible"
            print("[preflight] scoring TTFT on the first VISIBLE content "
                  "delta.")

    if pf_res.get("readable") != pf_res.get("attempted"):
        print(f"[preflight] only {pf_res.get('readable', 0)}/"
              f"{pf_res['attempted']} produced a valid completed answer. "
              "This gate accepts non-refusal visible content or a "
              "structurally valid non-refusal tool call, plus clean stream "
              "completion.")
        levers: list[dict] = []
        if candidates:
            print()
            probe_kwargs = {
                "budget": pf_res["budget"],
                "candidates": candidates,
                "probe_index": pf_res.get("failed_probe_index", 1),
            }
            if representative_plans is not None:
                probe_kwargs["representative_plans"] = representative_plans
            if guard is not None:
                probe_kwargs["runtime_quota_guard"] = guard
            if callable(setup_sink):
                probe_kwargs["row_sink"] = setup_sink
            levers = _probe_reasoning_levers(cfg, **probe_kwargs)
            probe_rows = []
            probe_evidence = []
            for lever in levers:
                row = lever.pop("_request_row", None)
                evidence = lever.get("evidence")
                if not isinstance(row, dict) or not isinstance(evidence, dict):
                    raise ValueError(
                        "reasoning probe did not return its request evidence")
                validated = _validate_probe_evidence_envelope(
                    evidence, row=row)
                if row.get("reasoning_control_probe") != validated:
                    raise ValueError(
                        "reasoning probe row disagrees with its evidence")
                probe_rows.append(row)
                probe_evidence.append(validated)
            args._preflight_request_rows.extend(probe_rows)
            args._preflight_evidence["reasoning_probe_requests"] = len(levers)
            args._preflight_evidence[
                "reasoning_control_probes"] = probe_evidence
            _print_lever_report(levers, pf_res["budget"])
            print()
        else:
            print("[preflight] no provider controls were guessed. pass a "
                  "model-documented control with --extra-body, or opt in to "
                  "specific candidates with --probe-extra-body.")
        if not getattr(args, "force", False):
            return _refuse(levers, args)
    return None


def _finalize_preflight_evidence(args, refused: int | None, *,
                                 cfg: dict | None = None,
                                 representative_plans=None) -> dict:
    """Freeze the truthful command-level preflight gate state.

    ``--force`` authorizes a diagnostic run after an unreadable HTTP-200
    preflight. It does not turn that gate into a pass, and it does not bypass
    a reachability failure. Keep the state separate from the command exit
    code so setup artifacts and downstream reports cannot infer "passed"
    merely because the command continued.
    """
    raw = getattr(args, "_preflight_evidence", None)
    if not isinstance(raw, dict):
        raw = {}
    counts = {}
    for field in ("attempted", "reachable", "readable",
                  "reasoning_probe_requests"):
        value = raw.get(field, 0)
        counts[field] = (
            int(value) if isinstance(value, int) and not isinstance(value, bool)
            and value >= 0 else 0)
    skipped = bool(raw.get("skipped", False))
    probe_evidence = raw.get("reasoning_control_probes")
    if probe_evidence is not None:
        if not isinstance(probe_evidence, list) \
                or len(probe_evidence) != counts["reasoning_probe_requests"]:
            raise ValueError(
                "reasoning-control probe evidence count disagrees with gate")
        probe_evidence = [
            _validate_probe_evidence_envelope(item)
            for item in probe_evidence
        ]
        indices = [item["candidate_index"] for item in probe_evidence]
        if indices != list(range(1, len(probe_evidence) + 1)):
            raise ValueError(
                "reasoning-control probe evidence order is invalid")
    elif counts["reasoning_probe_requests"]:
        raise ValueError("reasoning-control probe evidence is missing")
    force_requested = bool(getattr(args, "force", False))
    complete = bool(
        counts["attempted"] > 0
        and counts["reachable"] == counts["attempted"]
        and counts["readable"] == counts["attempted"])
    if skipped:
        outcome = "skipped"
    elif refused is not None:
        outcome = "preflight_refused"
    elif complete:
        outcome = "preflight_passed"
    elif force_requested and counts["reachable"] == counts["attempted"]:
        outcome = "preflight_forced_unreadable"
    elif force_requested:
        # Defensive state for an injected/custom gate. The production gate
        # refuses reachability failures even when --force is present.
        outcome = "preflight_forced_failed"
    else:
        outcome = "preflight_state_unknown"
    evidence = {
        "skipped": skipped,
        **counts,
        "outcome": outcome,
        "force_requested": force_requested,
        "gate_satisfied": outcome == "preflight_passed",
    }
    if probe_evidence is not None:
        evidence["reasoning_control_probes"] = probe_evidence
    if not skipped:
        import copy
        if not isinstance(cfg, dict):
            raise ValueError(
                "preflight evidence needs the exact execution config")
        plans = (list(representative_plans)
                 if representative_plans is not None
                 else list(getattr(
                     args, "_preflight_representative_plans", []) or []))
        if not plans:
            raise ValueError(
                "preflight evidence needs exact representative plans")
        from .runner import RunConfig, _preflight_binding_for_rows
        clean = {key: value for key, value in cfg.items()
                 if not key.startswith("_")}
        binding, binding_digest = _preflight_binding_for_rows(
            RunConfig(**copy.deepcopy(clean)), plans,
            list(getattr(args, "_preflight_request_rows", []) or []))
        evidence.update(
            evidence_mode="carried_setup_rows",
            binding=binding,
            binding_sha256=binding_digest,
        )
    args._preflight_evidence = evidence
    return evidence


def _freeze_and_prevalidate_cli_config(cfg: dict, directory: Path):
    """Freeze local files once, then validate the exact endpoint-free view."""
    import dataclasses
    from .runner import (
        RunConfig,
        _snapshot_run_inputs,
        prevalidate_run_inputs,
    )

    clean = {key: value for key, value in cfg.items()
             if not key.startswith("_")}
    public_rc = RunConfig(**clean)
    frozen_rc, identity = _snapshot_run_inputs(public_rc, directory)
    checked = prevalidate_run_inputs(frozen_rc)

    # A supplied profile is first inspected while building the friendly CLI
    # config and is then frozen here. If it changed in that narrow interval,
    # derive the cap from the frozen view and rebuild without another file read.
    if checked.profile is not None:
        import math
        cap = max(1, int(math.ceil(
            float(checked.profile.output_tokens["p95"]) * 1.5)))
        if frozen_rc.max_output_tokens_cap != cap:
            cfg["max_output_tokens_cap"] = cap
            frozen_rc = dataclasses.replace(
                frozen_rc, max_output_tokens_cap=cap)
            checked = prevalidate_run_inputs(
                frozen_rc, reuse_source=checked)

    expectations = {
        key: {"sha256": item["sha256"], "bytes": item["bytes"]}
        for key, item in identity.items()
    }
    cfg["input_expectations"] = expectations
    frozen_rc = dataclasses.replace(
        frozen_rc, input_expectations=expectations)
    checked.rc = frozen_rc
    if checked.workload is not None:
        checked.workload.rc = frozen_rc
    return dataclasses.asdict(frozen_rc), checked


def _input_validation_refusal(exc: BaseException, *, json_mode: bool = False) \
        -> int:
    """Render one clean, content-safe refusal before any endpoint traffic."""
    from .artifacts import redact_secrets

    message = str(redact_secrets(str(exc)))
    if json_mode:
        print(json.dumps({
            "passed": False,
            "stage": "input_validation",
            "exit_code": 2,
            "error": message,
        }, allow_nan=False))
    else:
        print("[input-validation] REFUSED before endpoint traffic: " + message)
    return 2


def cmd_benchmark(args) -> int:
    """One command from an endpoint URL to a report.

    The previous path was: author a profile JSON, run quickstart, edit the
    config, run it. Three of those four steps are things a person should not
    have to do to answer "does this endpoint meet my latency target".
    """
    import copy
    from .immutable_config import publish_legacy_copy, write_immutable_json
    from .runner import RunConfig, enforce_exact_analysis_envelope, run
    import tempfile

    json_mode = getattr(args, "format", "text") == "json"
    try:
        args.probe_extra_body = _validated_probe_candidates(
            getattr(args, "probe_extra_body", None) or [])
    except (TypeError, ValueError, OverflowError) as exc:
        return _input_validation_refusal(exc, json_mode=json_mode)
    cfg = _benchmark_config(args)
    args._preflight_request_rows = []
    with tempfile.TemporaryDirectory(
            prefix="traffic-replay-cli-inputs-") as frozen_dir:
        if json_mode:
            try:
                work_cfg, prevalidated = _freeze_and_prevalidate_cli_config(
                    cfg, Path(frozen_dir))
            except (OSError, TypeError, ValueError, RuntimeError,
                    OverflowError) as exc:
                return _input_validation_refusal(exc, json_mode=True)
        else:
            try:
                work_cfg, prevalidated = _freeze_and_prevalidate_cli_config(
                    cfg, Path(frozen_dir))
            except (OSError, TypeError, ValueError, RuntimeError,
                    OverflowError) as exc:
                return _input_validation_refusal(exc)

        # Include the maximum logical setup population (the two preflight
        # representatives plus every explicitly requested reasoning probe)
        # before credential lookup or setup traffic. The runner rechecks the
        # actual carried rows, but that is too late to authorize preflight.
        try:
            setup_rows = len(_quota_setup_plans(
                work_cfg, args,
                representative_plans=prevalidated.representative_plans))
            enforce_exact_analysis_envelope(
                prevalidated, setup_rows=setup_rows,
                context="benchmark including possible setup traffic")
            schedule = (prevalidated.full_schedule
                        or prevalidated.sizing_schedule_ceiling)
            measured_rows = len(schedule["timestamps"])
            measured_label = (
                "measured replay" if prevalidated.full_schedule is not None
                else "measured replay ceiling before sizing")
            calibration_rows = int(work_cfg.get("calibrate_n") or 0)
            count_stream = sys.stderr if json_mode else sys.stdout
            print(
                f"[plan] {measured_label}={measured_rows:,}; "
                f"setup/preflight up to={setup_rows:,}; "
                f"calibration={calibration_rows:,}; logical total up to="
                f"{measured_rows + setup_rows + calibration_rows:,}",
                file=count_stream)
            print(
                "[plan] retries can add physical POST attempts; the runtime "
                "quota guard admits each attempt separately.",
                file=count_stream)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            return _input_validation_refusal(exc, json_mode=json_mode)

        if json_mode:
            import contextlib
            with contextlib.redirect_stdout(sys.stderr):
                quota_refused = _quota_gate(
                    work_cfg, args, prevalidated=prevalidated)
        else:
            quota_refused = _quota_gate(
                work_cfg, args, prevalidated=prevalidated)
        if quota_refused is not None:
            if json_mode:
                print(json.dumps({"passed": False,
                                  "stage": "quota_plan",
                                  "exit_code": quota_refused,
                                  "quota_plan": args._quota_plan},
                                 allow_nan=False))
            return quota_refused
        if not args.skip_preflight:
            representatives = prevalidated.representative_plans
            _claim_setup_traffic_evidence(
                work_cfg, args, command="benchmark")
            try:
                if json_mode:
                    import contextlib
                    with contextlib.redirect_stdout(sys.stderr):
                        refused = _check_preflight(
                            work_cfg, args,
                            representative_plans=representatives)
                else:
                    refused = _check_preflight(
                        work_cfg, args,
                        representative_plans=representatives)
                preflight_gate = _finalize_preflight_evidence(
                    args, refused, cfg=work_cfg,
                    representative_plans=getattr(
                        args, "_preflight_representative_plans", None))
                setup_path = _seal_setup_traffic_evidence(
                    work_cfg, args,
                    outcome=preflight_gate["outcome"],
                    exit_code=refused)
                args._setup_artifact_reference = _setup_artifact_reference(
                    setup_path, preflight_gate)
            except BaseException as exc:
                _abort_setup_traffic_evidence(args, exc)
                raise
            setup_stream = sys.stderr if json_mode else sys.stdout
            print(f"[preflight] setup traffic evidence sealed at "
                  f"{setup_path}", file=setup_stream)
            if refused is not None:
                if json_mode:
                    print(json.dumps({"passed": False,
                                      "stage": "preflight",
                                      "exit_code": refused,
                                      "setup_artifact": str(setup_path)},
                                     allow_nan=False))
                return refused

        # Preflight can legitimately select first-visible TTFT. Preserve that
        # metric-only mutation in both the frozen execution view and public
        # rerun config; workload bytes/plans remain the validated objects above.
        if "ttft_definition" in work_cfg:
            cfg["ttft_definition"] = work_cfg["ttft_definition"]
        if work_cfg.get("endpoint", {}).get("include_usage") is False:
            cfg["endpoint"]["include_usage"] = False

        # Validate the final configuration before writing a rerun file or
        # starting the measured workload. The runner receives the private
        # frozen paths; the saved config retains the user's durable paths.
        rc = RunConfig(**work_cfg)
        saved = write_immutable_json(args.out_dir, "run-config", cfg)
        legacy_matches = publish_legacy_copy(
            saved, Path(args.out_dir) / "run-config.json")
        run_options = {}
        if args._preflight_request_rows:
            run_options["prior_request_rows"] = args._preflight_request_rows
        if not args.skip_preflight:
            run_options["preflight_gate"] = copy.deepcopy(
                args._preflight_evidence)
            run_options["setup_artifact_reference"] = copy.deepcopy(
                args._setup_artifact_reference)
        if getattr(args, "_runtime_quota_guard", None) is not None:
            run_options["runtime_quota_guard"] = args._runtime_quota_guard
        out = run(rc, quiet=json_mode, **run_options)
        code = _finish(out, getattr(args, "fail_on", "miss"),
                       getattr(args, "format", "text"))
        stream = sys.stderr if json_mode else sys.stdout
        print(file=stream)
        if not legacy_matches:
            print(f"note: {Path(args.out_dir) / 'run-config.json'} belongs to an "
                  "earlier run and was preserved unchanged.", file=stream)
        print(f"config saved to {saved}. reruns refuse if any external input "
              "bytes changed:", file=stream)
        print(f"  python3 -m traffic_replay run --config {saved}", file=stream)
        if getattr(args, "verify_after_run", False):
            from .run_verification import (
                create_run_verification_receipt,
                verify_run_receipt,
            )
            source_dir = Path(out["out_dir"])
            receipt_dir = create_run_verification_receipt(
                source_dir, str(source_dir) + "-verification")
            receipt = verify_run_receipt(receipt_dir, verify_source=False)
            decision = receipt["decision"]
            codes = {
                "integrity": decision["evidence_integrity"]["code"],
                "measurement": decision["measurement_validity"]["code"],
                "sla": decision["customer_sla"]["code"],
                "quota": decision["quota_state"]["code"],
                "capacity": decision["endpoint_capacity"]["code"],
            }
            completion = {
                "measured_replay_requests": decision["tested_load"][
                    "measured_replay_requests"],
                "captured_setup_and_calibration_requests": (
                    decision["tested_load"]["captured_quota_request_rows"]
                    - decision["tested_load"]["measured_replay_requests"]),
                "source_run_dir": str(source_dir),
                "verification_receipt_dir": str(receipt_dir),
                "authoritative_html": str(
                    Path(receipt_dir) / "verified-report.html"),
                "authoritative_markdown": str(
                    Path(receipt_dir) / "verified-report.md"),
                "decision_codes": codes,
            }
            if json_mode:
                print(json.dumps({"post_run_verification": completion},
                                 allow_nan=False), file=sys.stderr)
            else:
                print("\nAUTHORITATIVE VERIFIED RESULT", file=stream)
                for key, value in completion.items():
                    if key == "decision_codes":
                        print("decisions: " + ", ".join(
                            f"{name}={state}"
                            for name, state in value.items()), file=stream)
                    else:
                        print(f"{key}: {value}", file=stream)
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
    import tempfile
    import time as _time
    from .artifacts import redact_secrets
    from .runner import (
        RunConfig,
        exact_analysis_row_counts,
        prevalidate_run_inputs,
        run,
    )
    from .schedule import validate_exact_analysis_capacity
    from .sweep_artifacts import (
        SweepArtifacts, classify_sweep_rung, rate_label,
        sweep_acceptance_policy,
    )

    if args.duration <= 0:
        raise SystemExit("--duration must be a positive number of seconds")
    if args.cooldown < 0:
        raise SystemExit("--cooldown cannot be negative")
    try:
        args.probe_extra_body = _validated_probe_candidates(
            getattr(args, "probe_extra_body", None) or [])
    except (TypeError, ValueError, OverflowError) as exc:
        return _input_validation_refusal(exc)
    rates = _rungs(args.rate)
    sweep_started = _time.monotonic()
    base = _benchmark_config(args)
    # The ladder controls arrival rate directly. It must not also run the
    # unloaded concurrency-sizing pass, which would overwrite every rung.
    base.pop("sizing_concurrency", None)
    base["title"] = args.title or f"{args.endpoint} rate sweep"
    # A ladder cannot recalibrate its request construction independently at
    # each rung and still claim that only arrival rate changed. Calibrate once
    # in a separate benchmark, then pass the measured characters/token here.
    base["calibrate_n"] = 0
    base["cpt"] = args.cpt
    # A sweep base is the invariant configuration at its first actual rung,
    # not RunConfig's unrelated bursty defaults.  The verifier overwrites
    # these four fields for every rung, and an explicit first-rate base keeps a
    # long low-rate ladder from failing capacity validation after preflight.
    base.update(
        qps_base=rates[0], qps_burst=rates[0],
        qps_min=rates[0], qps_max=rates[0], rate_scale=1.0)
    args._preflight_request_rows = []
    args._preflight_evidence = {
        "skipped": bool(args.skip_preflight),
        "attempted": 0,
        "reachable": 0,
        "readable": 0,
        "reasoning_probe_requests": 0,
        "outcome": ("skipped" if args.skip_preflight
                    else "preflight_state_unknown"),
        "force_requested": bool(getattr(args, "force", False)),
        "gate_satisfied": False,
    }
    frozen_inputs = tempfile.TemporaryDirectory(
        prefix="traffic-replay-sweep-inputs-")
    try:
        try:
            work_base, first_prevalidated = \
                _freeze_and_prevalidate_cli_config(
                    base, Path(frozen_inputs.name))
            prevalidated_rungs = [first_prevalidated]
            setup_rows = len(_quota_setup_plans(
                work_base, args,
                representative_plans=(
                    first_prevalidated.representative_plans)))
            totals = exact_analysis_row_counts(first_prevalidated)
            validate_exact_analysis_capacity(
                **totals, setup_rows=setup_rows,
                context="complete sweep including possible setup traffic")
            for rate in rates[1:]:
                check_cfg = copy.deepcopy(work_base)
                check_cfg.update(
                    qps_base=rate, qps_burst=rate,
                    qps_min=rate, qps_max=rate, rate_scale=1.0,
                    duration_s=args.duration,
                    out_dir=str(Path(work_base["out_dir"])
                                / f"rate_{rate_label(rate)}"),
                    title=work_base["title"]
                          + f" @ {rate_label(rate)} requests/second")
                checked_rung = prevalidate_run_inputs(
                    RunConfig(**check_cfg),
                    reuse_source=first_prevalidated)
                prevalidated_rungs.append(checked_rung)
                for field, value in exact_analysis_row_counts(
                        checked_rung).items():
                    totals[field] += value
                validate_exact_analysis_capacity(
                    **totals, setup_rows=setup_rows,
                    context=("complete sweep including possible setup "
                             "traffic"))
            acceptance = work_base.get("acceptance_targets")
            if acceptance is None and first_prevalidated.profile is not None:
                acceptance = (first_prevalidated.profile.extra or {}).get(
                    "acceptance_targets")
            policy_ok, policy_reason = sweep_acceptance_policy(acceptance)
            if not policy_ok and not args.diagnostic_only:
                raise ValueError(
                    "production capacity sweep refused before endpoint "
                    f"traffic: {policy_reason}. Provide both a customer "
                    "latency target and --success-rate, or pass "
                    "--diagnostic-only to run every rung without publishing "
                    "a held rate or capacity conclusion")
        except (OSError, TypeError, ValueError, RuntimeError,
                OverflowError) as exc:
            frozen_inputs.cleanup()
            return _input_validation_refusal(exc)

        quota_refused = _quota_gate(
            work_base, args, rates=rates,
            prevalidated_rungs=prevalidated_rungs)
        if quota_refused is not None:
            frozen_inputs.cleanup()
            return quota_refused
        if not args.skip_preflight:
            _claim_setup_traffic_evidence(
                work_base, args, command="sweep")
            try:
                refused = _check_preflight(
                    work_base, args,
                    representative_plans=(
                        first_prevalidated.representative_plans))
                preflight_gate = _finalize_preflight_evidence(
                    args, refused, cfg=work_base,
                    representative_plans=getattr(
                        args, "_preflight_representative_plans", None))
                setup_path = _seal_setup_traffic_evidence(
                    work_base, args,
                    outcome=preflight_gate["outcome"],
                    exit_code=refused)
                args._setup_artifact_reference = _setup_artifact_reference(
                    setup_path, preflight_gate)
            except BaseException as exc:
                _abort_setup_traffic_evidence(args, exc)
                raise
            print(f"[preflight] setup traffic evidence sealed at "
                  f"{setup_path}")
            if refused is not None:
                frozen_inputs.cleanup()
                return refused
        # Preflight may legitimately select first-visible TTFT. Freeze and
        # validate the final config only after that mutation.
        if "ttft_definition" in work_base:
            base["ttft_definition"] = work_base["ttft_definition"]
        if work_base.get("endpoint", {}).get("include_usage") is False:
            base["endpoint"]["include_usage"] = False
        RunConfig(**work_base)
        sweep_artifact = SweepArtifacts.claim(
            args.out_dir, base, identity_config=work_base)
    except BaseException:
        frozen_inputs.cleanup()
        raise
    out_root = sweep_artifact.path
    args._sweep_endpoint_path = base["endpoint"]["path"]
    args._cooldown_events = 0
    args._cooldown_records = []
    args._sweep_planned_rates = list(rates)

    def apply_cooldown(after: str) -> None:
        if not args.cooldown:
            return
        wall_start = _time.time()
        mono_start = _time.monotonic()
        _time.sleep(args.cooldown)
        mono_end = _time.monotonic()
        wall_end = _time.time()
        args._cooldown_events += 1
        args._cooldown_records.append({
            "after": after,
            "requested_s": float(args.cooldown),
            "started_at_unix": wall_start,
            "finished_at_unix": wall_end,
            "elapsed_s": max(0.0, mono_end - mono_start),
        })

    nominal = (len(rates) * args.duration
               + max(0, len(rates) - 1) * args.cooldown
               + (args.cooldown if not args.skip_preflight else 0))
    print(f"[sweep] {len(rates)} rungs: "
          + ", ".join(rate_label(r) for r in rates)
          + " requests/second")
    print(f"[sweep] {args.duration}s of offered load each")
    print(f"[sweep] calibration requests per rung: 0; fixed at "
          f"{args.cpt:g} characters/token. Measure this once in a separate "
          "benchmark before the real sweep.")
    if args.cooldown:
        print(f"[sweep] {args.cooldown}s spacing after preflight and between "
              "rungs. This does not prove quota or cache reset.")
    print(f"[sweep] nominal scheduled time {nominal}s if every rung runs; "
          "preflight requests, response drain and report writing are extra")
    print()

    rungs: list[dict] = []
    try:
        if args.cooldown and not args.skip_preflight:
            apply_cooldown("preflight")
        for i, rate in enumerate(rates):
            public_cfg = copy.deepcopy(base)
            public_cfg.update(
                qps_base=rate, qps_burst=rate, qps_min=rate,
                qps_max=rate, rate_scale=1.0,
                duration_s=args.duration,
                out_dir=str(out_root / f"rate_{rate_label(rate)}"),
                title=base["title"]
                      + f" @ {rate_label(rate)} requests/second")
            execution_cfg = copy.deepcopy(work_base)
            execution_cfg.update(
                qps_base=rate, qps_burst=rate, qps_min=rate,
                qps_max=rate, rate_scale=1.0,
                duration_s=args.duration,
                out_dir=public_cfg["out_dir"], title=public_cfg["title"])
            rung_rc = RunConfig(**execution_cfg)
            rung_root = Path(public_cfg["out_dir"])
            rung_root.mkdir(parents=True, exist_ok=True)
            (rung_root / "run-config.json").write_text(
                json.dumps(public_cfg, indent=2) + "\n")
            print(f"[sweep] rung {i + 1}/{len(rates)}: "
                  f"{rate_label(rate)} rps")
            rung_started = _time.monotonic()
            source_position = None
            accounting = {
                field: None for field in (
                    "request_rows", "replay_rows", "calibration_rows",
                    "sizing_rows", "preflight_rows", "probe_rows",
                    "other_rows", "unknown_attempt_rows")
            }
            try:
                run_options = {}
                if i == 0 and args._preflight_request_rows:
                    run_options["prior_request_rows"] = \
                        args._preflight_request_rows
                if not args.skip_preflight:
                    inherited_gate = copy.deepcopy(args._preflight_evidence)
                    if i > 0:
                        inherited_gate["evidence_mode"] = \
                            "inherited_setup_artifact"
                    run_options["preflight_gate"] = inherited_gate
                    run_options["setup_artifact_reference"] = copy.deepcopy(
                        args._setup_artifact_reference)
                if getattr(args, "_runtime_quota_guard", None) is not None:
                    run_options["runtime_quota_guard"] = \
                        args._runtime_quota_guard
                out = run(rung_rc, quiet=False, **run_options)
                s, source_position = sweep_artifact.add_rung(
                    rate, out["out_dir"], expected_summary=out["summary"])
                accounting = sweep_artifact.rung_accounting(source_position)
                decision = classify_sweep_rung(s)
                kind, verdict_text = decision["kind"], decision["text"]
                out_dir = Path(out["out_dir"]).resolve(strict=True).relative_to(
                    out_root.resolve(strict=True)).as_posix()
            except Exception as exc:
                kind = "invalid"
                safe_error = redact_secrets(str(exc))
                verdict_text = (f"rung failed before a verified report: "
                                f"{type(exc).__name__}: {safe_error}")
                s = {}
                out_dir = rung_root.relative_to(out_root).as_posix()
                decision = {
                    "state": "INVALID", "quota_status": "UNKNOWN",
                    "first_event_definition": work_base.get(
                        "ttft_definition", "first_content"),
                    "latency_metric": None, "latency_basis": None,
                    "latency_n": None, "latency_p50": None,
                    "latency_p95": None, "e2e_metric": None,
                    "e2e_basis": None, "e2e_n": None,
                    "e2e_p50": None, "e2e_p95": None,
                    "success_rate_target": None,
                    "success_rate_actual": None,
                    "success_rate_wilson_lower_95": None,
                    "success_rate_statistically_demonstrated": None,
                    "request_start_lateness_p95": None,
                    "dispatch_lag_p95": None,
                    "response_identity_status": None,
                    "endpoint_metadata_stability": None,
                    "runtime_quota_admission_status": None,
                    "runtime_quota_guard_id": None,
                }
            cc = s.get("concurrency") or {}
            rungs.append({
                "rate": rate, "kind": kind, "text": verdict_text,
                **decision,
                "dir": out_dir, "source_position": source_position,
                "held": cc.get("in_flight_p50"),
                "achieved_rps": (s.get("arrivals") or {}).get(
                    "achieved_qps_overall"),
                "err": s.get("error_rate"),
                # Backward-compatible aliases now point to the configured
                # first-event and E2E populations, not hard-wired raw TTFT.
                "ttft_p50": decision.get("latency_p50"),
                "ttft_p95": decision.get("latency_p95"),
                "e2e_p50": decision.get("e2e_p50"),
                "wall_s": _time.monotonic() - rung_started,
                **accounting,
            })
            print(f"[sweep] rung {i + 1}: {kind.upper()} {verdict_text[:90]}")
            print()
            state = decision["state"]
            if state in {"INVALID", "QUOTA_LIMITED"}:
                print(f"[sweep] stopping: rung {i + 1} was {state}; higher "
                      "paid load is unsafe and cannot repair this evidence.")
                break
            if state == "FAIL" and not args.no_early_stop:
                print(f"[sweep] stopping: rung {i + 1} definitively missed "
                      "the configured acceptance policy. pass "
                      "--no-early-stop only to diagnose non-monotonicity.")
                break
            if state == "NO_CRITERION" and not args.diagnostic_only:
                # Defensive: the pre-traffic gate above should make this
                # unreachable, but never continue paid traffic on policy drift.
                print("[sweep] stopping: no publishable acceptance criterion")
                break
            if args.cooldown and i + 1 < len(rates):
                apply_cooldown(f"rung_{rate_label(rate)}")

        args._sweep_wall_s = _time.monotonic() - sweep_started
        return _sweep_report(rungs, sweep_artifact, args)
    finally:
        sweep_artifact.close()
        frozen_inputs.cleanup()


def _sweep_report(rungs: list[dict], sweep_artifact, args) -> int:
    """Render and seal the one conclusion derivable from rung evidence."""
    from .sweep_artifacts import render_sweep_report, sweep_outcome

    planned_rates = list(getattr(args, "_sweep_planned_rates",
                                 [rung["rate"] for rung in rungs]))
    attempted_rates = [float(rung["rate"]) for rung in rungs]
    omitted_rates = planned_rates[len(attempted_rates):]
    if not omitted_rates:
        termination_reason = "completed_planned_ladder"
    else:
        termination_reason = {
            "INVALID": "invalid_measurement",
            "QUOTA_LIMITED": "quota_limited",
            "FAIL": "definitive_sla_failure_early_stop",
            "NO_CRITERION": "missing_capacity_criterion",
        }.get(rungs[-1].get("state"), "stopped_before_planned_ladder_end")
    quota_evidence_fn = getattr(sweep_artifact, "pooled_quota_evidence", None)
    sweep_quota = (quota_evidence_fn() if callable(quota_evidence_fn) else {
        "traffic_population": "report_sink_has_no_manifest_bound_rows",
        "request_rows": sum(int(rung.get("request_rows") or 0)
                            for rung in rungs),
        "http_429_count": 0,
        "quota_status": "UNKNOWN",
        "observed_rate_windows": {},
        "configured_rate_limits": None,
        "runtime_quota_admission": {
            "status": "not_configured",
            "guard_ids": [],
            "denied_rows": 0,
            "denied_attempts_in_captured_rows": 0,
            "invariant_errors": [],
        },
    })
    complete_preflight = getattr(args, "_preflight_evidence", {
        "skipped": True,
        "attempted": 0,
        "reachable": 0,
        "readable": 0,
        "reasoning_probe_requests": 0,
        "outcome": "skipped",
        "force_requested": False,
        "gate_satisfied": False,
    })
    # The sweep's decision schema consumes the compact gate state. Every rung
    # retains the full cryptographic binding and setup-artifact reference in
    # its own sealed run; do not duplicate that large evidence in this report.
    preflight_decision = {
        key: complete_preflight[key] for key in (
            "skipped", "attempted", "reachable", "readable",
            "reasoning_probe_requests", "outcome", "force_requested",
            "gate_satisfied")
    }
    context = {
        "endpoint": getattr(args, "_sweep_endpoint_path", args.endpoint),
        "sweep_wall_s": getattr(args, "_sweep_wall_s", 0.0),
        "cooldown_s": float(getattr(args, "cooldown", 0)),
        "cooldown_events": int(getattr(args, "_cooldown_events", 0)),
        "cooldown_records": list(getattr(args, "_cooldown_records", [])),
        "planned_rates": planned_rates,
        "attempted_rates": attempted_rates,
        "omitted_rates": omitted_rates,
        "progression_policy": {
            "early_stop_on_definitive_fail": not bool(getattr(
                args, "no_early_stop", False)),
            "diagnostic_only": bool(getattr(args, "diagnostic_only", False)),
            "invalid_or_quota_always_stops": True,
        },
        "termination_reason": termination_reason,
        "sweep_quota_evidence": sweep_quota,
        "preflight": preflight_decision,
    }
    outcome = sweep_outcome(rungs, context["preflight"])
    body = render_sweep_report(rungs, context)
    path = sweep_artifact.path / "sweep.md"
    sweep_artifact.seal(
        body, rungs, exit_code=outcome["exit_code"],
        highest_held_rate=outcome["highest_held_rate"],
        report_context=context)
    print()
    print(body.rstrip())
    print()
    print(f"written to {path}")
    return outcome["exit_code"]


def cmd_verify_sweep(args) -> int:
    """Verify the complete sweep evidence chain without endpoint traffic."""
    from .artifacts import redact_secrets
    from .sweep_artifacts import verify_sweep_output

    try:
        manifest = verify_sweep_output(args.sweep_dir)
    except (OSError, ValueError) as exc:
        error = str(redact_secrets(str(exc)))
        if args.format == "json":
            print(json.dumps({"verified": False, "error": error},
                             allow_nan=False))
        else:
            print(f"INVALID SWEEP ARTIFACT: {error}", file=sys.stderr)
        return 2
    result = {
        "verified": True,
        "integrity_verified": True,
        "artifact_id": manifest["artifact_id"],
        "sweep_valid": manifest["sweep_valid"],
        "result_exit_code": manifest["exit_code"],
        "highest_held_rate_requests_per_second": manifest[
            "highest_held_rate_requests_per_second"],
        "highest_sla_passing_tested_rate_requests_per_second": manifest[
            "highest_sla_passing_tested_rate_requests_per_second"],
        "capacity_conclusion": manifest["capacity_conclusion"],
        "boundary_status": manifest["boundary_status"],
        "invalid_reasons": manifest["invalid_reasons"],
        "input_count": manifest["input_count"],
        "rung_count": manifest["rung_count"],
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
    elif manifest["sweep_valid"]:
        print("VERIFIED: the sweep report, config, source runs, traffic "
              "counts, experiment state and exit status agree.")
        print(f"artifact: {manifest['artifact_id']}")
        print(f"rungs: {manifest['rung_count']} attempted, "
              f"{manifest['input_count']} internally hash-verified")
        print("highest SLA-passing tested rate: "
              f"{manifest['highest_sla_passing_tested_rate_requests_per_second']}")
        print(f"capacity conclusion: {manifest['capacity_conclusion']}")
    else:
        print("VERIFIED EVIDENCE, INVALID SWEEP: no capacity conclusion.")
        for reason in manifest["invalid_reasons"]:
            print(f"- {reason}")
    # Exit status reports artifact integrity only. An intact but inconclusive
    # experiment is not tampering; callers can inspect capacity_conclusion or
    # the original result_exit_code as a separate policy gate.
    return 0


def cmd_verify_run(args) -> int:
    """Verify one sealed run and write a separate immutable receipt."""
    from .artifacts import redact_secrets
    from .run_verification import (
        create_run_verification_receipt,
        verify_run_receipt,
    )

    try:
        out = create_run_verification_receipt(args.run_dir, args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(redact_secrets(str(exc)))
        if args.format == "json":
            print(json.dumps({"verified": False, "error": error},
                             allow_nan=False))
        else:
            print(f"INVALID RUN ARTIFACT: {error}", file=sys.stderr)
        return 2

    # Re-open through the receipt verifier rather than trusting a normal path
    # read after creation; replacement or symlink races remain verification
    # failures at the CLI boundary too.
    try:
        receipt = verify_run_receipt(out, verify_source=False)
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(redact_secrets(str(exc)))
        if args.format == "json":
            print(json.dumps({"verified": False, "error": error},
                             allow_nan=False))
        else:
            print(f"INVALID VERIFICATION RECEIPT: {error}", file=sys.stderr)
        return 2
    source = receipt["source_run"]
    reconstructibility = receipt["source_reconstructibility"]
    verifier_reconstructibility = receipt[
        "verifier_source_reconstructibility"]
    capacity = receipt["decision"]["endpoint_capacity"]
    result = {
        "verified": True,
        "verification_code": receipt["verification_code"],
        "receipt_dir": str(out),
        "receipt_id": receipt["receipt_id"],
        "source_artifact_id": source["artifact_id"],
        "source_reconstructible": reconstructibility["reconstructible"],
        "source_reconstructibility_reason_codes": reconstructibility[
            "reason_codes"],
        "verifier_source_reconstructible": verifier_reconstructibility[
            "reconstructible"],
        "verifier_source_reconstructibility_reason_codes":
            verifier_reconstructibility["reason_codes"],
        "capacity_code": capacity["code"],
        "capacity_label": capacity["label"],
        "digital_signature": False,
        "assurance": receipt["assurance"],
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print("VERIFIED INTERNAL HASH CONSISTENCY: every canonical v3 "
              "artifact and the completion chain matched.")
        print(f"receipt: {out}")
        print(f"source artifact: {source['artifact_id']}")
        state = "yes" if reconstructibility["reconstructible"] else "no"
        print(f"source reconstructible: {state}")
        verifier_state = (
            "yes" if verifier_reconstructibility["reconstructible"] else "no")
        print(f"verifier source reconstructible: {verifier_state}")
        print(f"capacity: {capacity['code']} - {capacity['label']}")
        print(receipt["assurance"])
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
    ep: dict = {
        "base_url": args.host.rstrip("/"),
        "path": path,
        "temperature": getattr(args, "temperature", 0.0),
    }
    endpoint_adapter = getattr(args, "endpoint_adapter", None)
    if endpoint_adapter:
        ep["adapter"] = endpoint_adapter
    if args.auth_profile:
        ep["auth_profile"] = args.auth_profile
    else:
        ep["auth_token_env"] = args.token_env
    if args.model:
        ep["model"] = args.model
    production_policy = getattr(
        args, "production_connection_policy", None)
    if production_policy is not None:
        ep["production_connection_policy"] = production_policy

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
    calibration_requests = getattr(args, "calibrate_requests", None)
    if calibration_requests is not None:
        cfg["calibrate_n"] = calibration_requests
    if args.max_output_tokens is not None:
        cfg["max_output_tokens_cap"] = args.max_output_tokens
    if getattr(args, "ttft_definition", None) is not None:
        cfg["ttft_definition"] = args.ttft_definition

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
        # quickstart runs the generated config directly, without the benchmark
        # command's separate setup gate. Reserve calibration and sizing rows
        # and derive its replay ceiling from the same exact-analysis envelope.
        _apply_cli_sizing_resource_ceiling(cfg, setup_rows=0)
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
        print("no acceptance targets given, so the scorecard will fall back "
              "to the "
              "profile's. pass --ttft-p95 and --ttfg-p95 (and the other "
              "quantiles) to score against yours.")
    return 0


def _databricks_profile_host(name: str) -> str:
    """Read only the normalized host from one named Databricks CLI profile."""
    import configparser
    from .client import validate_bearer_transport

    cfg_path = Path(os.environ.get(
        "DATABRICKS_CONFIG_FILE", Path.home() / ".databrickscfg"))
    parser = configparser.ConfigParser(
        interpolation=None, default_section="__TRAFFIC_REPLAY_DEFAULTS_DISABLED__")
    try:
        read = parser.read(cfg_path)
    except (OSError, configparser.Error) as exc:
        raise ValueError(
            f"could not read Databricks config {cfg_path}: "
            f"{type(exc).__name__}") from None
    if not read or parser.defaults() or not parser.has_section(name):
        raise ValueError(
            f"Databricks auth profile {name!r} was not found in {cfg_path}")
    host = (parser[name].get("host") or "").strip()
    try:
        scheme, hostname, port = validate_bearer_transport(host)
    except ValueError as exc:
        raise ValueError(
            f"Databricks auth profile {name!r} has an invalid workspace host") \
            from exc
    default = 443 if scheme == "https" else 80
    return f"{scheme}://{hostname}" + (f":{port}" if port != default else "")


def _ready_chat_endpoints(profile: str) -> list[dict]:
    """Discover endpoint metadata without reading or printing credentials."""
    import subprocess
    from .json_input import loads_strict

    try:
        completed = subprocess.run(
            ["databricks", "serving-endpoints", "list", "--profile", profile,
             "--output", "json"],
            check=True, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise ValueError(
            "Databricks CLI was not found; install it before init-databricks") \
            from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, "stderr", "") or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError("Databricks endpoint discovery failed" + suffix) \
            from None
    try:
        payload = loads_strict(completed.stdout)
    except ValueError:
        raise ValueError("Databricks endpoint discovery returned invalid JSON") \
            from None
    if isinstance(payload, dict):
        payload = payload.get("endpoints", payload.get("serving_endpoints", []))
    if not isinstance(payload, list):
        raise ValueError("Databricks endpoint discovery returned no endpoint list")
    ready = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        state = item.get("state") if isinstance(item.get("state"), dict) else {}
        ready_state = state.get("ready", item.get("ready"))
        task = str(item.get("task") or item.get("task_type") or "").lower()
        config = item.get("config") if isinstance(item.get("config"), dict) else {}
        entities = config.get("served_entities") or config.get("served_models") or []
        entity_tasks = " ".join(
            str(entity.get("task") or entity.get("task_type") or "").lower()
            for entity in entities if isinstance(entity, dict))
        chat_known = "chat" in task or "chat" in entity_tasks
        foundation = any(
            isinstance(entity, dict) and (
                entity.get("foundation_model") or entity.get("entity_name"))
            for entity in entities)
        if str(ready_state).upper() == "READY" and (chat_known or foundation):
            ready.append({
                "name": item["name"],
                "task": task or entity_tasks or "foundation-model",
                "route_optimized": bool(item.get("route_optimized", False)),
            })
    return sorted(ready, key=lambda item: item["name"])


def cmd_init_databricks(args) -> int:
    """Discover a named workspace and emit a bounded editable starter run."""
    try:
        host = _databricks_profile_host(args.auth_profile)
        endpoints = _ready_chat_endpoints(args.auth_profile)
    except ValueError as exc:
        raise SystemExit(f"init-databricks refused: {exc}") from exc
    if not endpoints:
        raise SystemExit("init-databricks found no READY chat/foundation endpoints")
    names = [item["name"] for item in endpoints]
    endpoint = args.endpoint
    if endpoint is None:
        if len(names) == 1:
            endpoint = names[0]
        elif sys.stdin.isatty():
            print("READY chat/foundation endpoints:")
            for index, item in enumerate(endpoints, start=1):
                print(f"  {index}. {item['name']} ({item['task']})")
            try:
                selected = int(input("Select endpoint number: "))
                endpoint = names[selected - 1]
            except (ValueError, IndexError, EOFError):
                raise SystemExit("init-databricks requires a valid selection") \
                    from None
        else:
            raise SystemExit(
                "multiple READY endpoints found; rerun with --endpoint NAME. "
                "Candidates: " + ", ".join(names))
    if endpoint not in names:
        raise SystemExit(
            f"endpoint {endpoint!r} is not a discovered READY chat/foundation "
            "endpoint; candidates: " + ", ".join(names))

    out = Path(args.out)
    timestamps = "10\n30\n50\n"
    from .immutable_config import write_immutable_text
    trace = write_immutable_text(
        args.out_dir, "timestamps", timestamps)
    cfg = {
        "profile_path": str((Path(__file__).resolve().parent.parent /
                             "configs/profile_validation_small.json")),
        "endpoint": {
            "base_url": host,
            "path": f"/serving-endpoints/{endpoint}/invocations",
            "auth_profile": args.auth_profile,
            "adapter": "openai.chat_completions.sse/v1",
            "temperature": 0.0,
        },
        "timestamps_file": str(trace),
        "duration_s": 60,
        "calibrate_n": 0,
        "max_concurrency": 1,
        "max_pending_requests": 1,
        "out_dir": args.out_dir,
        "title": f"3-request Databricks diagnostic starter, {endpoint}",
        "label": ("LOW-COST DIAGNOSTIC STARTER - NOT CAPACITY EVIDENCE; "
                  "review workload, quotas, and acceptance targets first"),
    }
    from .runner import RunConfig
    RunConfig(**cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
    print(f"workspace: {host}")
    print(f"endpoint: {endpoint}")
    print("planned measured requests: 3; calibration: 0")
    print("benchmark preflight adds two representative requests; retries may "
          "add physical attempts and provider quota is not reserved")
    print(f"starter config: {out}")
    print("validate: python3 -m traffic_replay validate --port 0 --format json")
    print("run starter: python3 -m traffic_replay run --config " + str(out))
    print("guided benchmark: python3 -m traffic_replay benchmark --auth-profile "
          f"{args.auth_profile!r} --host {host!r} --endpoint {endpoint!r} "
          "--requests 3 --duration 60 --calibrate-requests 0 "
          "--verify-after-run")
    print("verification/report paths are printed by --verify-after-run")
    return 0


def main(argv=None) -> int:
    from . import __version__

    ap = argparse.ArgumentParser(prog="traffic_replay")
    ap.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}")
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
        "adapters", help="list installed, versioned endpoint wire adapters")
    s.add_argument("--format", choices=("text", "json"), default="text")
    s.set_defaults(fn=cmd_adapters)

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
    s.add_argument("--fixed-rate", type=float, default=None,
                   help="requests/second known before traffic starts; required "
                        "for a quota-planned benchmark")
    s.add_argument("--concurrency", dest="legacy_concurrency", type=int,
                   default=None, help=argparse.SUPPRESS)
    s.add_argument("--duration", type=int, default=300,
                   help="seconds. 300 gives five stability windows")
    s.add_argument(
        "--requests", type=int, default=None, metavar="N",
        help="schedule exactly N measured replay requests, evenly spaced "
             "across --duration; cannot be combined with rate or sizing flags")
    s.add_argument(
        "--calibrate-requests", type=_calibration_request_count_arg,
        default=12, metavar="N",
        help="unloaded token-calibration requests before replay (default: 12; "
             "use 0 only when token calibration is intentionally disabled)")
    s.add_argument("--input-tokens", default="10000",
                   help="prompt size as p50 or p50,p95. default 10000")
    s.add_argument("--output-tokens", default="200",
                   help="answer size as p50 or p50,p95. default 200")
    s.add_argument(
        "--cache-fraction", "--cache-hit-rate", dest="cache_fraction",
        default="0.3,0.7",
        help="intended reusable-prefix share of prompt tokens as p50 or "
             "p50,p95, 0 to 1; this is not a request hit probability "
             "(--cache-hit-rate is a compatibility alias)")
    s.add_argument("--prompts", default=None,
                   help="JSONL of your real prompts, instead of synthetic text")
    s.add_argument("--profile", default=None,
                   help="an existing profile JSON, instead of the flags above")
    s.add_argument("--auth-profile", default=None,
                   help="a ~/.databrickscfg PAT, databricks-cli U2M, or "
                        "workspace OAuth M2M profile; standard workspace-"
                        "origin routes only, not route-optimized serving")
    s.add_argument("--token-env", default="DATABRICKS_TOKEN",
                   help="env var holding a bearer token, if not using a profile")
    s.add_argument("--model", default=None,
                   help="only for shared /chat/completions routes")
    _add_temperature_arguments(s)
    s.add_argument(
        "--endpoint-adapter",
        default="openai.chat_completions.sse/v1",
        help="versioned request/response wire contract; list installed "
             "contracts with `traffic-replay adapters`")
    s.add_argument(
        "--production-connection-policy",
        choices=("fresh_http1_per_physical_attempt",), default=None,
        help="declare that the real application opens a fresh HTTP/1.1 "
             "connection for every physical attempt. Omit unless this exact "
             "production behavior is known")
    s.add_argument("--extra-body", default=None,
                   help="provider-documented JSON controls merged into each "
                        "request; model behavior is never inferred from its "
                        "name")
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
                   metavar="FRACTION_IN_(0,1)",
                   help="fraction strictly between 0 and 1, e.g. 0.99")
    s.add_argument(
        "--ttft-definition", choices=("first_content", "first_visible"),
        default=None,
        help="event scored by TTFT targets: first_content is the first "
             "visible, reasoning, or refusal onset; first_visible waits "
             "for user-visible assistant content")
    s.add_argument("--out-dir", default="results/benchmark")
    s.add_argument("--max-concurrency", type=int, default=None,
                   help="worker bound; sizing derives it when omitted")
    s.add_argument("--max-pending-requests", type=int, default=None,
                   help="bound on running plus queued client requests")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument(
        "--rate-limits", dest="rate_limits_file", default=None,
        metavar="JSON_FILE",
        help="dated provider quota snapshot; enables conservative planning "
             "and command-scoped no-wait admission for QPS, QPH, token "
             "windows, and serialized request bytes before every physical "
             "inference POST")
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
    s.add_argument(
        "--verify-after-run", action="store_true",
        help="write a sibling verification receipt and print authoritative "
             "verified report paths after a successful sealed run")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser(
        "sweep",
        help="climb an authorized rate ladder and report the highest tested "
             "rate that held; this is not an endpoint ceiling")
    s.add_argument("--host", required=True)
    s.add_argument("--endpoint", required=True)
    s.add_argument("--rate", default="1:32",
                   help="lo:hi, lo:hi:rungs, or a comma list. requests per "
                        "second. geometric by default, since the interesting "
                        "region is multiplicative")
    s.add_argument("--duration", type=int, default=120,
                   help="seconds per rung")
    s.add_argument("--cooldown", type=int, default=60,
                   help="spacing seconds after preflight and between rungs. "
                        "does not prove quota or cache reset")
    s.add_argument("--cpt", type=float, default=4.0,
                   help="fixed characters/token estimate. measure it once "
                        "in a separate benchmark; per-rung calibration is "
                        "always disabled")
    s.add_argument("--no-early-stop", action="store_true",
                   help="climb every rung even after one fails")
    s.add_argument("--input-tokens", default="10000")
    s.add_argument("--output-tokens", default="200")
    s.add_argument(
        "--cache-fraction", "--cache-hit-rate", dest="cache_fraction",
        default="0.3,0.7",
        help="intended reusable-prefix share of prompt tokens as p50 or "
             "p50,p95, 0 to 1; not a request hit probability")
    s.add_argument("--prompts", default=None)
    s.add_argument("--profile", default=None)
    s.add_argument(
        "--auth-profile", default=None,
        help="a ~/.databrickscfg PAT, databricks-cli U2M, or workspace OAuth "
             "M2M profile; standard workspace-origin routes only, not "
             "route-optimized serving")
    s.add_argument("--token-env", default="DATABRICKS_TOKEN")
    s.add_argument("--model", default=None)
    _add_temperature_arguments(s)
    s.add_argument(
        "--endpoint-adapter",
        default="openai.chat_completions.sse/v1",
        help="versioned request/response wire contract")
    s.add_argument(
        "--production-connection-policy",
        choices=("fresh_http1_per_physical_attempt",), default=None,
        help="declare the real application's exact fresh-HTTP/1.1-per-attempt "
             "behavior; omit for pooled, keep-alive, HTTP/2, or unknown clients")
    s.add_argument(
        "--extra-body", default=None,
        help="provider-documented JSON controls merged into each request; "
             "model behavior is never inferred from its name")
    s.add_argument("--ttft-p50", type=float, default=None)
    s.add_argument("--ttft-p90", type=float, default=None)
    s.add_argument("--ttft-p95", type=float, default=None)
    s.add_argument("--ttft-p99", type=float, default=None)
    s.add_argument("--ttfg-p50", type=float, default=None)
    s.add_argument("--ttfg-p90", type=float, default=None)
    s.add_argument("--ttfg-p95", type=float, default=None)
    s.add_argument("--ttfg-p99", type=float, default=None)
    s.add_argument(
        "--success-rate", type=float, default=None,
        metavar="FRACTION_IN_(0,1)",
        help="fraction strictly between 0 and 1, e.g. 0.99")
    s.add_argument(
        "--ttft-definition", choices=("first_content", "first_visible"),
        default=None,
        help="event scored by TTFT targets: first_content is the first "
             "visible, reasoning, or refusal onset; first_visible waits "
             "for user-visible assistant content")
    s.add_argument(
        "--diagnostic-only", action="store_true",
        help="allow a targetless ladder, run every safe rung, and publish no "
             "held-rate or endpoint-capacity conclusion")
    s.add_argument("--out-dir", default="results/sweep")
    s.add_argument("--max-concurrency", type=int, default=256,
                   help="fixed worker bound reused unchanged at every rung")
    s.add_argument("--max-pending-requests", type=int, default=None,
                   help="bound on running plus queued client requests")
    s.add_argument("--title", default=None)
    s.add_argument("--label", default=None)
    s.add_argument(
        "--rate-limits", dest="rate_limits_file", default=None,
        metavar="JSON_FILE",
        help="dated provider quota snapshot; the whole ladder is budgeted "
             "before preflight traffic and every physical inference POST is "
             "then admitted by the same command-local no-wait guard")
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

    s = sub.add_parser(
        "verify-run",
        help="verify a sealed run and write a separate sealed receipt")
    s.add_argument("run_dir")
    s.add_argument("--out", required=True,
                   help="new receipt directory; collisions get a unique "
                        "sibling and the source run is never modified")
    s.add_argument("--format", choices=("text", "json"), default="text")
    s.set_defaults(fn=cmd_verify_run)

    s = sub.add_parser(
        "verify-sweep",
        help="verify a sealed sweep and re-derive its only valid conclusion")
    s.add_argument("sweep_dir")
    s.add_argument("--format", choices=("text", "json"), default="text")
    s.set_defaults(fn=cmd_verify_sweep)

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
    s.add_argument(
        "--calibrate-requests", type=_calibration_request_count_arg,
        default=12, metavar="N",
        help="unloaded token-calibration requests before replay (default: 12; "
             "use 0 only when token calibration is intentionally disabled)")
    s.add_argument("--auth-profile", default=None,
                   help="a ~/.databrickscfg PAT, databricks-cli U2M, or "
                        "workspace OAuth M2M profile; standard workspace-"
                        "origin routes only, not route-optimized serving")
    s.add_argument("--token-env", default="DATABRICKS_TOKEN",
                   help="env var holding a bearer token, if not using a profile")
    s.add_argument("--model", default=None,
                   help="only for shared /chat/completions routes")
    _add_temperature_arguments(s)
    s.add_argument(
        "--endpoint-adapter",
        default="openai.chat_completions.sse/v1",
        help="versioned request/response wire contract")
    s.add_argument(
        "--production-connection-policy",
        choices=("fresh_http1_per_physical_attempt",), default=None,
        help="declare the real application's exact fresh-HTTP/1.1-per-attempt "
             "behavior; omit when unknown")
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
    s.add_argument("--success-rate", type=float, default=None,
                   metavar="FRACTION_IN_(0,1)",
                   help="acceptance fraction strictly between 0 and 1")
    s.add_argument(
        "--ttft-definition", choices=("first_content", "first_visible"),
        default=None,
        help="event scored by TTFT targets: first_content is the first "
             "visible, reasoning, or refusal onset; first_visible waits "
             "for user-visible assistant content")
    s.add_argument("--out", default="configs/quickstart.json")
    s.set_defaults(fn=cmd_quickstart)

    s = sub.add_parser(
        "init-databricks",
        help="discover READY endpoints from a named profile and write a "
             "bounded diagnostic starter config")
    s.add_argument("--auth-profile", required=True,
                   help="named ~/.databrickscfg workspace profile")
    s.add_argument("--endpoint", default=None,
                   help="READY endpoint name; interactively selected when "
                        "omitted and a terminal is available")
    s.add_argument("--out", default="configs/databricks-starter.json",
                   help="editable starter RunConfig path")
    s.add_argument("--out-dir", default="results/databricks-starter",
                   help="result directory used by the starter config")
    s.set_defaults(fn=cmd_init_databricks)

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
                   help="json prints only the validation comparison object")
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
    try:
        return args.fn(args)
    except Exception as exc:
        # A named profile is a fail-closed credential boundary, but a missing,
        # expired, mixed, or unsafe profile is an expected operator error, not
        # a Python traceback. Keep JSON stdout as exactly one document.
        from .runner import AuthProfileError
        if not isinstance(exc, AuthProfileError):
            raise
        message = str(exc)
        if getattr(args, "format", "text") == "json":
            print(json.dumps({
                "passed": False,
                "stage": "authentication",
                "exit_code": 2,
                "error": message,
            }, allow_nan=False))
        else:
            print(f"authentication failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
