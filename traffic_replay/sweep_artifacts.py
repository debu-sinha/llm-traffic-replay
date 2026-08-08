"""Integrity chain for rate-sweep aggregate evidence.

A sweep is not a loose Markdown file next to several runs. It is a sealed
aggregate whose manifest binds the rendered conclusion, the exact
base configuration, and the already-sealed manifest and summary identity of
every completed rung used to reach that conclusion.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import hmac
import math
import os
from pathlib import Path
import stat
import time
import uuid

from . import __version__
from .aggregate import (
    _artifact_declarations,
    _fsync_directory,
    _fsync_fd,
    _has_path,
    _identity_digest,
    _read_regular_bytes,
    _require_regular,
    _require_run_dir,
    _verify_artifacts,
    _write_compare_fd,
)
from .artifacts import (
    canonical_sha256,
    redact_secrets,
    sanitize_title,
    snapshot_source_state,
    strict_json_dumps,
)


_WRITING_MARKER = ".traffic-replay-writing"
_COMPLETE_MARKER = ".traffic-replay-complete"


def rate_label(value: int | float) -> str:
    """Injective, filesystem-safe rendering of one finite positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid sweep rate: {value!r}")
    text = repr(number)
    return text[:-2] if text.endswith(".0") else text


def _strict_object(raw: bytes, label: str, path: Path) -> dict:
    from .json_input import loads_strict

    try:
        value = loads_strict(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def sweep_outcome(rungs: list[dict]) -> dict:
    """Derive the only valid aggregate result from ordered rung evidence."""
    if not isinstance(rungs, list) or not rungs:
        raise ValueError("a sweep requires at least one rung attempt")
    rates = []
    for position, rung in enumerate(rungs):
        if not isinstance(rung, dict):
            raise ValueError(f"invalid sweep rung record at position {position}")
        rate = rung.get("rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                or not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError(f"invalid sweep rung rate at position {position}")
        rates.append(float(rate))
        if rung.get("kind") not in {"ok", "caution", "miss", "invalid"}:
            raise ValueError(f"invalid sweep verdict at position {position}")
    if any(right <= left for left, right in zip(rates, rates[1:])):
        raise ValueError("sweep rung rates must be strictly increasing")

    unverified = [r for r in rungs if r.get("source_position") is None]
    seen_non_ok = False
    non_monotonic = False
    for rung in rungs:
        if rung["kind"] != "ok":
            seen_non_ok = True
        elif seen_non_ok:
            non_monotonic = True
    good = [r for r in rungs if r["kind"] == "ok"]
    invalid_reasons = []
    if unverified:
        invalid_reasons.append("one or more rung attempts produced no verified report")
    invalid_reports = [
        r for r in rungs
        if r.get("source_position") is not None and r["kind"] == "invalid"]
    if invalid_reports:
        invalid_reasons.append(
            "one or more manifest-bound rung reports are invalid measurements")
    if non_monotonic:
        invalid_reasons.append(
            "a higher rung passed after a lower rung did not pass")
    calibration_rows = sum(
        int(r.get("calibration_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    sizing_rows = sum(
        int(r.get("sizing_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    other_rows = sum(
        int(r.get("other_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    unknown_attempt_rows = sum(
        int(r.get("unknown_attempt_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    if calibration_rows:
        invalid_reasons.append(
            f"{calibration_rows} per-rung calibration request"
            f"{'s were' if calibration_rows != 1 else ' was'} mixed into "
            "the ladder")
    if sizing_rows:
        invalid_reasons.append(
            f"{sizing_rows} concurrency-sizing request"
            f"{'s were' if sizing_rows != 1 else ' was'} mixed into the ladder")
    if other_rows:
        invalid_reasons.append(
            f"{other_rows} request row{'s have' if other_rows != 1 else ' has'} "
            "an unrecognized traffic phase")
    if unknown_attempt_rows:
        invalid_reasons.append(
            f"{unknown_attempt_rows} request row"
            f"{'s have' if unknown_attempt_rows != 1 else ' has'} unknown "
            "provider-attempt timing or count")
    invalid = bool(invalid_reasons)
    return {
        "invalid": invalid,
        "invalid_reasons": invalid_reasons,
        "unverified": unverified,
        "invalid_reports": invalid_reports,
        "non_monotonic": non_monotonic,
        "calibration_rows": calibration_rows,
        "sizing_rows": sizing_rows,
        "other_rows": other_rows,
        "unknown_attempt_rows": unknown_attempt_rows,
        "good": good,
        "highest_held_rate": (None if invalid or not good else good[-1]["rate"]),
        "exit_code": 2 if invalid else 0 if good else 1,
    }


def _validated_report_context(context: object, d: Path | None = None,
                              *, expected_endpoint: str | None = None,
                              rung_count: int | None = None) -> dict:
    where = f" in {d}" if d is not None else ""
    if not isinstance(context, dict):
        raise ValueError(f"invalid sweep report context{where}")
    expected_context_fields = {
        "endpoint", "sweep_wall_s", "cooldown_s", "cooldown_events",
        "preflight",
    }
    if set(context) != expected_context_fields:
        raise ValueError(f"unknown or missing sweep report context field{where}")
    endpoint = context.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip() \
            or endpoint != sanitize_title(endpoint):
        raise ValueError(f"invalid sweep report endpoint{where}")
    if expected_endpoint is not None and endpoint != expected_endpoint:
        raise ValueError(f"sweep report endpoint disagrees with base config{where}")
    wall = context.get("sweep_wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) < 0:
        raise ValueError(f"invalid sweep report wall time{where}")
    cooldown = context.get("cooldown_s")
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) \
            or not math.isfinite(float(cooldown)) or float(cooldown) < 0:
        raise ValueError(f"invalid sweep cooldown{where}")
    events = context.get("cooldown_events")
    if isinstance(events, bool) or not isinstance(events, int) or events < 0:
        raise ValueError(f"invalid sweep cooldown event count{where}")
    preflight = context.get("preflight")
    if not isinstance(preflight, dict) \
            or not isinstance(preflight.get("skipped"), bool):
        raise ValueError(f"invalid sweep preflight evidence{where}")
    expected_preflight_fields = {
        "skipped", "attempted", "reachable", "readable",
        "reasoning_probe_requests",
    }
    if set(preflight) != expected_preflight_fields:
        raise ValueError(f"unknown or missing sweep preflight field{where}")
    counts = {}
    for field in ("attempted", "reachable", "readable",
                  "reasoning_probe_requests"):
        value = preflight.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid sweep preflight {field}{where}")
        counts[field] = value
    if counts["reachable"] > counts["attempted"] \
            or counts["readable"] > counts["reachable"]:
        raise ValueError(f"sweep preflight counts disagree{where}")
    if preflight["skipped"] and any(counts.values()):
        raise ValueError(f"skipped sweep preflight claims traffic{where}")
    if rung_count is not None:
        expected_events = 0
        if float(cooldown) > 0:
            expected_events = max(0, rung_count - 1)
            if not preflight["skipped"]:
                expected_events += 1
        if events != expected_events:
            raise ValueError(
                f"sweep cooldown accounting disagrees with attempted rungs{where}")
    return {
        "endpoint": endpoint,
        "sweep_wall_s": float(wall),
        "cooldown_s": float(cooldown),
        "cooldown_events": events,
        "preflight": {
            "skipped": preflight["skipped"],
            **counts,
        },
    }


def render_sweep_report(rungs: list[dict], report_context: dict) -> str:
    """Render the only report text a sealed sweep is allowed to contain."""
    context = _validated_report_context(report_context)
    outcome = sweep_outcome(rungs)

    def number(value, digits=0):
        return "-" if value is None else f"{value:,.{digits}f}"

    def percent(value):
        return "-" if value is None else f"{value:.1%}"

    def past_verdict(kind: str) -> str:
        return {
            "ok": "held", "caution": "cautioned", "miss": "missed",
            "invalid": "was invalid",
        }[kind]

    rows = [
        "| rate asked | achieved | held | error | TTFT p50 | TTFT p95 "
        "| E2E p50 | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rung in rungs:
        rows.append(
            f"| {rate_label(rung['rate'])} rps | "
            f"{number(rung['achieved_rps'], 1)} | "
            f"{number(rung['held'])} | {percent(rung['err'])} | "
            f"{number(rung['ttft_p50'])} | {number(rung['ttft_p95'])} | "
            f"{number(rung['e2e_p50'])} | {rung['kind'].upper()} |")

    unverified = outcome["unverified"]
    good = outcome["good"]
    if outcome["invalid"]:
        rates = ", ".join(rate_label(r["rate"]) for r in unverified)
        detail = "; ".join(outcome["invalid_reasons"])
        head = (
            f"INVALID SWEEP: {detail}. "
            + (f"Unverified rate{'s' if len(unverified) != 1 else ''}: "
               f"{rates} rps. " if unverified else "")
            + "All successful rungs are diagnostic only; this sweep makes no "
              "capacity conclusion.")
    elif good:
        best = good[-1]
        head = ("Highest rate that held: "
                f"{rate_label(best['rate'])} requests/second, "
                f"which carried about {number(best['held'])} concurrent.")

        def sentence(value: str) -> str:
            value = value.strip()
            return value if value.endswith(".") else value + "."

        nxt = next((r for r in rungs if r["rate"] > best["rate"]), None)
        if nxt:
            head += (f" The next rung, {rate_label(nxt['rate'])} rps, "
                     f"{past_verdict(nxt['kind'])}: "
                     + sentence(nxt["text"]))
        else:
            head += (" That was the top of the ladder, so the real ceiling "
                     "may be higher. Raise --rate to find it.")
    else:
        first = rungs[0]
        detail = str(first["text"]).strip()
        head = ("No rung held. The lowest rate tested "
                f"({rate_label(first['rate'])} rps) already "
                f"{past_verdict(first['kind'])}: "
                + (detail if detail.endswith(".") else detail + "."))

    report_links = [
        (f"- {rate_label(r['rate'])} rps: `{r['dir']}/report.html`"
         if r.get("source_position") is not None else
         f"- {rate_label(r['rate'])} rps: no verified report was produced")
        for r in rungs]
    preflight = context["preflight"]
    if preflight["skipped"]:
        preflight_text = "Preflight traffic: skipped; 0 requests sent."
    else:
        preflight_text = (
            f"Preflight traffic: {preflight['attempted']} representative "
            f"requests attempted ({preflight['reachable']} reached HTTP 200, "
            f"{preflight['readable']} produced readable answers), plus "
            f"{preflight['reasoning_probe_requests']} explicitly requested "
            "reasoning-control probe requests.")
    verified = [r for r in rungs if r.get("source_position") is not None]
    request_rows = sum(int(r["request_rows"]) for r in verified)
    replay_rows = sum(int(r["replay_rows"]) for r in verified)
    calibration_rows = sum(int(r["calibration_rows"]) for r in verified)
    sizing_rows = sum(int(r["sizing_rows"]) for r in verified)
    preflight_rows = sum(int(r["preflight_rows"]) for r in verified)
    probe_rows = sum(int(r["probe_rows"]) for r in verified)
    other_rows = sum(int(r["other_rows"]) for r in verified)
    unknown_attempt_rows = sum(
        int(r["unknown_attempt_rows"]) for r in verified)
    traffic_text = (
        f"Authenticated rung traffic: {request_rows} request rows "
        f"({replay_rows} replay, {calibration_rows} calibration, "
        f"{sizing_rows} sizing, {preflight_rows} preflight, "
        f"{probe_rows} probe, {other_rows} other; "
        f"{unknown_attempt_rows} rows with unknown provider-attempt "
        "timing/count).")
    if outcome["unverified"]:
        traffic_text += (
            f" {len(outcome['unverified'])} unverified rung attempt"
            f"{'s have' if len(outcome['unverified']) != 1 else ' has'} "
            "traffic that cannot be fully accounted from a sealed run.")
    cooldown_text = (
        f"Cooldown spacing: {context['cooldown_s']:g}s after preflight and "
        f"between measured rungs; {context['cooldown_events']} spacing "
        f"event{'s' if context['cooldown_events'] != 1 else ''} recorded. "
        "This sweep is sequential and stateful. Cooldown is spacing only; "
        "it proves neither QPH recovery nor provider burst or cache reset.")

    body = "\n".join([
        f"# Rate ladder: {context['endpoint']}", "", head, "",
        (f"Sweep command wall time: {context['sweep_wall_s']:.1f}s. Per-rung "
         "wall time includes setup and response drain; the configured "
         "duration is offered-load schedule time."), "", preflight_text,
        traffic_text, cooldown_text, "", "\n".join(rows), "",
        "The axis is arrival rate because that is what an open-loop generator "
        "controls. Concurrency is reported as measured, not as asked for: "
        "in-flight is arrival rate times service time, and service time rises "
        "under load, so it is an outcome rather than an input.", "",
        "Per-rung reports:", "", *report_links,
    ])
    return body + "\n"


def _atomic_text(dir_fd: int, name: str, value: str) -> dict:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"unsafe sweep artifact name: {name!r}")
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600, dir_fd=dir_fd)
    raw = value.encode("utf-8")
    try:
        _write_compare_fd(fd, raw, name)
        os.fsync(fd)
    except Exception:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    try:
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except Exception:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    _fsync_fd(dir_fd)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _claim_dir(requested: Path, artifact_id: str,
               created_at: float) -> tuple[Path, int]:
    """Claim a fresh directory; an existing path is never entered or reused."""
    requested.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10_000):
        candidate = (requested if attempt == 0 else requested.with_name(
            f"{requested.name}-{uuid.uuid4().hex[:12]}"))
        try:
            candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError:
            continue
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        dir_fd = -1
        try:
            dir_fd = os.open(candidate, flags)
            marker_fd = os.open(
                _WRITING_MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600, dir_fd=dir_fd)
            try:
                marker = strict_json_dumps({
                    "artifact_id": artifact_id,
                    "artifact_type": "sweep",
                    "status": "writing",
                    "created_at_unix": created_at,
                }).encode("utf-8") + b"\n"
                _write_compare_fd(marker_fd, marker, _WRITING_MARKER)
                os.fsync(marker_fd)
            finally:
                os.close(marker_fd)
            _fsync_fd(dir_fd)
            _fsync_directory(candidate.parent)
            return candidate, dir_fd
        except Exception:
            if dir_fd >= 0:
                os.close(dir_fd)
            raise
    raise RuntimeError(f"could not claim a unique sweep directory: {requested}")


def _verified_run_snapshot(run_dir: str | Path, position: int,
                           rate: float) -> tuple[dict, dict]:
    """Authenticate one completed run and snapshot its exact summary identity."""
    d = Path(run_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"sweep rung directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"sweep rung is not a regular directory: {d}")
    manifest = _require_run_dir(d, "summary.json")
    manifest_raw = _read_regular_bytes(d / "manifest.json")
    current_manifest = _strict_object(
        manifest_raw, "manifest.json", d / "manifest.json")
    if current_manifest != manifest:
        raise ValueError(f"input manifest changed while reading sweep rung: {d}")
    _strict_object(
        _read_regular_bytes(d / _COMPLETE_MARKER), "completion marker",
        d / _COMPLETE_MARKER)

    expected = _artifact_declarations(manifest, d)["summary.json"]
    summary_raw = _read_regular_bytes(d / "summary.json")
    summary_sha = hashlib.sha256(summary_raw).hexdigest()
    if not hmac.compare_digest(summary_sha, expected["sha256"]):
        raise ValueError(f"artifact SHA-256 mismatch for {d / 'summary.json'}")
    if len(summary_raw) != expected["bytes"]:
        raise ValueError(f"artifact byte count mismatch for {d / 'summary.json'}")
    summary = _strict_object(summary_raw, "summary.json", d / "summary.json")
    request_metadata = _artifact_declarations(manifest, d)["requests.jsonl"]
    requests_raw = _read_regular_bytes(d / "requests.jsonl")
    from .json_input import loads_strict
    phase_counts: dict[str, int] = {}
    parsed_rows = 0
    unknown_attempt_rows = 0
    for line_number, raw in enumerate(requests_raw.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n") or not raw.strip():
            raise ValueError(
                f"invalid requests.jsonl record {line_number} in {d}")
        try:
            row = loads_strict(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid requests.jsonl record {line_number} in {d}: "
                f"{exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"requests.jsonl record {line_number} is not an object in {d}")
        phase = row.get("phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError(
                f"requests.jsonl record {line_number} has no phase in {d}")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        attempts = row.get("request_attempts")
        known_attempts = (isinstance(attempts, int)
                          and not isinstance(attempts, bool)
                          and attempts >= 0)
        sent_at = row.get("first_send_unix")
        known_send_time = (isinstance(sent_at, (int, float))
                           and not isinstance(sent_at, bool)
                           and math.isfinite(float(sent_at))
                           and float(sent_at) >= 0)
        if not known_attempts or (attempts > 0 and not known_send_time):
            unknown_attempt_rows += 1
        parsed_rows += 1
    if parsed_rows != request_metadata["row_count"]:
        raise ValueError(f"strict request row count disagrees with manifest in {d}")
    replay_rows = phase_counts.get("replay", 0)
    calibration_rows = phase_counts.get("calibration", 0)
    sizing_rows = phase_counts.get("sizing", 0)
    preflight_rows = phase_counts.get("preflight", 0)
    probe_rows = phase_counts.get("probe", 0)
    other_rows = (parsed_rows - replay_rows - calibration_rows - sizing_rows
                  - preflight_rows - probe_rows)
    schedule_rows = manifest["schedule_identity"]["shard_count"]
    if replay_rows != schedule_rows:
        raise ValueError(f"replay row count disagrees with schedule identity in {d}")

    source = {
        "position": position,
        "rate_requests_per_second": float(rate),
        "artifact_id": manifest["artifact_id"],
        "logical_run_id": manifest["logical_run_id"],
        "execution_id": manifest["execution_id"],
        "workload_id": manifest["workload_id"],
        "manifest": {
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "bytes": len(manifest_raw),
        },
        "summary": {
            "sha256": summary_sha,
            "bytes": len(summary_raw),
        },
        "request_rows": parsed_rows,
        "replay_rows": replay_rows,
        "calibration_rows": calibration_rows,
        "sizing_rows": sizing_rows,
        "preflight_rows": preflight_rows,
        "probe_rows": probe_rows,
        "other_rows": other_rows,
        "unknown_attempt_rows": unknown_attempt_rows,
        "effective_config_sha256": manifest.get("effective_config_sha256"),
        "effective_config": manifest.get("effective_config"),
    }
    return summary, source


def _validate_source_shape(source: object, position: int, d: Path) -> None:
    if not isinstance(source, dict) or source.get("position") != position:
        raise ValueError(f"invalid sweep source position in {d}")
    rate = source.get("rate_requests_per_second")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
            or not math.isfinite(float(rate)) or float(rate) <= 0:
        raise ValueError(f"invalid sweep source rate in {d}")
    for field in ("artifact_id", "logical_run_id", "execution_id", "workload_id"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid sweep source {field} in {d}")
    relative = source.get("relative_path")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"invalid sweep source relative_path in {d}")
    rel_path = Path(relative)
    if rel_path.is_absolute() or ".." in rel_path.parts or rel_path == Path("."):
        raise ValueError(f"unsafe sweep source relative_path in {d}")
    for field in ("manifest", "summary"):
        metadata = source.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid sweep source {field} metadata in {d}")
        _identity_digest(metadata.get("sha256"),
                         f"sources[{position}].{field}.sha256", d)
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid sweep source {field} byte count in {d}")
    phase_fields = ("replay_rows", "calibration_rows", "sizing_rows",
                    "preflight_rows", "probe_rows", "other_rows")
    for field in ("request_rows", *phase_fields, "unknown_attempt_rows"):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid sweep source {field} in {d}")
    if source["request_rows"] != sum(source[field] for field in phase_fields):
        raise ValueError(f"sweep source request phase counts disagree in {d}")
    if source["unknown_attempt_rows"] > source["request_rows"]:
        raise ValueError(f"sweep source unknown attempt count disagrees in {d}")
    effective = source.get("effective_config")
    if not isinstance(effective, dict):
        raise ValueError(f"invalid sweep source effective_config in {d}")
    digest = _identity_digest(
        source.get("effective_config_sha256"),
        f"sources[{position}].effective_config_sha256", d)
    if canonical_sha256(effective) != digest:
        raise ValueError(f"sweep source effective config digest disagrees in {d}")


def _nested_regular_dir(root: Path, relative: str) -> Path:
    """Resolve a nested directory while refusing every symlink component."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError(f"unsafe nested sweep source path: {relative!r}")
    current = root
    root_info = current.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"sweep root is not a regular directory: {root}")
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"missing nested sweep source: {current}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                f"nested sweep source component is not a regular directory: "
                f"{current}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"nested sweep source escapes aggregate: {current}") from exc
    return current


def _capture_base_identity(base_config: dict) -> dict:
    """Pin immutable workload inputs before the first measured rung."""
    from .runner import _read_stable_bytes

    inputs = {}
    for field, key in (("profile_path", "profile"),
                       ("prompts_file", "prompts"),
                       ("timestamps_file", "timestamps")):
        path = base_config.get(field)
        if not path:
            continue
        raw, _info = _read_stable_bytes(path)
        inputs[key] = {
            "name": Path(path).name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    if len(inputs) not in {1, 2} or not ({"profile", "prompts"} & set(inputs)):
        raise ValueError("sweep base config has no identifiable workload input")
    return {"inputs": inputs}


def _validate_base_identity(identity: object, d: Path) -> dict:
    if not isinstance(identity, dict) or not isinstance(identity.get("inputs"), dict):
        raise ValueError(f"invalid sweep base identity in {d}")
    inputs = identity["inputs"]
    if len(inputs) not in {1, 2} or not ({"profile", "prompts"} & set(inputs)):
        raise ValueError(f"invalid sweep workload inputs in {d}")
    if any(key not in {"profile", "prompts", "timestamps"} for key in inputs):
        raise ValueError(f"unknown sweep workload input in {d}")
    for key, metadata in inputs.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid sweep {key} identity in {d}")
        name = metadata.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"invalid sweep {key} input name in {d}")
        _identity_digest(metadata.get("sha256"),
                         f"base_identity.inputs.{key}.sha256", d)
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid sweep {key} input size in {d}")
    return inputs


def _expected_rung_identity(base_config: dict, base_identity: dict,
                            rate: float) -> tuple[dict, str]:
    from .runner import RunConfig, _effective_config, _resolved_workload_id

    cfg = copy.deepcopy(base_config)
    cfg.update(
        qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, out_dir=f"rate_{rate_label(rate)}",
        title=(f"{base_config['title']} @ {rate_label(rate)} "
               "requests/second"))
    rc = RunConfig(**cfg)
    effective = _effective_config(rc, rc)
    workload_id = _resolved_workload_id(rc, base_identity["inputs"])
    return effective, workload_id


def _validate_source_compatibility(manifest: dict, base_config: dict,
                                   base_identity: dict, rate: float,
                                   d: Path) -> None:
    expected_config, expected_workload = _expected_rung_identity(
        base_config, base_identity, rate)
    actual_config = manifest.get("effective_config")
    if not isinstance(actual_config, dict) \
            or manifest.get("effective_config_sha256") != canonical_sha256(
                actual_config):
        raise ValueError(f"sweep rung effective config digest is invalid: {d}")
    if actual_config != expected_config:
        raise ValueError(
            f"sweep rung effective config does not match the sealed base and "
            f"rate {rate:g}: {d}")
    if manifest.get("workload_id") != expected_workload:
        raise ValueError(
            f"sweep rung workload_id does not match its sealed config: {d}")
    expected_inputs = base_identity["inputs"]
    actual_inputs = manifest.get("inputs")
    if not isinstance(actual_inputs, dict) or set(actual_inputs) != set(expected_inputs):
        raise ValueError(f"sweep rung workload inputs do not match the base: {d}")
    for key, expected in expected_inputs.items():
        actual = actual_inputs.get(key)
        if not isinstance(actual, dict) or any(
                actual.get(field) != expected[field]
                for field in ("sha256", "bytes")):
            raise ValueError(f"sweep rung {key} bytes do not match the base: {d}")
    endpoint = expected_config["endpoint"]
    for field, expected in (
            ("endpoint_base_url", endpoint.get("base_url")),
            ("endpoint_path", endpoint.get("path")),
            ("endpoint_model", endpoint.get("model"))):
        if manifest.get(field) != expected:
            raise ValueError(f"sweep rung {field} does not match the base: {d}")
    expected_mode = "prompts" if base_config.get("prompts_file") else "profile"
    if manifest.get("input_mode") != expected_mode:
        raise ValueError(f"sweep rung input mode does not match the base: {d}")
    primary = expected_inputs[expected_mode]
    if manifest.get("profile_sha256") != primary["sha256"]:
        raise ValueError(f"sweep rung primary input digest does not match: {d}")


def _validate_rung_record(record: object, source: dict, summary: dict,
                          source_path: Path, aggregate: Path) -> None:
    """Prove that a headline row is the projection of its bound summary."""
    from .metrics import _verdict

    if not isinstance(record, dict):
        raise ValueError(f"invalid sweep rung record in {aggregate}")
    position = source["position"]
    if record.get("source_position") != position:
        raise ValueError(f"sweep rung/source position mismatch in {aggregate}")
    rate = record.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
            or float(rate) != float(source["rate_requests_per_second"]):
        raise ValueError(f"sweep rung/source rate mismatch in {aggregate}")
    shown_value = record.get("dir")
    if not isinstance(shown_value, str) or not shown_value.strip():
        raise ValueError(f"invalid sweep rung report directory in {aggregate}")
    shown_dir = Path(shown_value)
    if shown_dir.is_absolute() or ".." in shown_dir.parts \
            or shown_dir == Path("."):
        raise ValueError(f"unsafe sweep rung report directory in {aggregate}")
    if shown_dir.as_posix() != source["relative_path"]:
        raise ValueError(f"sweep rung report directory mismatch in {aggregate}")
    if source_path.resolve(strict=True) != \
            (aggregate / shown_dir).resolve(strict=True):
        raise ValueError(f"sweep rung path escapes aggregate in {aggregate}")

    kind, text = _verdict(summary)
    expected = {
        "kind": kind,
        "text": text,
        "held": (summary.get("concurrency") or {}).get("in_flight_p50"),
        "achieved_rps": (summary.get("arrivals") or {}).get(
            "achieved_qps_overall"),
        "err": summary.get("error_rate"),
        "ttft_p50": (summary.get("ttft_ms") or {}).get("p50"),
        "ttft_p95": (summary.get("ttft_ms") or {}).get("p95"),
        "e2e_p50": (summary.get("e2e_ms") or {}).get("p50"),
        "request_rows": source["request_rows"],
        "replay_rows": source["replay_rows"],
        "calibration_rows": source["calibration_rows"],
        "sizing_rows": source["sizing_rows"],
        "preflight_rows": source["preflight_rows"],
        "probe_rows": source["probe_rows"],
        "other_rows": source["other_rows"],
        "unknown_attempt_rows": source["unknown_attempt_rows"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"sweep rung {position} {field} disagrees with manifest-bound "
                f"summary.json in {aggregate}")
    wall = record.get("wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) < 0:
        raise ValueError(f"invalid sweep rung wall time in {aggregate}")


class SweepArtifacts:
    """Exclusive sweep directory and its pending source-evidence chain."""

    def __init__(self, path: Path, dir_fd: int, artifact_id: str,
                 created_at: float, base_text: str, base_metadata: dict,
                 source_state: dict, base_config: dict,
                 base_identity: dict):
        self.path = path
        self._dir_fd = dir_fd
        self.artifact_id = artifact_id
        self.created_at = created_at
        self._base_text = base_text
        self._base_metadata = base_metadata
        self._source_state = source_state
        self._base_config = base_config
        self._base_identity = base_identity
        self._sources: list[tuple[Path, dict, dict]] = []
        self._source_inodes: dict[tuple[int, int], Path] = {}
        self._artifact_ids: dict[str, Path] = {}
        self._rates: set[float] = set()
        self._complete = False

    @classmethod
    def claim(cls, requested: str | Path, base_config: dict, *,
              identity_config: dict | None = None) -> "SweepArtifacts":
        from .runner import RunConfig
        import dataclasses

        if not isinstance(base_config, dict):
            raise ValueError("sweep base config must be an object")
        # Validate a private copy because RunConfig normalizes legacy fields.
        # The bytes sealed below remain exactly what the caller supplied.
        public_rc = RunConfig(**copy.deepcopy(base_config))
        identity_source = (base_config if identity_config is None
                           else identity_config)
        identity_rc = RunConfig(**copy.deepcopy(identity_source))
        public_value = dataclasses.asdict(public_rc)
        identity_value = dataclasses.asdict(identity_rc)
        for field in ("profile_path", "prompts_file", "timestamps_file"):
            public_path = public_value.pop(field)
            identity_path = identity_value.pop(field)
            if bool(public_path) != bool(identity_path) or (
                    public_path is not None
                    and Path(public_path).name != Path(identity_path).name):
                raise ValueError(
                    "sweep identity input names do not match its public config")
        if public_value != identity_value:
            raise ValueError(
                "sweep identity config may differ only by frozen input paths")
        created_at = time.time()
        artifact_id = f"sweep-{uuid.uuid4().hex}"
        # Capture Git/source identity before the output path exists, otherwise
        # a default results/ path inside the checkout makes its own run dirty.
        source_state = snapshot_source_state(Path(__file__).parent)
        base_identity = _capture_base_identity(identity_source)
        path, dir_fd = _claim_dir(Path(requested), artifact_id, created_at)
        try:
            safe_config = redact_secrets(base_config)
            base_text = strict_json_dumps(safe_config, indent=2) + "\n"
            metadata = _atomic_text(
                dir_fd, "sweep-base-config.json", base_text)
            return cls(path, dir_fd, artifact_id, created_at,
                       base_text, metadata, source_state, safe_config,
                       base_identity)
        except Exception:
            os.close(dir_fd)
            raise

    def add_rung(self, rate: float, run_dir: str | Path,
                 expected_summary: dict | None = None) -> tuple[dict, int]:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f"invalid sweep rung rate: {rate!r}")
        value = float(rate)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid sweep rung rate: {rate!r}")
        if value in self._rates:
            raise ValueError(f"duplicate sweep rung rate: {value:g}")
        d = Path(run_dir)
        summary, source = _verified_run_snapshot(d, len(self._sources), value)
        run_manifest = _strict_object(
            _read_regular_bytes(d / "manifest.json"), "manifest.json",
            d / "manifest.json")
        _validate_source_compatibility(
            run_manifest, self._base_config, self._base_identity, value, d)
        if expected_summary is not None and summary != expected_summary:
            raise ValueError(
                f"runner summary disagrees with manifest-bound summary.json: {d}")
        identity = d.stat()
        inode = (identity.st_dev, identity.st_ino)
        if inode in self._source_inodes:
            raise ValueError(
                f"duplicate sweep rung directory: {d} is the same as "
                f"{self._source_inodes[inode]}")
        artifact_id = source["artifact_id"]
        if artifact_id in self._artifact_ids:
            raise ValueError(
                f"duplicate input artifact_id {artifact_id!r}: {d} and "
                f"{self._artifact_ids[artifact_id]}")
        try:
            relative = d.resolve(strict=True).relative_to(
                self.path.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(
                f"sweep rung must be inside its aggregate directory: {d}") from exc
        if relative == Path("."):
            raise ValueError("the sweep aggregate cannot be its own rung")
        source["relative_path"] = relative.as_posix()
        safe_d = _nested_regular_dir(self.path, source["relative_path"])
        safe_identity = safe_d.stat()
        if (safe_identity.st_dev, safe_identity.st_ino) != inode:
            raise ValueError(f"sweep rung path changed while being added: {d}")
        self._source_inodes[inode] = d
        self._artifact_ids[artifact_id] = d
        self._rates.add(value)
        self._sources.append((d, source, summary))
        return summary, source["position"]

    def rung_accounting(self, position: int) -> dict:
        """Return manifest-bound phase counts for one already-added rung."""
        if isinstance(position, bool) or not isinstance(position, int) \
                or not 0 <= position < len(self._sources):
            raise ValueError(f"invalid sweep source position: {position!r}")
        source = self._sources[position][1]
        return {
            field: source[field] for field in (
                "request_rows", "replay_rows", "calibration_rows",
                "sizing_rows", "preflight_rows", "probe_rows", "other_rows",
                "unknown_attempt_rows")
        }

    def seal(self, sweep_text: str, rungs: list[dict], *,
             exit_code: int, highest_held_rate: float | None,
             report_context: dict) -> Path:
        if self._complete or self._dir_fd < 0:
            raise RuntimeError("sweep artifact is already closed")
        if _read_regular_bytes(self.path / "sweep-base-config.json") \
                != self._base_text.encode("utf-8"):
            raise ValueError("sweep base config changed before sealing")

        # Re-verify every source binding immediately before publication. This
        # detects a run that was replaced or edited after it joined the sweep.
        sources = []
        source_summaries = []
        source_paths = []
        for position, (d, expected, _summary) in enumerate(self._sources):
            d = _nested_regular_dir(self.path, expected["relative_path"])
            current_summary, current = _verified_run_snapshot(
                d, position, expected["rate_requests_per_second"])
            current_manifest = _strict_object(
                _read_regular_bytes(d / "manifest.json"), "manifest.json",
                d / "manifest.json")
            _validate_source_compatibility(
                current_manifest, self._base_config, self._base_identity,
                expected["rate_requests_per_second"], d)
            current["relative_path"] = expected["relative_path"]
            if current != expected:
                raise ValueError(
                    f"sweep rung changed before aggregate sealing: {d}")
            sources.append(current)
            source_summaries.append(current_summary)
            source_paths.append(d)

        source_positions = [r.get("source_position") for r in rungs
                            if r.get("source_position") is not None]
        if source_positions != list(range(len(sources))):
            raise ValueError(
                "sweep rung records must reference each verified source "
                "exactly once and in order")
        for r in rungs:
            position = r.get("source_position")
            if position is None:
                claimed = [r.get(key) for key in (
                    "held", "achieved_rps", "err", "ttft_p50", "ttft_p95",
                    "e2e_p50", "request_rows", "replay_rows",
                    "calibration_rows", "sizing_rows", "preflight_rows",
                    "probe_rows", "other_rows", "unknown_attempt_rows")]
                if r.get("kind") != "invalid" or any(v is not None for v in claimed):
                    raise ValueError(
                        "an unsealed sweep attempt cannot contribute a verdict "
                        "or measurement")
            else:
                if isinstance(position, bool) or not isinstance(position, int):
                    raise ValueError("sweep source positions must be integers")
                _validate_rung_record(
                    r, sources[position], source_summaries[position],
                    source_paths[position], self.path)

        outcome = sweep_outcome(rungs)
        if highest_held_rate != outcome["highest_held_rate"]:
            raise ValueError("highest held rate disagrees with manifest-bound rungs")
        if isinstance(exit_code, bool) or exit_code != outcome["exit_code"]:
            raise ValueError("sweep exit code disagrees with manifest-bound rungs")

        expected_endpoint = self._base_config["endpoint"]["path"]
        context = _validated_report_context(
            report_context, self.path, expected_endpoint=expected_endpoint,
            rung_count=len(rungs))
        if not outcome["unverified"]:
            if sum(source["preflight_rows"] for source in sources) \
                    != context["preflight"]["attempted"]:
                raise ValueError(
                    "manifest-bound preflight rows disagree with report context")
            if sum(source["probe_rows"] for source in sources) \
                    != context["preflight"]["reasoning_probe_requests"]:
                raise ValueError(
                    "manifest-bound probe rows disagree with report context")
            if any(source["preflight_rows"] or source["probe_rows"]
                   for source in sources[1:]):
                raise ValueError(
                    "preflight/probe traffic may be attached only to the first rung")
        canonical_report = render_sweep_report(rungs, context)
        if sweep_text != canonical_report:
            raise ValueError(
                "sweep.md is not the canonical report derived from rung evidence")

        sweep_metadata = _atomic_text(self._dir_fd, "sweep.md", sweep_text)
        source_state = self._source_state
        source_commit = source_state.get("git_commit")
        source_tree = source_state.get("source_tree_sha256")
        reconstructible = bool(
            source_state.get("git_dirty") is False
            and isinstance(source_commit, str) and source_commit.strip()
            and isinstance(source_tree, str) and len(source_tree) == 64)
        manifest = {
            "manifest_schema_version": 3,
            "artifact_type": "sweep",
            "artifact_id": self.artifact_id,
            "artifact_created_at_utc": datetime.fromtimestamp(
                self.created_at, timezone.utc).isoformat(),
            "artifact_created_at_unix": self.created_at,
            "operation": "rate_sweep",
            "harness_version": __version__,
            "git_commit": source_state.get("git_commit"),
            "git_dirty": source_state.get("git_dirty"),
            "source": source_state,
            "source_tree_sha256": source_state.get("source_tree_sha256"),
            "generator_source_reconstructible": reconstructible,
            "base_identity": self._base_identity,
            "report_context": context,
            "input_count": len(sources),
            "rung_count": len(rungs),
            "sources": sources,
            "rungs": redact_secrets(rungs),
            "highest_held_rate_requests_per_second": highest_held_rate,
            "exit_code": int(exit_code),
            "sweep_valid": not outcome["invalid"],
            "invalid_reasons": outcome["invalid_reasons"],
            "artifacts": {
                "sweep-base-config.json": self._base_metadata,
                "sweep.md": sweep_metadata,
            },
        }
        manifest_text = strict_json_dumps(manifest, indent=2) + "\n"
        manifest_metadata = _atomic_text(
            self._dir_fd, "manifest.json", manifest_text)
        completion_text = strict_json_dumps({
            "artifact_id": self.artifact_id,
            "artifact_type": "sweep",
            "status": "complete",
            "completed_at_unix": time.time(),
            "manifest_sha256": manifest_metadata["sha256"],
            "manifest_bytes": manifest_metadata["bytes"],
        }) + "\n"
        _atomic_text(self._dir_fd, _WRITING_MARKER, completion_text)
        os.replace(_WRITING_MARKER, _COMPLETE_MARKER,
                   src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
        _fsync_fd(self._dir_fd)
        self._complete = True
        self.close()
        _fsync_directory(self.path.parent)
        verify_sweep_output(self.path)
        return self.path

    def close(self) -> None:
        if self._dir_fd >= 0:
            os.close(self._dir_fd)
            self._dir_fd = -1


def verify_sweep_output(out_dir: str | Path) -> dict:
    """Verify a sweep aggregate's complete marker, manifest and artifacts."""
    d = Path(out_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"sweep directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"sweep directory is not a regular directory: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(f"sweep is still being written: {d}")
    for name in (_COMPLETE_MARKER, "manifest.json", "sweep.md",
                 "sweep-base-config.json"):
        _require_regular(d / name, name)
    completion_raw = _read_regular_bytes(d / _COMPLETE_MARKER)
    completion = _strict_object(
        completion_raw, "completion marker", d / _COMPLETE_MARKER)
    manifest_raw = _read_regular_bytes(d / "manifest.json")
    manifest = _strict_object(manifest_raw, "manifest.json", d / "manifest.json")
    if manifest.get("manifest_schema_version") != 3 \
            or manifest.get("artifact_type") != "sweep":
        raise ValueError(f"unsupported sweep manifest in {d}")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"invalid sweep artifact_id in {d}")
    if completion.get("status") != "complete" \
            or completion.get("artifact_type") != "sweep" \
            or completion.get("artifact_id") != artifact_id:
        raise ValueError(f"completion marker and sweep manifest disagree in {d}")
    actual_manifest = hashlib.sha256(manifest_raw).hexdigest()
    actual_bytes = len(manifest_raw)
    expected_manifest = _identity_digest(
        completion.get("manifest_sha256"),
        "completion marker manifest_sha256", d)
    if not hmac.compare_digest(actual_manifest, expected_manifest):
        raise ValueError(f"manifest SHA-256 mismatch for sweep {d}")
    declared_bytes = completion.get("manifest_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) \
            or declared_bytes != actual_bytes:
        raise ValueError(f"manifest byte count mismatch for sweep {d}")
    _verify_artifacts(
        d, manifest, ("sweep-base-config.json", "sweep.md"))
    declarations = _artifact_declarations(manifest, d)
    base_raw = _read_regular_bytes(d / "sweep-base-config.json")
    report_raw = _read_regular_bytes(d / "sweep.md")
    for name, raw in (("sweep-base-config.json", base_raw),
                      ("sweep.md", report_raw)):
        expected = declarations[name]
        if not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(), expected["sha256"]):
            raise ValueError(f"artifact SHA-256 mismatch for {d / name}")
        if len(raw) != expected["bytes"]:
            raise ValueError(f"artifact byte count mismatch for {d / name}")
    base_config = _strict_object(
        base_raw, "sweep-base-config.json", d / "sweep-base-config.json")
    if redact_secrets(base_config) != base_config:
        raise ValueError(f"sweep base config contains unredacted secrets in {d}")
    from .runner import RunConfig
    try:
        RunConfig(**copy.deepcopy(base_config))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sweep base config in {d}: {exc}") from exc
    base_identity = manifest.get("base_identity")
    _validate_base_identity(base_identity, d)
    expected_endpoint = base_config["endpoint"]["path"]
    sources = manifest.get("sources")
    if not isinstance(sources, list) \
            or manifest.get("input_count") != len(sources):
        raise ValueError(f"invalid sources in sweep manifest for {d}")
    seen = set()
    rates = set()
    source_summaries = []
    source_paths = []
    for position, source in enumerate(sources):
        _validate_source_shape(source, position, d)
        artifact = source["artifact_id"]
        rate = float(source["rate_requests_per_second"])
        if artifact in seen:
            raise ValueError(f"duplicate input artifact_id in sweep manifest for {d}")
        if rate in rates:
            raise ValueError(f"duplicate input rate in sweep manifest for {d}")
        seen.add(artifact)
        rates.add(rate)
        source_path = _nested_regular_dir(d, source["relative_path"])
        summary, current = _verified_run_snapshot(source_path, position, rate)
        source_manifest = _strict_object(
            _read_regular_bytes(source_path / "manifest.json"),
            "manifest.json", source_path / "manifest.json")
        _validate_source_compatibility(
            source_manifest, base_config, base_identity, rate, source_path)
        current["relative_path"] = source["relative_path"]
        if current != source:
            raise ValueError(
                f"manifest-bound sweep source changed or was replaced: {source_path}")
        source_summaries.append(summary)
        source_paths.append(source_path)
    rungs = manifest.get("rungs")
    if not isinstance(rungs, list) or manifest.get("rung_count") != len(rungs):
        raise ValueError(f"invalid rung records in sweep manifest for {d}")
    referenced = [r.get("source_position") for r in rungs
                  if isinstance(r, dict) and r.get("source_position") is not None]
    if referenced != list(range(len(sources))):
        raise ValueError(f"sweep rung/source references disagree in {d}")
    for record in rungs:
        if not isinstance(record, dict):
            raise ValueError(f"invalid sweep rung record in {d}")
        position = record.get("source_position")
        if position is None:
            claimed = [record.get(key) for key in (
                "held", "achieved_rps", "err", "ttft_p50", "ttft_p95",
                "e2e_p50", "request_rows", "replay_rows",
                "calibration_rows", "sizing_rows", "preflight_rows",
                "probe_rows", "other_rows", "unknown_attempt_rows")]
            if record.get("kind") != "invalid" or any(
                    value is not None for value in claimed):
                raise ValueError(
                    f"unsealed sweep attempt claims a measurement in {d}")
        else:
            if isinstance(position, bool) or not isinstance(position, int):
                raise ValueError(f"sweep source positions must be integers in {d}")
            _validate_rung_record(
                record, sources[position], source_summaries[position],
                source_paths[position], d)
    outcome = sweep_outcome(rungs)
    if manifest.get("highest_held_rate_requests_per_second") \
            != outcome["highest_held_rate"]:
        raise ValueError(f"highest held rate disagrees with sweep rungs in {d}")
    if manifest.get("exit_code") != outcome["exit_code"]:
        raise ValueError(f"exit code disagrees with sweep rungs in {d}")
    if manifest.get("sweep_valid") is not (not outcome["invalid"]):
        raise ValueError(f"sweep_valid disagrees with rung evidence in {d}")
    if manifest.get("invalid_reasons") != outcome["invalid_reasons"]:
        raise ValueError(f"invalid reasons disagree with rung evidence in {d}")
    context = _validated_report_context(
        manifest.get("report_context"), d,
        expected_endpoint=expected_endpoint, rung_count=len(rungs))
    if not outcome["unverified"]:
        if sum(source["preflight_rows"] for source in sources) \
                != context["preflight"]["attempted"]:
            raise ValueError(
                f"manifest-bound preflight rows disagree with report context in {d}")
        if sum(source["probe_rows"] for source in sources) \
                != context["preflight"]["reasoning_probe_requests"]:
            raise ValueError(
                f"manifest-bound probe rows disagree with report context in {d}")
        if any(source["preflight_rows"] or source["probe_rows"]
               for source in sources[1:]):
            raise ValueError(
                f"preflight/probe traffic is attached after the first rung in {d}")
    expected_report = render_sweep_report(rungs, context).encode("utf-8")
    if report_raw != expected_report:
        raise ValueError(
            f"sweep.md is not the canonical report derived from evidence in {d}")
    return manifest
