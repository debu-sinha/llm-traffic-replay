"""External verification receipts for immutable run artifacts.

A run cannot externally verify the manifest that contains its own summary. This
module verifies a completed v3 run from the outside, re-derives the canonical
decision with an explicit integrity context, and writes a separate sealed
receipt.  The source run is never opened for writing.

The receipt proves internal SHA-256 byte consistency only.  It is deliberately
not described as a digital signature: it does not prove authorship, trusted
time, repository availability, or that the files cannot be changed later.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import stat
import struct
import time
import uuid

from . import __version__
from .aggregate import (
    _COMPLETE_MARKER,
    _MAX_REQUEST_JSONL_LINE_BYTES,
    _MAX_REQUEST_JOURNAL_BYTES,
    _WRITING_MARKER,
    _artifact_declarations,
    _fsync_directory,
    _fsync_fd,
    _has_path,
    _identity_digest,
    _load_json_object,
    _measure_regular,
    _read_regular_bytes,
    _regular_identity,
    _require_regular,
    _require_run_dir,
    _verify_artifacts,
    _verify_run_completion_marker,
)
from .artifacts import snapshot_source_state, strict_json_dumps
from .json_input import json_error_detail, loads_strict
from .report_decision import IntegrityContext, build_report_decision


RECEIPT_SCHEMA_VERSION = 1
_CANONICAL_RUN_ARTIFACTS = (
    "requests.jsonl",
    "summary.json",
    "report.md",
    "report.html",
    "start.json",
)
_COMMIT_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_ASSURANCE = (
    "SHA-256 hashes establish internal byte consistency only. This receipt "
    "is not a digital signature, does not prove authorship or trusted time, "
    "and does not prevent later mutation."
)
_VERIFICATION_SCOPE = {
    "manifest_schema_version": 3,
    "required_canonical_artifacts": list(_CANONICAL_RUN_ARTIFACTS),
    "all_manifest_declared_artifacts_hash_checked": True,
    "strict_json_objects": [
        "start.json", "summary.json", "manifest.json",
        ".traffic-replay-complete",
    ],
    "strict_jsonl_objects": ["requests.jsonl"],
    "summary_request_cross_checks": [
        "replay count versus schedule identity",
        "replay global-index digest, bounds, uniqueness, and shard partition",
        "replay scheduled-time digest and bounds",
        "replay total/ok/failed",
        "judged/acceptable outcomes",
        "preflight gate counts and acceptable outcomes",
        "all-phase HTTP 429 count, status coverage, denominator, and phases",
    ],
    "source_bindings_reread_before_receipt_seal": True,
    "sealed_receipt_artifacts": [
        "verification.json", "verified-report.md", "verified-report.html",
    ],
}


def _nonzero_digest(value: object, pattern: re.Pattern) -> bool:
    return (isinstance(value, str) and bool(pattern.fullmatch(value))
            and any(char != "0" for char in value.lower()))


def _metadata(raw: bytes) -> dict:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _strict_bound_object(d: Path, manifest: dict, name: str) -> tuple[dict, dict]:
    expected = _artifact_declarations(manifest, d)[name]
    raw = _read_regular_bytes(d / name)
    actual = _metadata(raw)
    if not hmac.compare_digest(actual["sha256"], expected["sha256"]):
        raise ValueError(f"artifact SHA-256 mismatch for {d / name}")
    if actual["bytes"] != expected["bytes"]:
        raise ValueError(f"artifact byte count mismatch for {d / name}")
    try:
        value = loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"invalid {name} in {d / name}: {json_error_detail(exc)}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object: {d / name}")
    return value, actual


def _strict_bound_requests(d: Path, manifest: dict) -> tuple[dict, dict]:
    """Strictly parse and hash the manifest-bound journal in one read."""
    name = "requests.jsonl"
    expected = _artifact_declarations(manifest, d)[name]
    path = d / name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    rows = 0
    phases: dict[str, int] = {}
    replay_rows = 0
    replay_ok = 0
    replay_failed = 0
    status_observed = 0
    http_429 = 0
    http_429_phases: dict[str, int] = {}
    answer_rows_judged = 0
    acceptable_outcomes = 0
    preflight_rows_judged = 0
    preflight_acceptable_outcomes = 0
    preflight_http_200 = 0
    reasoning_control_probes: list[dict | None] = []
    setup_request_links: list[dict] = []
    replay_identity_rows: list[tuple[int, float, str]] = []
    replay_request_ids: set[str] = set()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        if before.st_size != expected["bytes"] \
                or before.st_size > _MAX_REQUEST_JOURNAL_BYTES:
            raise ValueError(f"artifact byte count is invalid for {path}")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            line_number = 0
            while True:
                raw = handle.readline(_MAX_REQUEST_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > _MAX_REQUEST_JSONL_LINE_BYTES:
                    raise ValueError(
                        f"requests.jsonl line {line_number} exceeds the "
                        f"{_MAX_REQUEST_JSONL_LINE_BYTES:,}-byte limit in {d}")
                digest.update(raw)
                size += len(raw)
                if size > expected["bytes"]:
                    raise ValueError(
                        f"requests.jsonl exceeds its declared byte count in "
                        f"{d}")
                if not raw.endswith(b"\n") or not raw.strip():
                    raise ValueError(
                        f"invalid requests.jsonl record {line_number} in {d}")
                try:
                    row = loads_strict(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {line_number}: "
                        f"{json_error_detail(exc)}") from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"requests.jsonl line {line_number} is not an object "
                        f"in {d}")
                rows += 1
                phase = row.get("phase")
                if not isinstance(phase, str) or not phase:
                    raise ValueError(
                        f"requests.jsonl line {line_number} has no valid "
                        f"phase in {d}")
                phases[phase] = phases.get(phase, 0) + 1
                if phase in {"preflight", "probe"}:
                    from .runner import _prepare_prior_request_rows
                    try:
                        prepared_setup = _prepare_prior_request_rows([row])[0]
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"requests.jsonl line {line_number} has invalid "
                            f"setup request binding in {d}: {exc}") from exc
                    if prepared_setup != row:
                        raise ValueError(
                            f"requests.jsonl line {line_number} setup "
                            f"request evidence is not canonical in {d}")
                    setup_request_links.append({
                        "phase": phase,
                        "trace_request_id": row.get("request_id"),
                        "setup_request_binding_sha256": row.get(
                            "setup_request_binding_sha256"),
                        "logical_request_body_sha256": row.get(
                            "request_body_sha256"),
                        "physical_request_body_sha256s": list(
                            row.get("physical_request_body_sha256s") or []),
                    })
                envelope_present = "reasoning_control_probe" in row
                if phase == "probe":
                    envelope = row.get("reasoning_control_probe")
                    if envelope_present:
                        # Validate the versioned schema and request/logical/
                        # physical body links before trusting the gate copy.
                        from .runner import \
                            _validated_reasoning_control_probe_evidence
                        envelope = _validated_reasoning_control_probe_evidence(
                            envelope, row=row,
                            label=("requests.jsonl line "
                                   f"{line_number}.reasoning_control_probe"))
                    reasoning_control_probes.append(envelope)
                elif envelope_present:
                    raise ValueError(
                        f"requests.jsonl line {line_number} carries probe "
                        f"evidence outside a probe row in {d}")
                status = row.get("status")
                if status is not None and (
                        isinstance(status, bool)
                        or not isinstance(status, int)
                        or not 100 <= status <= 599):
                    raise ValueError(
                        f"requests.jsonl line {line_number} has an invalid "
                        f"HTTP status in {d}")
                if status is not None:
                    status_observed += 1
                if status == 429:
                    http_429 += 1
                    http_429_phases[phase] = http_429_phases.get(phase, 0) + 1
                # v0.7 timing schema: TTSE is the first complete framed event
                # emitted by the selected adapter parser.  It must not precede
                # the raw response-body clock, and content milestones must not
                # precede it.  Older sealed artifacts omit TTSE and remain
                # verifiable under their legacy schema.
                for timing_field in ("ttse_ms", "caller_ttse_ms"):
                    timing_value = row.get(timing_field)
                    if timing_value is not None and (
                            isinstance(timing_value, bool)
                            or not isinstance(timing_value, (int, float))
                            or not math.isfinite(float(timing_value))
                            or float(timing_value) < 0):
                        raise ValueError(
                            f"requests.jsonl line {line_number} has invalid "
                            f"{timing_field} in {d}")
                if row.get("ttse_ms") is not None:
                    ttse_value = float(row["ttse_ms"])
                    ttfb = row.get("ttfb_ms")
                    if ttfb is None or isinstance(ttfb, bool) \
                            or not isinstance(ttfb, (int, float)) \
                            or not math.isfinite(float(ttfb)) \
                            or float(ttfb) < 0 \
                            or ttse_value < float(ttfb):
                        raise ValueError(
                            f"requests.jsonl line {line_number} has TTSE "
                            f"before or without a valid TTFB in {d}")
                    for milestone_field in (
                            "ttft_ms", "ttfr_ms", "ttfv_ms",
                            "ttf_tool_call_ms"):
                        milestone = row.get(milestone_field)
                        if milestone is not None and (
                                isinstance(milestone, bool)
                                or not isinstance(milestone, (int, float))
                                or not math.isfinite(float(milestone))
                                or float(milestone) < ttse_value):
                            raise ValueError(
                                f"requests.jsonl line {line_number} has "
                                f"{milestone_field} before TTSE in {d}")
                if row.get("caller_ttse_ms") is not None:
                    caller_ttse_value = float(row["caller_ttse_ms"])
                    caller_ttfb = row.get("caller_ttfb_ms")
                    if caller_ttfb is None \
                            or isinstance(caller_ttfb, bool) \
                            or not isinstance(caller_ttfb, (int, float)) \
                            or not math.isfinite(float(caller_ttfb)) \
                            or float(caller_ttfb) < 0 \
                            or caller_ttse_value < float(caller_ttfb):
                        raise ValueError(
                            f"requests.jsonl line {line_number} has caller "
                            f"TTSE before or without a valid caller TTFB in "
                            f"{d}")
                    for caller_milestone_field in (
                            "caller_ttft_ms", "caller_ttfr_ms",
                            "caller_ttfv_ms", "caller_ttf_tool_call_ms"):
                        caller_milestone = row.get(caller_milestone_field)
                        if caller_milestone is not None and (
                                isinstance(caller_milestone, bool)
                                or not isinstance(
                                    caller_milestone, (int, float))
                                or not math.isfinite(float(caller_milestone))
                                or float(caller_milestone)
                                < caller_ttse_value):
                            raise ValueError(
                                f"requests.jsonl line {line_number} has "
                                f"{caller_milestone_field} before caller "
                                f"TTSE in {d}")
                if phase == "preflight":
                    if status == 200:
                        preflight_http_200 += 1
                    observed = any(field in row for field in (
                        "visible_content_seen", "reasoning_seen",
                        "valid_tool_calls", "refusal_seen"))
                    if not observed and row.get("ok") is False:
                        # Legacy transport failures remain judgeable failures.
                        preflight_rows_judged += 1
                    elif observed:
                        visible = row.get("visible_content_seen", False)
                        stream_complete = row.get("stream_complete", False)
                        valid_tool_calls = row.get("valid_tool_calls", 0)
                        parse_errors = row.get("parse_errors", 0)
                        refusal_seen = row.get("refusal_seen", False)
                        if not isinstance(visible, bool) \
                                or not isinstance(stream_complete, bool) \
                                or isinstance(valid_tool_calls, bool) \
                                or not isinstance(valid_tool_calls, int) \
                                or valid_tool_calls < 0 \
                                or isinstance(parse_errors, bool) \
                                or not isinstance(parse_errors, int) \
                                or parse_errors < 0 \
                                or not isinstance(refusal_seen, bool):
                            raise ValueError(
                                "requests.jsonl line "
                                f"{line_number} has invalid preflight "
                                f"answer-outcome fields in {d}")
                        preflight_rows_judged += 1
                        if status == 200 \
                                and (visible or valid_tool_calls > 0) \
                                and not refusal_seen and stream_complete \
                                and parse_errors == 0:
                            preflight_acceptable_outcomes += 1
                if phase != "replay":
                    continue
                replay_rows += 1
                request_id = row.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError(
                        f"requests.jsonl line {line_number} has no valid "
                        f"replay request_id in {d}")
                if request_id in replay_request_ids:
                    raise ValueError(
                        f"duplicate replay request_id {request_id!r} in {d}")
                replay_request_ids.add(request_id)
                global_index = row.get("global_index")
                if isinstance(global_index, bool) \
                        or not isinstance(global_index, int) \
                        or global_index < 0:
                    raise ValueError(
                        f"requests.jsonl line {line_number} has no valid "
                        f"replay global_index in {d}")
                scheduled_s = row.get("scheduled_s")
                if isinstance(scheduled_s, bool) \
                        or not isinstance(scheduled_s, (int, float)) \
                        or not math.isfinite(float(scheduled_s)):
                    raise ValueError(
                        f"requests.jsonl line {line_number} has no valid "
                        f"replay scheduled_s in {d}")
                replay_identity_rows.append(
                    (global_index, float(scheduled_s), request_id))
                ok = row.get("ok")
                if not isinstance(ok, bool):
                    raise ValueError(
                        f"requests.jsonl line {line_number} has a non-boolean "
                        f"replay ok field in {d}")
                if ok:
                    replay_ok += 1
                else:
                    replay_failed += 1

                observed = any(field in row for field in (
                    "visible_content_seen", "reasoning_seen",
                    "valid_tool_calls", "refusal_seen"))
                # Current failure rows carry these fields too. Legacy failures
                # remain judgeable as failures; a legacy success does not.
                if not observed and not ok:
                    answer_rows_judged += 1
                    continue
                if not observed:
                    continue
                visible = row.get("visible_content_seen", False)
                stream_complete = row.get("stream_complete", False)
                valid_tool_calls = row.get("valid_tool_calls", 0)
                parse_errors = row.get("parse_errors", 0)
                refusal_seen = row.get("refusal_seen", False)
                if not isinstance(visible, bool) \
                        or not isinstance(stream_complete, bool) \
                        or isinstance(valid_tool_calls, bool) \
                        or not isinstance(valid_tool_calls, int) \
                        or valid_tool_calls < 0 \
                        or isinstance(parse_errors, bool) \
                        or not isinstance(parse_errors, int) \
                        or parse_errors < 0 \
                        or not isinstance(refusal_seen, bool):
                    raise ValueError(
                        f"requests.jsonl line {line_number} has invalid "
                        f"answer-outcome fields in {d}")
                answer_rows_judged += 1
                if (visible or valid_tool_calls > 0) \
                        and not refusal_seen \
                        and stream_complete and parse_errors == 0:
                    acceptable_outcomes += 1
            after = os.fstat(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    if _regular_identity(before) != _regular_identity(after):
        raise ValueError(f"requests.jsonl changed while reading in {d}")
    actual = {"sha256": digest.hexdigest(), "bytes": size, "row_count": rows}
    if not hmac.compare_digest(actual["sha256"], expected["sha256"]):
        raise ValueError(f"artifact SHA-256 mismatch for {path}")
    if actual["bytes"] != expected["bytes"]:
        raise ValueError(f"artifact byte count mismatch for {path}")
    if actual["row_count"] != expected.get("row_count"):
        raise ValueError(f"artifact row count mismatch for {path}")

    ordered = sorted(replay_identity_rows)
    ordered_indices = [row[0] for row in ordered]
    if len(set(ordered_indices)) != len(ordered_indices):
        raise ValueError(f"duplicate replay global_index in {d}")
    index_identity = manifest["index_identity"]
    shard_index = index_identity["shard_index"]
    shard_total = index_identity["shard_total"]
    misplaced = [value for value in ordered_indices
                 if value % shard_total != shard_index]
    if misplaced:
        raise ValueError(
            f"replay global_index {misplaced[0]} disagrees with shard "
            f"partition in {d}")
    index_digest = hashlib.sha256(b"".join(
        struct.pack("<q", value) for value in ordered_indices)).hexdigest()
    if not hmac.compare_digest(
            index_digest,
            index_identity["global_indices_sha256"].lower()):
        raise ValueError(
            f"index_identity SHA-256 disagrees with requests.jsonl for {d}")
    index_min = min(ordered_indices) if ordered_indices else None
    index_max = max(ordered_indices) if ordered_indices else None
    if index_identity["count"] != len(ordered_indices) \
            or index_identity.get("min") != index_min \
            or index_identity.get("max") != index_max:
        raise ValueError(
            f"index_identity count/min/max disagrees with requests.jsonl "
            f"for {d}")

    ordered_timestamps = [row[1] for row in ordered]
    schedule_digest = hashlib.sha256(b"".join(
        struct.pack("<d", value) for value in ordered_timestamps)).hexdigest()
    schedule_identity = manifest["schedule_identity"]
    if not hmac.compare_digest(
            schedule_digest,
            schedule_identity["shard_timestamps_sha256"].lower()):
        raise ValueError(
            f"schedule_identity SHA-256 disagrees with requests.jsonl for {d}")
    schedule_min = min(ordered_timestamps) if ordered_timestamps else None
    schedule_max = max(ordered_timestamps) if ordered_timestamps else None
    if schedule_identity["shard_count"] != len(ordered_timestamps) \
            or schedule_identity.get("shard_min_s") != schedule_min \
            or schedule_identity.get("shard_max_s") != schedule_max:
        raise ValueError(
            f"schedule_identity count/min/max disagrees with requests.jsonl "
            f"for {d}")
    if shard_total == 1 and (
            not hmac.compare_digest(
                schedule_digest,
                schedule_identity["global_timestamps_sha256"].lower())
            or schedule_identity["global_count"] != len(ordered_timestamps)
            or schedule_identity.get("global_min_s") != schedule_min
            or schedule_identity.get("global_max_s") != schedule_max):
        raise ValueError(
            f"global schedule_identity disagrees with unsharded "
            f"requests.jsonl for {d}")
    scheduled_requests = manifest["schedule"].get("requests")
    if isinstance(scheduled_requests, bool) \
            or not isinstance(scheduled_requests, int) \
            or scheduled_requests != len(ordered_timestamps):
        raise ValueError(
            f"schedule.requests disagrees with requests.jsonl for {d}")
    evidence = {
        "request_rows": rows,
        "phases": {name: phases[name] for name in sorted(phases)},
        "replay_rows": replay_rows,
        "replay_ok": replay_ok,
        "replay_failed": replay_failed,
        "http_status_observed_for": status_observed,
        "http_429_count": http_429,
        "http_429_phases": {
            name: http_429_phases[name] for name in sorted(http_429_phases)
        },
        "answer_rows_judged": answer_rows_judged,
        "acceptable_outcomes": acceptable_outcomes,
        "preflight_rows_judged": preflight_rows_judged,
        "preflight_acceptable_outcomes": preflight_acceptable_outcomes,
        "preflight_http_200": preflight_http_200,
        "reasoning_control_probes": reasoning_control_probes,
        "setup_request_links": setup_request_links,
        "replay_global_indices_sha256": index_digest,
        "replay_schedule_sha256": schedule_digest,
    }
    return actual, evidence


def _exact_nonnegative_int(value: object, label: str, d: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {label} in manifest-bound summary for {d}")
    return value


def _validate_summary_request_consistency(
        d: Path, manifest: dict, summary: dict, evidence: dict) -> dict:
    """Fail closed when the summary contradicts its manifest-bound journal."""
    expected_replay = manifest["schedule_identity"]["shard_count"]
    comparisons = (
        ("requests_total", summary.get("requests_total"),
         evidence["replay_rows"]),
        ("requests_ok", summary.get("requests_ok"), evidence["replay_ok"]),
        ("requests_failed", summary.get("requests_failed"),
         evidence["replay_failed"]),
    )
    if evidence["replay_rows"] != expected_replay:
        raise ValueError(
            f"manifest-bound replay row count disagrees with schedule identity "
            f"for {d}: {evidence['replay_rows']} != {expected_replay}")
    for label, claimed, observed in comparisons:
        claimed_count = _exact_nonnegative_int(claimed, label, d)
        if claimed_count != observed:
            raise ValueError(
                f"manifest-bound summary {label} disagrees with requests.jsonl "
                f"for {d}: {claimed_count} != {observed}")

    http = summary.get("http_429")
    if not isinstance(http, dict):
        raise ValueError(
            f"manifest-bound summary is missing HTTP-429 evidence for {d}")
    http_comparisons = (
        ("http_429_count", summary.get("http_429_count"),
         evidence["http_429_count"]),
        ("http_429.count", http.get("count"), evidence["http_429_count"]),
        ("http_429.request_rows_examined", http.get("request_rows_examined"),
         evidence["request_rows"]),
        ("http_429.http_status_observed_for",
         http.get("http_status_observed_for"),
         evidence["http_status_observed_for"]),
    )
    for label, claimed, observed in http_comparisons:
        claimed_count = _exact_nonnegative_int(claimed, label, d)
        if claimed_count != observed:
            raise ValueError(
                f"manifest-bound summary {label} disagrees with requests.jsonl "
                f"for {d}: {claimed_count} != {observed}")
    if http.get("phases") != evidence["http_429_phases"]:
        raise ValueError(
            f"manifest-bound HTTP-429 phase counts disagree with "
            f"requests.jsonl for {d}")

    answers = summary.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(
            f"manifest-bound summary is missing answer evidence for {d}")
    for label, claimed, observed in (
            ("answers.judged", answers.get("judged"),
             evidence["answer_rows_judged"]),
            ("answers.acceptable_outcomes", answers.get(
                "acceptable_outcomes"), evidence["acceptable_outcomes"])):
        claimed_count = _exact_nonnegative_int(claimed, label, d)
        if claimed_count != observed:
            raise ValueError(
                f"manifest-bound summary {label} disagrees with requests.jsonl "
                f"for {d}: {claimed_count} != {observed}")
    return evidence


def _validate_preflight_gate_consistency(
        d: Path, summary: dict, start: dict, evidence: dict) -> None:
    """Bind every durable preflight label to its rows and answer facts."""
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    gate = run.get("preflight_gate")
    start_gate = start.get("preflight_gate")
    setup_kind = run.get("artifact_kind") == "command_setup_traffic"
    if gate is None:
        if start_gate is not None or setup_kind:
            raise ValueError(f"missing preflight gate evidence in {d}")
        return
    required = {
        "skipped", "attempted", "reachable", "readable",
        "reasoning_probe_requests", "outcome", "force_requested",
        "gate_satisfied", "evidence_mode", "binding", "binding_sha256",
    }
    optional = {"reasoning_control_probes"}
    if not isinstance(gate, dict) \
            or not required.issubset(gate) \
            or set(gate) - required - optional \
            or start_gate != gate:
        raise ValueError(f"invalid or inconsistent preflight gate in {d}")
    if gate.get("skipped") is not False \
            or not isinstance(gate.get("force_requested"), bool) \
            or not isinstance(gate.get("gate_satisfied"), bool):
        raise ValueError(f"invalid preflight gate flags in {d}")
    mode = gate.get("evidence_mode")
    if mode not in {"carried_setup_rows", "inherited_setup_artifact"}:
        raise ValueError(f"invalid preflight evidence mode in {d}")
    from .runner import (
        _PREFLIGHT_BINDING_SCHEMA, _binding_sha256, _is_sha256,
        _validated_setup_artifact_reference,
    )
    binding = gate.get("binding")
    if not isinstance(binding, dict) \
            or binding.get("schema_version") != _PREFLIGHT_BINDING_SCHEMA \
            or not _is_sha256(gate.get("binding_sha256")) \
            or _binding_sha256(binding) != gate.get("binding_sha256") \
            or not isinstance(binding.get("execution"), dict) \
            or not isinstance(binding.get("setup_requests"), list):
        raise ValueError(f"invalid cryptographic preflight binding in {d}")
    counts = {}
    for field in ("attempted", "reachable", "readable",
                  "reasoning_probe_requests"):
        item = gate.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"invalid preflight gate {field} in {d}")
        counts[field] = item
    if counts["attempted"] <= 0 \
            or counts["reachable"] > counts["attempted"] \
            or counts["readable"] > counts["reachable"]:
        raise ValueError(f"preflight gate counts disagree in {d}")
    carried = mode == "carried_setup_rows"
    if carried:
        if evidence["phases"].get("preflight", 0) != counts["attempted"] \
                or evidence["phases"].get("probe", 0) != \
                counts["reasoning_probe_requests"] \
                or evidence["preflight_rows_judged"] != counts["attempted"] \
                or evidence["preflight_http_200"] != counts["reachable"] \
                or evidence["preflight_acceptable_outcomes"] != \
                counts["readable"]:
            raise ValueError(
                f"preflight gate counts disagree with requests.jsonl in {d}")
        if binding["setup_requests"] != evidence.get(
                "setup_request_links"):
            raise ValueError(
                f"preflight binding setup links disagree with the journal "
                f"in {d}")
    elif evidence["phases"].get("preflight", 0) \
            or evidence["phases"].get("probe", 0) \
            or evidence.get("setup_request_links"):
        raise ValueError(
            f"inherited preflight evidence duplicates setup rows in {d}")
    row_probe_evidence = evidence.get("reasoning_control_probes")
    if not isinstance(row_probe_evidence, list) \
            or (carried and len(row_probe_evidence) !=
                counts["reasoning_probe_requests"]) \
            or (not carried and row_probe_evidence):
        raise ValueError(
            f"reasoning-control probe row count disagrees in {d}")
    if "reasoning_control_probes" in gate:
        supplied = gate.get("reasoning_control_probes")
        if not isinstance(supplied, list) \
                or len(supplied) != counts["reasoning_probe_requests"] \
                or (carried and supplied != row_probe_evidence) \
                or any(item is None for item in supplied):
            raise ValueError(
                f"reasoning-control probe gate evidence disagrees in {d}")
        from .runner import _validated_reasoning_control_probe_evidence
        validated = [
            _validated_reasoning_control_probe_evidence(
                item,
                label=f"preflight_gate.reasoning_control_probes[{index}]")
            for index, item in enumerate(supplied)
        ]
        if validated != supplied \
                or [item["candidate_index"] for item in validated] != \
                list(range(1, len(validated) + 1)):
            raise ValueError(
                f"reasoning-control probe gate ordering is invalid in {d}")
    elif carried and any(item is not None for item in row_probe_evidence):
        raise ValueError(
            f"preflight gate omits manifest-bound probe evidence in {d}")
    complete = (
        counts["reachable"] == counts["attempted"]
        and counts["readable"] == counts["attempted"])
    outcome = gate.get("outcome")
    valid = (
        (outcome == "preflight_passed" and complete
         and gate["gate_satisfied"] is True)
        or (outcome == "preflight_forced_unreadable"
            and gate["force_requested"] is True
            and gate["gate_satisfied"] is False
            and counts["reachable"] == counts["attempted"]
            and counts["readable"] < counts["attempted"])
        or (setup_kind and outcome == "preflight_refused"
            and gate["gate_satisfied"] is False and not complete)
        or (setup_kind and outcome == "preflight_forced_failed"
            and gate["force_requested"] is True
            and gate["gate_satisfied"] is False
            and counts["reachable"] < counts["attempted"])
        or (setup_kind and outcome == "preflight_state_unknown"
            and gate["gate_satisfied"] is False)
    )
    if not valid:
        raise ValueError(f"preflight gate outcome disagrees with evidence in {d}")

    if setup_kind:
        setup = summary.get("setup_traffic")
        if not isinstance(setup, dict) \
                or setup.get("artifact_kind") != "command_setup_traffic" \
                or setup.get("outcome") != outcome \
                or setup.get("preflight_gate") != gate \
                or setup.get("request_rows") != evidence["request_rows"] \
                or setup.get("performance_result") is not False \
                or setup.get("sla_result") is not False \
                or setup.get("capacity_result") is not False \
                or run.get("setup_outcome") != outcome \
                or start.get("setup_outcome") != outcome:
            raise ValueError(f"setup traffic outcome disagrees in {d}")
        exit_code = setup.get("exit_code")
        if outcome == "preflight_refused":
            if isinstance(exit_code, bool) or not isinstance(exit_code, int) \
                    or exit_code <= 0:
                raise ValueError(f"invalid refused preflight exit code in {d}")
        elif exit_code is not None:
            raise ValueError(f"continued preflight claims an exit code in {d}")
    elif outcome not in {
            "preflight_passed", "preflight_forced_unreadable"}:
        raise ValueError(
            f"measured run carries a non-executable preflight state in {d}")
    if not setup_kind:
        reference = run.get("setup_artifact_reference")
        if start.get("setup_artifact_reference") != reference:
            raise ValueError(
                f"setup artifact reference disagrees between start and "
                f"summary in {d}")
        _validated_setup_artifact_reference(reference, gate)


def _remeasure_nonstructured_artifacts(d: Path, manifest: dict) -> None:
    """Second-pass every bound artifact not already read as strict JSON."""
    declarations = _artifact_declarations(manifest, d)
    structured = {"summary.json", "start.json", "requests.jsonl"}
    for name, expected in declarations.items():
        if name in structured:
            continue
        actual, actual_bytes, actual_rows = _measure_regular(d / name)
        if not hmac.compare_digest(actual, expected["sha256"]):
            raise ValueError(f"artifact SHA-256 mismatch for {d / name}")
        if actual_bytes != expected["bytes"]:
            raise ValueError(f"artifact byte count mismatch for {d / name}")
        if "row_count" in expected and actual_rows != expected["row_count"]:
            raise ValueError(f"artifact row count mismatch for {d / name}")


def _source_reconstructibility(manifest: dict, start: dict) -> dict:
    """Assess whether the source identity is clean, complete, and consistent."""
    reasons: list[str] = []
    details: list[str] = []
    commit = manifest.get("git_commit")
    tree = manifest.get("source_tree_sha256")
    dirty = manifest.get("git_dirty")
    if dirty is not False:
        reasons.append("GIT_STATE_DIRTY_OR_UNKNOWN")
        details.append("manifest git_dirty is not false")
    if not _nonzero_digest(commit, _COMMIT_RE):
        reasons.append("GIT_COMMIT_DIGEST_INVALID")
        details.append("manifest git_commit is not a 40- or 64-hex digest")
    if not _nonzero_digest(tree, _SHA256_RE):
        reasons.append("SOURCE_TREE_DIGEST_INVALID")
        details.append("manifest source_tree_sha256 is not a SHA-256 digest")

    for label, source in (
            ("manifest.source", manifest.get("source")),
            ("start.source", start.get("source"))):
        if not isinstance(source, dict):
            reasons.append("SOURCE_IDENTITY_MISSING")
            details.append(f"{label} is missing")
            continue
        for field, expected in (
                ("git_commit", commit),
                ("git_dirty", dirty),
                ("source_tree_sha256", tree)):
            observed = source.get(field)
            if type(observed) is not type(expected) or observed != expected:
                reasons.append("SOURCE_IDENTITY_INCONSISTENT")
                details.append(f"{label}.{field} disagrees with the manifest")

    reason_codes = list(dict.fromkeys(reasons))
    return {
        "reconstructible": not reason_codes,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": tree,
        "reason_codes": reason_codes,
        "reason": (
            "A clean Git commit and SHA-256 source-tree identity are recorded "
            "consistently in start.json and manifest.json; repository and "
            "commit availability were not checked."
            if not reason_codes else "; ".join(details)
        ),
        "boundary": (
            "This checks recorded identity and digest shape/consistency; it "
            "does not fetch the repository or prove commit availability."
        ),
    }


def _generator_reconstructibility(source: dict) -> dict:
    reasons: list[str] = []
    details: list[str] = []
    dirty = source.get("git_dirty") if isinstance(source, dict) else None
    commit = source.get("git_commit") if isinstance(source, dict) else None
    tree = (source.get("source_tree_sha256")
            if isinstance(source, dict) else None)
    if dirty is not False:
        reasons.append("VERIFIER_GIT_STATE_DIRTY_OR_UNKNOWN")
        details.append("verifier git_dirty is not false")
    if not _nonzero_digest(commit, _COMMIT_RE):
        reasons.append("VERIFIER_GIT_COMMIT_DIGEST_INVALID")
        details.append("verifier git_commit is not a valid Git object digest")
    if not _nonzero_digest(tree, _SHA256_RE):
        reasons.append("VERIFIER_SOURCE_TREE_DIGEST_INVALID")
        details.append("verifier source-tree digest is not valid SHA-256")
    return {
        "reconstructible": not reasons,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": tree,
        "reason_codes": reasons,
        "reason": (
            "The external verifier ran from a clean recorded Git commit and "
            "SHA-256 source-tree identity; repository and commit availability "
            "were not checked."
            if not reasons else "; ".join(details)
        ),
        "boundary": (
            "This checks recorded verifier identity; it does not fetch the "
            "repository or prove commit availability."
        ),
    }


def _gate_capacity_on_source(decision: dict, reconstructibility: dict) -> dict:
    """Never issue the positive held conclusion for unreconstructible source."""
    result = deepcopy(decision)
    capacity = result.get("endpoint_capacity")
    if reconstructibility["reconstructible"] \
            or not isinstance(capacity, dict) \
            or capacity.get("code") != "HELD_AT_TESTED_LOAD":
        return result
    result["endpoint_capacity"] = {
        **capacity,
        "code": "INCONCLUSIVE",
        "label": "Endpoint capacity inconclusive",
        "severity": "warning",
        "reason": (
            "The artifact bytes are internally consistent, but the source "
            "identity is dirty, incomplete, or inconsistent. Tested-load "
            "facts remain observations; no held-capacity conclusion is issued."
        ),
        "reason_codes": list(dict.fromkeys([
            *reconstructibility["reason_codes"],
            "SOURCE_NOT_RECONSTRUCTIBLE",
        ])),
        "endpoint_ceiling_established": False,
        "provider_headroom_established": False,
    }
    return result


def _gate_capacity_on_generator(decision: dict,
                                reconstructibility: dict) -> dict:
    result = deepcopy(decision)
    capacity = result.get("endpoint_capacity")
    if reconstructibility["reconstructible"] \
            or not isinstance(capacity, dict) \
            or capacity.get("code") != "HELD_AT_TESTED_LOAD":
        return result
    result["endpoint_capacity"] = {
        **capacity,
        "code": "INCONCLUSIVE",
        "label": "Endpoint capacity inconclusive",
        "severity": "warning",
        "reason": (
            "The source run is internally consistent, but the external "
            "verifier did not run from a clean, reconstructible source "
            "identity. No held-capacity conclusion is issued."
        ),
        "reason_codes": list(dict.fromkeys([
            *reconstructibility["reason_codes"],
            "VERIFIER_SOURCE_NOT_RECONSTRUCTIBLE",
        ])),
        "endpoint_ceiling_established": False,
        "provider_headroom_established": False,
    }
    return result


def _source_binding(d: Path, manifest: dict, summary_meta: dict,
                    start_meta: dict, requests_meta: dict,
                    request_evidence: dict) -> dict:
    manifest_raw = _read_regular_bytes(d / "manifest.json")
    completion_raw = _read_regular_bytes(d / _COMPLETE_MARKER)
    declarations = _artifact_declarations(manifest, d)
    artifacts = {
        name: dict(declarations[name]) for name in sorted(declarations)
    }
    # Use the values observed during the strict reads, not merely copied
    # declarations, for the structured artifacts that drive the decision.
    artifacts["summary.json"] = summary_meta
    artifacts["start.json"] = start_meta
    artifacts["requests.jsonl"] = requests_meta
    return {
        "artifact_id": manifest["artifact_id"],
        "logical_run_id": manifest["logical_run_id"],
        "execution_id": manifest["execution_id"],
        "workload_id": manifest["workload_id"],
        "manifest": _metadata(manifest_raw),
        "summary": summary_meta,
        "start": start_meta,
        "completion": _metadata(completion_raw),
        "artifacts": artifacts,
        "request_evidence": request_evidence,
    }


def verify_run_output(run_dir: str | Path) -> dict:
    """Verify one immutable v3 run and re-derive its decision in memory.

    No source file is opened for writing.  Every artifact declared by the
    manifest is checked, and all five canonical artifacts are mandatory.
    Structured evidence is then read again through no-follow file descriptors
    so malformed or concurrently replaced JSON cannot drive the receipt.
    """
    d = Path(run_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"run directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"run directory is not a regular directory: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(f"run is still being written: {d}")
    for name in (*_CANONICAL_RUN_ARTIFACTS, "manifest.json",
                 _COMPLETE_MARKER):
        _require_regular(d / name, name)

    # First pass: validate the v3 identity, the completion chain, and every
    # manifest declaration.  Requiring the canonical names prevents a partial
    # manifest from binding only the two files needed by merge/compare.
    manifest = _require_run_dir(d, "summary.json")
    declarations = _artifact_declarations(manifest, d)
    missing = [name for name in _CANONICAL_RUN_ARTIFACTS
               if name not in declarations]
    if missing:
        raise ValueError(
            f"manifest for {d} is missing canonical v3 artifact integrity "
            f"entries: {', '.join(missing)}")

    # Second pass: strict semantic reads for the JSON evidence and journal.
    # These reads also detect a replacement after the first integrity pass.
    summary, summary_meta = _strict_bound_object(d, manifest, "summary.json")
    start, start_meta = _strict_bound_object(d, manifest, "start.json")
    requests_meta, request_evidence = _strict_bound_requests(d, manifest)
    _validate_summary_request_consistency(
        d, manifest, summary, request_evidence)
    _validate_preflight_gate_consistency(
        d, summary, start, request_evidence)

    manifest_raw = _read_regular_bytes(d / "manifest.json")
    try:
        current_manifest = loads_strict(manifest_raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"invalid manifest.json in {d}: {json_error_detail(exc)}") from exc
    if current_manifest != manifest:
        raise ValueError(f"manifest changed while verifying run: {d}")
    completion = _load_json_object(d / _COMPLETE_MARKER, "completion marker")
    _verify_run_completion_marker(d, current_manifest)

    # Re-measure every non-JSON declaration after the structured reads. Each
    # bound file is therefore read twice overall without scanning a large
    # request journal a wasteful third time.
    _remeasure_nonstructured_artifacts(d, current_manifest)
    if _load_json_object(d / "manifest.json", "manifest.json") != manifest:
        raise ValueError(f"manifest changed during final run verification: {d}")
    if _load_json_object(d / _COMPLETE_MARKER, "completion marker") != completion:
        raise ValueError(
            f"completion marker changed during final run verification: {d}")

    reconstructibility = _source_reconstructibility(manifest, start)
    decision = build_report_decision(
        summary,
        IntegrityContext(
            "verified",
            "All canonical v3 artifacts and the completion/manifest chain "
            "matched their internal SHA-256 bindings. This is not a digital "
            "signature or authorship proof.",
        ),
    )
    decision = _gate_capacity_on_source(decision, reconstructibility)
    return {
        "manifest": manifest,
        "summary": summary,
        "start": start,
        "binding": _source_binding(
            d, manifest, summary_meta, start_meta, requests_meta,
            request_evidence),
        "source_reconstructibility": reconstructibility,
        "decision": decision,
    }


def _ensure_external_output(source: Path, requested: Path) -> None:
    source_resolved = source.resolve(strict=True)
    requested_resolved = requested.resolve(strict=False)
    try:
        requested_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"verification receipt must be outside the immutable source run: "
            f"{requested}")
    if requested_resolved.parent != source_resolved.parent:
        raise ValueError(
            "verification receipt must be a sibling of the immutable source "
            f"run: {requested}")


def _write_all(fd: int, raw: bytes, name: str) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"short write while creating {name}")
        view = view[written:]


def _claim_receipt_dir(requested: Path, receipt_id: str,
                       created_at: float) -> tuple[Path, int]:
    requested.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10_000):
        candidate = requested if attempt == 0 else requested.with_name(
            f"{requested.name}-{uuid.uuid4().hex[:12]}")
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
                0o600,
                dir_fd=dir_fd,
            )
            try:
                marker = strict_json_dumps({
                    "artifact_id": receipt_id,
                    "artifact_type": "run_verification_receipt",
                    "status": "writing",
                    "created_at_unix": created_at,
                }).encode("utf-8") + b"\n"
                _write_all(marker_fd, marker, _WRITING_MARKER)
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
    raise RuntimeError(f"could not claim a unique receipt directory: {requested}")


def _atomic_text(dir_fd: int, name: str, value: str) -> dict:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"unsafe receipt artifact name: {name!r}")
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600, dir_fd=dir_fd)
    raw = value.encode("utf-8")
    try:
        _write_all(fd, raw, name)
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
    return _metadata(raw)


def _same_source(first: dict, second: dict) -> bool:
    return (
        first["binding"] == second["binding"]
        and first["source_reconstructibility"]
        == second["source_reconstructibility"]
        and first["decision"] == second["decision"]
    )


def _verified_report_context(receipt: dict) -> dict:
    def reproducibility(value: dict) -> dict:
        reason_codes = list(value.get("reason_codes") or [])
        return {
            "code": "PASS" if value.get("reconstructible") else "FAILED",
            "reason": value["reason"],
            "reason_codes": reason_codes,
        }

    return {
        "view_label": receipt["view_label"],
        "receipt_id": receipt["receipt_id"],
        "source_artifact_id": receipt["source_run"]["artifact_id"],
        "source_manifest_sha256": receipt["source_run"]["manifest"][
            "sha256"],
        "verifier_version": receipt["verifier_version"],
        "verified_at_utc": receipt["created_at_utc"],
        "assurance": receipt["assurance"],
        "decision": receipt["decision"],
        "source_reproducibility": reproducibility(
            receipt["source_reconstructibility"]),
        "verifier_reproducibility": reproducibility(
            receipt["verifier_source_reconstructibility"]),
    }


def _verified_report_title(summary: dict, source: Path) -> str:
    run = summary.get("run")
    title = run.get("title") if isinstance(run, dict) else None
    return str(title) if title not in (None, "") else source.name


def create_run_verification_receipt(
        run_dir: str | Path, out_dir: str | Path) -> Path:
    """Verify ``run_dir`` and create one unique, separately sealed receipt."""
    source = Path(run_dir)
    requested = Path(out_dir)
    _ensure_external_output(source, requested)

    first = verify_run_output(source)
    generator_source = snapshot_source_state(Path(__file__).parent)
    generator_reconstructibility = _generator_reconstructibility(
        generator_source)
    receipt_decision = _gate_capacity_on_generator(
        first["decision"], generator_reconstructibility)
    created_at = time.time()
    receipt_id = f"run-verification-{uuid.uuid4().hex}"
    out, dir_fd = _claim_receipt_dir(requested, receipt_id, created_at)
    try:
        source_locator = {
            "kind": "sibling_directory",
            "directory_name": source.resolve(strict=True).name,
        }
        receipt = {
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "artifact_type": "run_verification_receipt",
            "receipt_id": receipt_id,
            "created_at_utc": datetime.fromtimestamp(
                created_at, timezone.utc).isoformat(),
            "created_at_unix": created_at,
            "verified": True,
            "view_label": "EXTERNAL VERIFIED VIEW",
            "verifier_version": __version__,
            "report_title": _verified_report_title(first["summary"], source),
            "verification_code": "INTERNAL_HASH_CONSISTENCY_VERIFIED",
            "verification_scope": _VERIFICATION_SCOPE,
            "assurance": _ASSURANCE,
            "digital_signature": False,
            "source_locator": source_locator,
            "source_run": first["binding"],
            "source_reconstructibility": first[
                "source_reconstructibility"],
            "verifier_source_reconstructibility":
                generator_reconstructibility,
            "decision": receipt_decision,
        }
        verification_metadata = _atomic_text(
            dir_fd,
            "verification.json",
            strict_json_dumps(receipt, indent=2) + "\n",
        )

        # The source is read again after the receipt payload exists but before
        # its manifest is sealed.  Any observed change aborts completion.
        final = verify_run_output(source)
        if not _same_source(first, final):
            raise ValueError(
                f"source run changed while creating verification receipt: "
                f"{source}")

        from .metrics import render_html, render_markdown
        report_context = _verified_report_context(receipt)
        report_title = receipt["report_title"]
        markdown_metadata = _atomic_text(
            dir_fd,
            "verified-report.md",
            render_markdown(
                final["summary"], report_title,
                verification_context=report_context),
        )
        html_metadata = _atomic_text(
            dir_fd,
            "verified-report.html",
            render_html(
                final["summary"], report_title,
                verification_context=report_context),
        )

        # Rendering can be non-trivial for a large report. Re-open the source
        # after both derivative views have been written so a mutation during
        # rendering cannot be hidden behind the earlier verification pass.
        # The receipt is promoted only when all three observations agree.
        post_render = verify_run_output(source)
        if not _same_source(first, post_render):
            raise ValueError(
                f"source run changed while rendering verification receipt: "
                f"{source}")

        manifest = {
            "manifest_schema_version": 3,
            "artifact_type": "run_verification_receipt",
            "artifact_id": receipt_id,
            "artifact_created_at_utc": datetime.fromtimestamp(
                created_at, timezone.utc).isoformat(),
            "artifact_created_at_unix": created_at,
            "operation": "verify_run",
            "harness_version": __version__,
            "git_commit": generator_source.get("git_commit"),
            "git_dirty": generator_source.get("git_dirty"),
            "source_tree_sha256": generator_source.get("source_tree_sha256"),
            "source": generator_source,
            "assurance": _ASSURANCE,
            "digital_signature": False,
            "source_locator": source_locator,
            "source_run": post_render["binding"],
            "source_reconstructible": post_render[
                "source_reconstructibility"]["reconstructible"],
            "verifier_source_reconstructible":
                generator_reconstructibility["reconstructible"],
            "capacity_conclusion": receipt_decision["endpoint_capacity"],
            "artifacts": {
                "verification.json": verification_metadata,
                "verified-report.md": markdown_metadata,
                "verified-report.html": html_metadata,
            },
        }
        manifest_metadata = _atomic_text(
            dir_fd,
            "manifest.json",
            strict_json_dumps(manifest, indent=2) + "\n",
        )
        completion = {
            "artifact_id": receipt_id,
            "artifact_type": "run_verification_receipt",
            "status": "complete",
            "completed_at_unix": time.time(),
            "manifest_sha256": manifest_metadata["sha256"],
            "manifest_bytes": manifest_metadata["bytes"],
        }
        _atomic_text(
            dir_fd,
            _WRITING_MARKER,
            strict_json_dumps(completion) + "\n",
        )
        os.replace(
            _WRITING_MARKER,
            _COMPLETE_MARKER,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
        )
        _fsync_fd(dir_fd)
    finally:
        os.close(dir_fd)
    _fsync_directory(out.parent)
    # Verify the receipt's own completion chain. The source was already read
    # twice around receipt construction; callers can later use the default
    # verify_run_receipt behavior to compare the receipt with current bytes.
    verify_run_receipt(out, verify_source=False)
    return out


def _validate_receipt_binding_shape(binding: object, d: Path) -> dict:
    if not isinstance(binding, dict):
        raise ValueError(f"invalid source binding in receipt {d}")
    for field in ("artifact_id", "logical_run_id", "execution_id",
                  "workload_id"):
        value = binding.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid source binding {field} in receipt {d}")
    for field in ("manifest", "summary", "start", "completion"):
        value = binding.get(field)
        if not isinstance(value, dict):
            raise ValueError(
                f"invalid source binding {field} metadata in receipt {d}")
        _identity_digest(value.get("sha256"), f"source.{field}.sha256", d)
        size = value.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"invalid source binding {field} byte count in receipt {d}")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"invalid source artifacts in receipt {d}")
    for name in _CANONICAL_RUN_ARTIFACTS:
        value = artifacts.get(name)
        if not isinstance(value, dict):
            raise ValueError(
                f"receipt source binding is missing {name} in {d}")
        _identity_digest(
            value.get("sha256"), f"source.artifacts.{name}.sha256", d)
        size = value.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"invalid source artifact {name} byte count in receipt {d}")
    request_evidence = binding.get("request_evidence")
    if not isinstance(request_evidence, dict):
        raise ValueError(f"invalid source request evidence in receipt {d}")
    for field in (
            "request_rows", "replay_rows", "replay_ok", "replay_failed",
            "http_status_observed_for", "http_429_count",
            "answer_rows_judged", "acceptable_outcomes",
            "preflight_rows_judged", "preflight_acceptable_outcomes",
            "preflight_http_200"):
        value = request_evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"invalid source request evidence {field} in receipt {d}")
    for field in (
            "replay_global_indices_sha256", "replay_schedule_sha256"):
        _identity_digest(
            request_evidence.get(field),
            f"source.request_evidence.{field}", d)
    if request_evidence["replay_rows"] != (
            request_evidence["replay_ok"]
            + request_evidence["replay_failed"]):
        raise ValueError(f"source replay counts disagree in receipt {d}")
    if request_evidence["http_status_observed_for"] > request_evidence[
            "request_rows"] \
            or request_evidence["http_429_count"] > request_evidence[
                "http_status_observed_for"] \
            or request_evidence["acceptable_outcomes"] > request_evidence[
                "answer_rows_judged"] \
            or request_evidence["preflight_acceptable_outcomes"] > \
            request_evidence["preflight_rows_judged"] \
            or request_evidence["preflight_acceptable_outcomes"] > \
            request_evidence["preflight_http_200"] \
            or request_evidence["preflight_http_200"] > \
            request_evidence["preflight_rows_judged"]:
        raise ValueError(f"source request evidence counts disagree in receipt {d}")
    for field in ("phases", "http_429_phases"):
        counts = request_evidence.get(field)
        if not isinstance(counts, dict) or any(
                not isinstance(name, str) or not name
                or isinstance(count, bool) or not isinstance(count, int)
                or count < 0
                for name, count in counts.items()):
            raise ValueError(
                f"invalid source request evidence {field} in receipt {d}")
    if sum(request_evidence["phases"].values()) != request_evidence[
            "request_rows"] \
            or sum(request_evidence["http_429_phases"].values()) != \
            request_evidence["http_429_count"]:
        raise ValueError(f"source request phase counts disagree in receipt {d}")
    return binding


def verify_run_receipt(receipt_dir: str | Path, *,
                       source_run: str | Path | None = None,
                       verify_source: bool = True) -> dict:
    """Verify a receipt seal and, by default, its current source binding.

    ``verify_source=False`` is used immediately after creation and by the CLI
    only to re-open the just-verified receipt through strict no-follow reads.
    Long-lived consumers should retain the default so later source mutation is
    detected.
    """
    d = Path(receipt_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"verification receipt directory not found: {d}") \
            from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            f"verification receipt is not a regular directory: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(f"verification receipt is still being written: {d}")
    for name in (
            _COMPLETE_MARKER, "manifest.json", "verification.json",
            "verified-report.md", "verified-report.html"):
        _require_regular(d / name, name)

    completion = _load_json_object(d / _COMPLETE_MARKER, "completion marker")
    manifest = _load_json_object(d / "manifest.json", "manifest.json")
    if manifest.get("manifest_schema_version") != 3 \
            or manifest.get("artifact_type") != "run_verification_receipt":
        raise ValueError(f"unsupported verification receipt manifest in {d}")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"invalid verification receipt artifact_id in {d}")
    if completion.get("status") != "complete" \
            or completion.get("artifact_type") != \
            "run_verification_receipt" \
            or completion.get("artifact_id") != artifact_id:
        raise ValueError(
            f"completion marker and verification receipt manifest disagree "
            f"in {d}")
    manifest_sha, manifest_bytes, _rows = _measure_regular(
        d / "manifest.json")
    expected_sha = _identity_digest(
        completion.get("manifest_sha256"),
        "completion marker manifest_sha256",
        d,
    )
    if not hmac.compare_digest(manifest_sha, expected_sha):
        raise ValueError(f"manifest SHA-256 mismatch for receipt {d}")
    declared_bytes = completion.get("manifest_bytes")
    if isinstance(declared_bytes, bool) \
            or not isinstance(declared_bytes, int) \
            or declared_bytes != manifest_bytes:
        raise ValueError(f"manifest byte count mismatch for receipt {d}")
    _verify_artifacts(
        d, manifest,
        ("verification.json", "verified-report.md", "verified-report.html"))
    verification = _load_json_object(
        d / "verification.json", "verification.json")

    if verification.get("receipt_schema_version") != RECEIPT_SCHEMA_VERSION \
            or verification.get("artifact_type") != \
            "run_verification_receipt" \
            or verification.get("receipt_id") != artifact_id \
            or verification.get("verified") is not True \
            or verification.get("view_label") != "EXTERNAL VERIFIED VIEW" \
            or verification.get("verifier_version") != \
            manifest.get("harness_version") \
            or not isinstance(verification.get("report_title"), str) \
            or not verification.get("report_title").strip() \
            or verification.get("verification_scope") != _VERIFICATION_SCOPE \
            or verification.get("digital_signature") is not False \
            or verification.get("assurance") != _ASSURANCE:
        raise ValueError(f"invalid verification receipt payload in {d}")
    binding = _validate_receipt_binding_shape(
        verification.get("source_run"), d)
    source_locator = verification.get("source_locator")
    if not isinstance(source_locator, dict) \
            or source_locator.get("kind") != "sibling_directory":
        raise ValueError(f"invalid source locator in receipt {d}")
    directory_name = source_locator.get("directory_name")
    if not isinstance(directory_name, str) or not directory_name \
            or Path(directory_name).name != directory_name \
            or directory_name in {".", ".."}:
        raise ValueError(f"unsafe source locator in receipt {d}")
    if manifest.get("source_run") != binding \
            or manifest.get("source_locator") != source_locator \
            or manifest.get("assurance") != _ASSURANCE \
            or manifest.get("digital_signature") is not False:
        raise ValueError(
            f"verification payload and receipt manifest disagree in {d}")

    reconstructibility = verification.get("source_reconstructibility")
    verifier_reconstructibility = verification.get(
        "verifier_source_reconstructibility")
    decision = verification.get("decision")
    if not isinstance(reconstructibility, dict) \
            or not isinstance(reconstructibility.get("reconstructible"), bool):
        raise ValueError(f"invalid source reconstructibility in receipt {d}")
    if not isinstance(decision, dict) \
            or not isinstance(decision.get("endpoint_capacity"), dict):
        raise ValueError(f"invalid decision in receipt {d}")
    if not isinstance(verifier_reconstructibility, dict) \
            or not isinstance(
                verifier_reconstructibility.get("reconstructible"), bool):
        raise ValueError(
            f"invalid verifier source reconstructibility in receipt {d}")
    verifier_source = manifest.get("source")
    if not isinstance(verifier_source, dict):
        raise ValueError(f"missing recorded verifier source in receipt {d}")
    for field in ("git_commit", "git_dirty", "source_tree_sha256"):
        if manifest.get(field) != verifier_source.get(field):
            raise ValueError(
                f"receipt manifest {field} disagrees with recorded verifier "
                f"source in {d}")
    if _generator_reconstructibility(verifier_source) != \
            verifier_reconstructibility:
        raise ValueError(
            f"verifier source reconstructibility disagrees with recorded "
            f"verifier source in {d}")
    if manifest.get("source_reconstructible") is not reconstructibility[
            "reconstructible"]:
        raise ValueError(
            f"source reconstructibility disagrees with manifest {d}")
    if manifest.get("capacity_conclusion") != decision[
            "endpoint_capacity"]:
        raise ValueError(f"capacity conclusion disagrees with receipt {d}")
    if manifest.get("verifier_source_reconstructible") is not \
            verifier_reconstructibility["reconstructible"]:
        raise ValueError(
            f"verifier source reconstructibility disagrees with manifest {d}")
    if not verify_source:
        return verification

    source_path = (Path(source_run) if source_run is not None
                   else d.parent / directory_name)
    current = verify_run_output(source_path)
    current_binding = current["binding"]
    if current_binding != binding:
        raise ValueError(
            f"source run no longer matches verification receipt {d}")
    if current["source_reconstructibility"] != verification.get(
            "source_reconstructibility"):
        raise ValueError(
            f"source reconstructibility disagrees with receipt {d}")
    expected_decision = _gate_capacity_on_generator(
        current["decision"], verifier_reconstructibility)
    if expected_decision != verification.get("decision"):
        raise ValueError(f"re-derived decision disagrees with receipt {d}")
    from .metrics import render_html, render_markdown
    report_context = _verified_report_context(verification)
    report_title = verification["report_title"]
    expected_views = {
        "verified-report.md": render_markdown(
            current["summary"], report_title,
            verification_context=report_context).encode("utf-8"),
        "verified-report.html": render_html(
            current["summary"], report_title,
            verification_context=report_context).encode("utf-8"),
    }
    for name, expected in expected_views.items():
        if not hmac.compare_digest(
                hashlib.sha256(_read_regular_bytes(d / name)).digest(),
                hashlib.sha256(expected).digest()):
            raise ValueError(
                f"{name} is not the canonical external verified view in {d}")
    return verification


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "create_run_verification_receipt",
    "verify_run_output",
    "verify_run_receipt",
]
