"""Explicit, provider-qualified reasoning-control probes and refusal UX."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time

import pytest

from traffic_replay.cli import (
    _json_object_arg,
    _print_lever_report,
    _probe_candidate_sha256,
    _probe_evidence_envelope,
    _probe_reasoning_levers,
    _validated_probe_candidates,
)


def _cap(levers, budget=512):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_lever_report(levers, budget)
    return buf.getvalue()


def test_the_answering_candidate_is_printed_ready_to_test():
    """The user can copy one line without treating one answer as proof."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"},
         "verdict": "works", "detail": "answered, finish stop, 109 tokens"},
        {"name": "enable_thinking=false", "extra": {"enable_thinking": False},
         "verdict": "unknown", "detail": "accepted, still no visible answer"},
    ])
    assert """--extra-body '{"reasoning_effort": "none"}'""" in out
    assert "ANSWERED" in out
    assert "does not prove the provider applied" in out


def test_a_rejection_keeps_the_reason_the_endpoint_gave():
    """The refusal is often the most useful line, because it names why."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"}, "verdict": "rejected",
         "detail": 'http 400: reasoning_effort="none" is not supported'},
    ])
    assert "rejected" in out
    assert "is not supported" in out


def test_when_nothing_works_it_says_so_and_names_the_next_move():
    """Silence here would leave the user with an unusable run and no idea
    what to change."""
    out = _cap([
        {"name": "reasoning_effort=minimal",
         "extra": {"reasoning_effort": "minimal"},
         "verdict": "unknown", "detail": "accepted, still no visible answer"},
    ])
    assert "none of the supplied candidates produced an answer" in out
    assert "--output-tokens" in out
    assert "wrong model for a budget this size" in out
    assert "unknown" in out
    assert "--extra-body" not in out.split("none of the supplied")[1]


def test_the_first_working_lever_wins_when_several_do():
    """Candidate order is user-controlled, so the first working one wins."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"}, "verdict": "works",
         "detail": "answered, finish stop, 109 tokens"},
        {"name": "reasoning_effort=low",
         "extra": {"reasoning_effort": "low"}, "verdict": "works",
         "detail": "answered, finish length, 512 tokens"},
    ])
    assert '{"reasoning_effort": "none"}' in out
    assert '{"reasoning_effort": "low"}' not in out.split("test this:")[1]


def test_an_errored_probe_does_not_break_the_report():
    out = _cap([{"name": "thinking.type=disabled",
                 "extra": {"thinking": {"type": "disabled"}},
                 "verdict": "error", "detail": "connection reset"}])
    assert "error" in out
    assert "none of the supplied candidates produced an answer" in out


def test_probe_argument_requires_a_finite_json_object():
    assert _json_object_arg('{"reasoning_effort":"none"}') == {
        "reasoning_effort": "none"}
    for value in ("[]", "null", '{"temperature": NaN}', "not-json",
                  '{"api_key":"sensitive-value"}',
                  '{"service_token":"opaque-value"}',
                  '{"headers":{"X-Custom-Auth":"opaque-value"}}'):
        with pytest.raises(argparse.ArgumentTypeError):
            _json_object_arg(value)


def test_probe_argument_rejection_never_echoes_secret_material():
    secret = "dapi0123456789secret-value-never-echo"
    with pytest.raises(argparse.ArgumentTypeError) as error:
        _json_object_arg('{"api_key":"' + secret + '"}')
    assert secret not in str(error.value)
    assert "api_key" not in str(error.value)


def test_probe_candidate_population_is_bounded_and_unique():
    with pytest.raises(ValueError, match="non-empty"):
        _validated_probe_candidates([{}])
    with pytest.raises(ValueError, match="at most 16"):
        _validated_probe_candidates([
            {"reasoning_effort": str(index)} for index in range(17)])
    with pytest.raises(ValueError, match="duplicate"):
        _validated_probe_candidates([
            {"reasoning_effort": "none"},
            {"reasoning_effort": "none"},
        ])
    with pytest.raises(ValueError, match="4096"):
        _validated_probe_candidates([{"reasoning_effort": "x" * 4097}])


def test_all_candidates_are_validated_before_any_probe_send(monkeypatch):
    sends = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, *_args, **_kwargs):
            sends.append(True)
            raise AssertionError("no candidate may be sent")

    monkeypatch.setattr("traffic_replay.client.EndpointClient", Client)
    cfg = {
        "endpoint": {
            "base_url": "https://example.invalid",
            "path": "/invocations",
        },
        "profile_path": "configs/profile_validation_small.json",
    }
    with pytest.raises(ValueError, match="credentials|secret-like"):
        _probe_reasoning_levers(
            cfg, 32,
            [{"reasoning_effort": "none"},
             {"api_key": "dapi0123456789never-send"}],
            representative_plans=[{
                "messages": [{"role": "user", "content": "fixture"}],
                "request_id": "representative", "intended": (8, 4, 0.0, -1),
                "chars": 7, "global_index": 0, "sample_index": 0,
                "prompt_index": None, "body_request_id": None,
                "construction": None,
            }])
    assert sends == []


def _probe_result(*, status: int, visible: bool, error: str | None = None,
                  rejected: bool | None = None,
                  error_body: bytes | None = None):
    from traffic_replay.client import RequestResult

    now = time.time()
    return RequestResult(
        request_id="placeholder", scheduled_s=0.0, dispatch_lag_ms=0.0,
        t_send_unix=now, ttfb_ms=1.0, ttft_ms=(1.0 if visible else None),
        ttfr_ms=None, ttfv_ms=(1.0 if visible else None), e2e_ms=2.0,
        status=status, ok=status == 200, error=error, content_chunks=int(visible),
        interchunk_max_ms=None, finish_reason=("stop" if visible else None),
        prompt_tokens=8, completion_tokens=(1 if visible else 0),
        cached_tokens=0, cached_tokens_source="fixture",
        intended_input_tokens=8, intended_output_tokens=4,
        intended_cache_fraction=0.0, doc_id=-1, chars_sent=7,
        stream_complete=status == 200, visible_content_seen=visible,
        max_tokens_requested=32, first_send_unix=now,
        first_attempt_unix=now, finished_unix=now + 0.002,
        connection_attempts=1, request_attempts=1,
        probe_candidate_rejected=rejected,
        http_error_body_sample_bytes=(
            len(error_body) if error_body is not None else None),
        http_error_body_sample_sha256=(
            hashlib.sha256(error_body).hexdigest()
            if error_body is not None else None),
    )


def test_probe_rows_preserve_accepted_rejected_and_unknown_evidence(
        monkeypatch):
    from traffic_replay.client import serialize_request_body

    outcomes = [
        _probe_result(status=200, visible=True),
        _probe_result(
            status=400, visible=False,
            error="http 400 (body sample bytes=51, sha256=fixture)",
            rejected=True,
            error_body=b'{"error":"invalid parameter reasoning_effort"}'),
        _probe_result(status=503, visible=False, error="unavailable"),
    ]

    class Client:
        position = 0

        def __init__(self, cfg, *_args, **_kwargs):
            self.cfg = cfg

        def send(self, messages, max_tokens, request_id, *_args,
                 probe_candidate=None, **_kwargs):
            result = outcomes[type(self).position]
            type(self).position += 1
            assert probe_candidate == candidates[type(self).position - 1]
            result.request_id = request_id
            body = serialize_request_body(
                self.cfg, messages, max_tokens, include_usage=True)
            result.physical_request_body_sha256s = [
                hashlib.sha256(body).hexdigest()]
            return result

    monkeypatch.setattr("traffic_replay.client.EndpointClient", Client)
    candidates = [
        {"reasoning_effort": "none"},
        {"reasoning_effort": "minimal"},
        {"reasoning_effort": "low"},
    ]
    cfg = {
        "endpoint": {
            "base_url": "https://example.invalid",
            "path": "/invocations",
        },
        "profile_path": "configs/profile_validation_small.json",
    }
    plan = {
        "messages": [{"role": "user", "content": "fixture"}],
        "request_id": "representative", "intended": (8, 4, 0.0, -1),
        "chars": 7, "global_index": 0, "sample_index": 0,
        "prompt_index": None, "body_request_id": None,
        "construction": None,
    }
    rows = []
    levers = _probe_reasoning_levers(
        cfg, 32, candidates, representative_plans=[plan], row_sink=rows.append)

    assert [item["disposition"] for item in levers] == [
        "accepted", "rejected", "unknown"]
    assert [item["evidence_method"] for item in levers] == [
        "single_request_behavior_observation",
        "request_validation_response",
        "non_validation_http_failure",
    ]
    assert [item["effective_status"] for item in levers] == [
        "unknown", "not_applied_request_rejected", "unknown"]
    assert all(item["effective_value"] is None for item in levers)
    assert len(rows) == 3
    for position, (candidate, lever, row) in enumerate(
            zip(candidates, levers, rows, strict=True), start=1):
        evidence = row["reasoning_control_probe"]
        assert evidence == lever["evidence"]
        assert evidence["candidate_index"] == position
        assert evidence["candidate_redacted"] == candidate
        assert evidence["candidate_canonical_sha256"] == \
            _probe_candidate_sha256(candidate)
        assert evidence["request_id"] == row["request_id"]
        assert evidence["logical_request_body_sha256"] == \
            row["request_body_sha256"]
        assert evidence["physical_request_body_sha256s"] == \
            row["physical_request_body_sha256s"]
        assert len(evidence["logical_request_body_sha256"]) == 64


@pytest.mark.parametrize(
    "status,body",
    [
        (400, b'{"error":{"message":"context length exceeded"}}'),
        (400, b'{"error":{"message":"invalid request"}}'),
        (400, b'{"error":{"message":"invalid parameter other_field; '
         b'request included reasoning_effort"}}'),
        (422, b'{"detail":"unsupported parameter reasoning_effort"}'),
    ],
    ids=("unrelated-400", "vague-400", "other-field-400", "named-422"),
)
def test_probe_validation_status_without_candidate_specific_proof_is_unknown(
        monkeypatch, status, body):
    from traffic_replay.adapters import get_endpoint_adapter
    from traffic_replay.client import serialize_request_body

    candidate = {"reasoning_effort": "none"}
    adapter = get_endpoint_adapter("openai.chat_completions.sse/v1")
    assert adapter.probe_control_rejected(status, body, candidate) is False
    result = _probe_result(
        status=status, visible=False,
        error=(f"http {status} (body sample bytes={len(body)}, "
               f"sha256={hashlib.sha256(body).hexdigest()[:16]})"),
        rejected=False, error_body=body)

    class Client:
        def __init__(self, cfg, *_args, **_kwargs):
            self.cfg = cfg

        def send(self, messages, max_tokens, request_id, *_args,
                 probe_candidate=None, **_kwargs):
            assert probe_candidate == candidate
            result.request_id = request_id
            wire = serialize_request_body(
                self.cfg, messages, max_tokens, include_usage=True)
            result.physical_request_body_sha256s = [
                hashlib.sha256(wire).hexdigest()]
            return result

    monkeypatch.setattr("traffic_replay.client.EndpointClient", Client)
    cfg = {
        "endpoint": {
            "base_url": "https://example.invalid",
            "path": "/invocations",
        },
        "profile_path": "configs/profile_validation_small.json",
    }
    plan = {
        "messages": [{"role": "user", "content": "fixture"}],
        "request_id": "representative", "intended": (8, 4, 0.0, -1),
        "chars": 7, "global_index": 0, "sample_index": 0,
        "prompt_index": None, "body_request_id": None,
        "construction": None,
    }
    rows = []
    levers = _probe_reasoning_levers(
        cfg, 32, [candidate], representative_plans=[plan],
        row_sink=rows.append)

    assert levers[0]["disposition"] == "unknown"
    assert levers[0]["effective_status"] == "unknown"
    assert levers[0]["effective_value"] is None
    assert levers[0]["evidence_method"] == \
        "candidate_rejection_unverified_http_failure"
    assert rows[0]["http_error_body_sample_bytes"] == len(body)
    assert rows[0]["http_error_body_sample_sha256"] == \
        hashlib.sha256(body).hexdigest()
    persisted = json.dumps(rows[0], sort_keys=True)
    assert body.decode() not in persisted


def test_probe_report_redacts_secret_values():
    secret = "sensitive-value-that-must-not-leak"
    out = _cap([{"name": "candidate 1 (api_key)",
                 "extra": {"api_key": secret},
                 "verdict": "works", "detail": "answered"}])
    assert secret not in out
    assert "<redacted>" in out


# ---- refusing a run we already know is void -----------------------------

class _Args:
    force = False


def test_it_refuses_and_hands_back_the_working_command():
    """Found by following our own guide as a new user. The preflight said the
    model could not answer, printed the exact flag that fixes it, then ran the
    full five minute test anyway and came back INVALID with 1,872 requests and
    zero readable answers."""
    import contextlib
    import io
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([{"name": "reasoning_effort=none",
                         "extra": {"reasoning_effort": "none"},
                         "verdict": "works", "detail": "answered"}], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "STOPPING before the load starts" in out
    assert """--extra-body '{"reasoning_effort": "none"}'""" in out
    assert "not proof that the provider applied" in out
    assert "--force" in out


def test_when_nothing_works_it_refuses_and_says_what_to_change():
    import contextlib
    import io
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([{"name": "reasoning_effort=minimal",
                         "extra": {"reasoning_effort": "minimal"},
                         "verdict": "unknown", "detail": "no answer"}], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "no supplied reasoning-control candidate helped" in out
    assert "--output-tokens" in out
    assert "fits this output budget" in out


def test_when_nothing_was_probed_refusal_does_not_claim_a_probe_failed():
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "no reasoning controls were probed" in out
    assert "--probe-extra-body" in out
    assert "no supplied reasoning-control candidate helped" not in out


def _no_answer_preflight():
    return {
        "attempted": 2,
        "reachable": 2,
        "readable": 0,
        "usage_reported": True,
        "cache_reported": True,
        "reasoning": True,
        "budgets": [40, 90],
        "budget": 90,
        "failed_probe_index": 1,
    }


def test_preflight_never_guesses_provider_controls(monkeypatch):
    from traffic_replay.cli import _check_preflight
    args = argparse.Namespace(force=False, probe_extra_body=[])

    monkeypatch.setattr("traffic_replay.cli._preflight",
                        lambda _cfg: _no_answer_preflight())

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("no control candidate was authorized")

    monkeypatch.setattr("traffic_replay.cli._probe_reasoning_levers",
                        unexpected_probe)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _check_preflight({
            "endpoint": {"base_url": "https://example.invalid",
                         "path": "/invocations"}}, args)
    assert code == 3
    assert "no provider controls were guessed" in buf.getvalue()


def test_preflight_probes_only_the_explicit_candidates(monkeypatch):
    from traffic_replay.cli import _check_preflight

    candidate = {"reasoning_effort": "none"}
    args = argparse.Namespace(force=False, probe_extra_body=[candidate])

    seen = {}
    monkeypatch.setattr("traffic_replay.cli._preflight",
                        lambda _cfg: _no_answer_preflight())

    row = {
        "phase": "probe", "request_id": "probe-01-fixture",
        "request_body_sha256": "a" * 64,
        "physical_request_body_sha256s": ["b" * 64],
        "request_attempts": 1,
    }
    evidence = _probe_evidence_envelope(
        candidate=candidate, position=1, disposition="accepted",
        evidence_method="single_request_behavior_observation", row=row)
    row["reasoning_control_probe"] = evidence

    def probe(_cfg, budget, candidates, probe_index):
        seen.update(budget=budget, candidates=candidates,
                    probe_index=probe_index)
        return [{"name": "candidate 1", "extra": candidate,
                 "verdict": "works", "detail": "answered",
                 "evidence": evidence, "_request_row": row}]

    monkeypatch.setattr("traffic_replay.cli._probe_reasoning_levers", probe)
    with contextlib.redirect_stdout(io.StringIO()):
        code = _check_preflight({
            "endpoint": {"base_url": "https://example.invalid",
                         "path": "/invocations"}}, args)
    assert code == 3
    assert seen == {"budget": 90, "candidates": [candidate],
                    "probe_index": 1}
