"""Prompts mode: the user replays their real prompts, not a profile.

The end-to-end test does NOT mock the loader or the endpoint. It writes a
real prompts file, runs the whole pipeline against the bundled mock, and
asserts the actual prompt text (by char length) reached the endpoint. That
is the guard against a loader that silently drops to synthetic text.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from traffic_replay.mock_server import serve
from traffic_replay.prompts import load_prompts
from traffic_replay.runner import RunConfig, run


def _write(name, text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    open(p, "w").write(text)
    return p


# ---- loader units --------------------------------------------------------

def test_load_jsonl_three_shapes():
    p = _write("p.jsonl", "\n".join([
        json.dumps({"prompt": "hello"}),
        json.dumps({"messages": [{"role": "system", "content": "be terse"},
                                 {"role": "user", "content": "hi"}]}),
        json.dumps("bare string"),
    ]) + "\n")
    got = load_prompts(p)
    assert len(got) == 3
    assert got[0] == [{"role": "user", "content": "hello"}]
    assert [m["role"] for m in got[1]] == ["system", "user"]
    assert got[2] == [{"role": "user", "content": "bare string"}]


def test_load_txt_one_per_line_skips_blanks():
    p = _write("p.txt", "first prompt\n\n  second prompt  \n")
    got = load_prompts(p)
    assert got == [[{"role": "user", "content": "first prompt"}],
                   [{"role": "user", "content": "  second prompt  "}]]


def test_load_json_array():
    p = _write("p.json", json.dumps(["a", {"text": "b"}]))
    assert load_prompts(p) == [[{"role": "user", "content": "a"}],
                               [{"role": "user", "content": "b"}]]


def test_extensions_are_case_insensitive_and_unknown_ones_fail():
    assert load_prompts(_write("p.JSON", '["a"]')) == [
        [{"role": "user", "content": "a"}]]
    assert load_prompts(_write("p.NDJSON", '"a"\n')) == [
        [{"role": "user", "content": "a"}]]
    with pytest.raises(ValueError, match="unsupported prompts extension"):
        load_prompts(_write("p.yaml", "hello"))


def test_loader_rejects_bad_inputs():
    with pytest.raises(ValueError):
        load_prompts("/no/such/file.jsonl")
    with pytest.raises(ValueError):
        load_prompts(_write("empty.jsonl", "\n\n"))
    with pytest.raises(ValueError):
        load_prompts(_write("bad.jsonl", "{not json}\n"))
    with pytest.raises(ValueError):
        load_prompts(_write("noshape.jsonl", json.dumps({"foo": "bar"}) + "\n"))
    with pytest.raises(ValueError):
        load_prompts(_write("arr.json", json.dumps({"not": "an array"})))
    with pytest.raises(ValueError, match="item 1"):
        load_prompts(_write("bad-item.json", json.dumps(["ok", {"bad": 1}])))
    with pytest.raises(ValueError, match="line 2"):
        load_prompts(_write("bad-shape.jsonl", '"ok"\n{"bad":1}\n'))
    with pytest.raises(ValueError, match="duplicate key 'prompt'"):
        load_prompts(_write(
            "duplicate.jsonl", '{"prompt":"safe","prompt":"changed"}\n'))
    with pytest.raises(ValueError, match="duplicate key 'content'"):
        load_prompts(_write(
            "duplicate.json",
            '[{"messages":[{"role":"user","content":"safe",'
            '"content":"changed"}]}]'))
    # content must be a string: null and multimodal (list of parts) fail loud
    with pytest.raises(ValueError):
        load_prompts(_write("null.jsonl", json.dumps(
            {"messages": [{"role": "user", "content": None}]}) + "\n"))
    with pytest.raises(ValueError):
        load_prompts(_write("mm.jsonl", json.dumps(
            {"messages": [{"role": "user",
                           "content": [{"type": "text", "text": "hi"}]}]}) + "\n"))


def test_inline_role_content_message_preserves_role():
    p = _write("p.jsonl", json.dumps(
        {"role": "assistant", "content": "prior turn"}) + "\n")
    assert load_prompts(p) == [[{"role": "assistant", "content": "prior turn"}]]


def test_utf8_bom_is_accepted_without_changing_prompt_text():
    p = _write("p.json", "\ufeff" + json.dumps(["café"]))
    assert load_prompts(p) == [[{"role": "user", "content": "café"}]]


def test_empty_message_role_is_rejected():
    p = _write("p.jsonl", json.dumps({
        "messages": [{"role": "  ", "content": "hello"}],
    }) + "\n")
    with pytest.raises(ValueError, match="non-empty"):
        load_prompts(p)


def test_directory_is_not_misreported_as_a_prompts_file(tmp_path):
    with pytest.raises(ValueError, match="not a readable file"):
        load_prompts(str(tmp_path))


# ---- config guards -------------------------------------------------------

def _endpoint(port):
    return {"base_url": f"http://127.0.0.1:{port}",
            "path": "/serving-endpoints/mock/invocations",
            "auth_token_env": "TRAFFIC_REPLAY_NO_TOKEN"}


def test_run_rejects_both_or_neither_source():
    with pytest.raises(ValueError):
        run(RunConfig(endpoint=_endpoint(1), profile_path="a.json",
                      prompts_file="b.jsonl", duration_s=1))
    with pytest.raises(ValueError):
        run(RunConfig(endpoint=_endpoint(1), duration_s=1))


# ---- end to end against the bundled mock (no mocking) --------------------

def test_prompts_mode_sends_the_real_text_end_to_end():
    prompts = [
        {"prompt": "Summarize the returns policy for a late delivery."},
        {"messages": [{"role": "system", "content": "You are support."},
                      {"role": "user", "content": "Reset my password?"}]},
        {"text": "Escalate this ticket and apologize to the customer."},
    ]
    pf = _write("prompts.jsonl", "\n".join(json.dumps(x) for x in prompts))
    d = tempfile.mkdtemp()

    truth = Path(d) / "truth.jsonl"
    srv = serve(0, truth)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.3)
    try:
        rc = RunConfig(
            endpoint=_endpoint(port), prompts_file=pf,
            duration_s=6, qps_base=2.0, qps_burst=4.0, qps_min=1.0,
            qps_max=6.0, max_concurrency=4, calibrate_n=2,
            out_dir=os.path.join(d, "results"),
            title="prompts mode e2e", max_output_tokens_cap=24,
            acceptance_targets={"ttft_ms": {"p50": 5000},
                                "success_rate": 0.99})
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()

    rows = [json.loads(x) for x in
            Path(out["out_dir"], "requests.jsonl").read_text().splitlines()]
    replay = [r for r in rows if r.get("phase") == "replay"]
    assert replay, "no replay requests recorded"
    assert all(r["ok"] for r in replay)

    # the real prompt text reached the endpoint: chars_sent equals the
    # content lengths of the three prompts, nothing synthetic in between
    expected = {
        len("Summarize the returns policy for a late delivery."),
        len("You are support.") + len("Reset my password?"),
        len("Escalate this ticket and apologize to the customer."),
    }
    assert {r["chars_sent"] for r in replay} <= expected
    assert len({r["chars_sent"] for r in replay}) >= 1

    report = Path(out["out_dir"], "report.md").read_text()
    assert "real prompts replayed verbatim" in report
    assert "token targeting: n/a for real prompts" in report
    # the targets came from RunConfig, not the profile, and the
    # scorecard has to say so
    assert "targets from the run config" in report
    assert "the profile" not in report.split("## Acceptance scorecard")[1][:80]
    assert out["summary"]["run"]["input_mode"] == "prompts"
    assert out["summary"]["run"]["prompts_count"] == 3
