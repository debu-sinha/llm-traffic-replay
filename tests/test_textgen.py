"""Text materialization: identical shared prefixes (the property caching
depends on), deterministic docs, sane token targeting, calibration bounds."""
import pytest

from traffic_replay.textgen import TextMaterializer, calibrate_cpt


def test_same_doc_yields_identical_leading_text():
    m = TextMaterializer(cpt=4.0)
    a = m.prefix_text(doc_id=7, prefix_tokens=2_000, doc_len_tokens=6_000)
    b = m.prefix_text(doc_id=7, prefix_tokens=1_200, doc_len_tokens=6_000)
    assert a.startswith(b)  # shorter cut is an exact leading slice
    c = m.prefix_text(doc_id=8, prefix_tokens=1_200, doc_len_tokens=6_000)
    assert b != c  # different docs differ


def test_determinism_across_instances():
    a = TextMaterializer(cpt=4.0).prefix_text(3, 1_000, 6_000)
    b = TextMaterializer(cpt=4.0).prefix_text(3, 1_000, 6_000)
    assert a == b


def test_char_budget_tracks_cpt():
    m = TextMaterializer(cpt=4.0)
    t = m.prefix_text(5, 2_500, 6_000)
    assert abs(len(t) - 2_500 * 4.0) <= 4.0  # cut at char budget


def test_suffix_unique_per_request():
    m = TextMaterializer(cpt=4.0)
    s1 = m.suffix_text("req-a", 800)
    s2 = m.suffix_text("req-b", 800)
    assert s1 != s2
    assert "req-a" in s1 and "req-b" in s2


def test_short_suffix_never_overshoots_its_character_budget():
    m = TextMaterializer(cpt=4.0)
    for tokens in (0, 1, 2, 8, 16):
        s = m.suffix_text("request-identity", tokens)
        assert len(s) == round(tokens * 4.0)


def test_short_suffixes_do_not_all_share_a_constant_leading_marker():
    m = TextMaterializer(cpt=4.0)
    values = {m.suffix_text(f"request-{i}", 1) for i in range(20)}
    assert len(values) > 10


def test_total_message_character_target_is_exact():
    m = TextMaterializer(cpt=3.7)
    for prefix, suffix in ((0, 1), (100, 1), (100, 7), (123, 456)):
        msgs = m.messages("global-17", doc_id=(2 if prefix else -1),
                          prefix_tokens=prefix,
                          doc_len_tokens=(1_000 if prefix else 0),
                          suffix_tokens=suffix)
        rep = m.construction_report(msgs, prefix + suffix)
        assert rep["error_chars"] == 0
        assert rep["actual_chars"] == round((prefix + suffix) * 3.7)


def test_messages_structure():
    m = TextMaterializer(cpt=4.0)
    msgs = m.messages("rid1", doc_id=2, prefix_tokens=1_000,
                      doc_len_tokens=6_000, suffix_tokens=500)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    zero = m.messages("rid2", doc_id=-1, prefix_tokens=0,
                      doc_len_tokens=0, suffix_tokens=500)
    assert len(zero) == 1 and zero[0]["role"] == "user"


def test_calibration_guardrails():
    assert calibrate_cpt(4.0, 40_000, 10_000) == 4.0
    assert calibrate_cpt(4.0, 30_000, 10_000) == 3.0
    assert calibrate_cpt(4.0, 0, 10_000) == 4.0      # no data, no change
    assert calibrate_cpt(4.0, 40_000, 0) == 4.0
    assert calibrate_cpt(4.0, 1_000_000, 10) == 12.0  # clamped


@pytest.mark.parametrize("kwargs", [
    {"cpt": True}, {"cpt": "4"}, {"seed_root": True},
    {"seed_root": -1}, {"doc_cache_size": True},
])
def test_materializer_controls_are_strict(kwargs):
    with pytest.raises(ValueError):
        TextMaterializer(**kwargs)


def test_positive_prefix_requires_a_real_document():
    m = TextMaterializer()
    with pytest.raises(ValueError, match="doc_id"):
        m.prefix_text(-1, 10, 10)


@pytest.mark.parametrize("args", [
    (4.0, -1, 10), (4.0, 10, -1), (True, 10, 10), (4.0, 1.5, 10),
])
def test_calibration_inputs_are_not_coerced(args):
    with pytest.raises(ValueError):
        calibrate_cpt(*args)
