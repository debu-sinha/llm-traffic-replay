"""Pool must construct the intended cache structure: right-sized documents,
popularity skew, and constructed fractions near the sampled targets."""
import numpy as np

from traffic_replay import profile as prof
from traffic_replay.prefix_pool import PrefixPool

SPEC = prof.Profile(
    name="t", provenance="test",
    input_tokens={"p50": 10_000, "p95": 24_000},
    output_tokens={"p50": 40, "p95": 90},
    cache_fraction={"p50": 0.60, "p95": 0.87},
)


def test_constructed_fraction_tracks_targets():
    d = prof.sample(SPEC, 8_000, seed=9)
    pool = PrefixPool(seed=13)
    a = pool.assign(d["prefix_tokens"])
    rep = pool.structure_report(a, d["input_tokens"])
    # Construction can undershoot slightly when a document is shorter than
    # the wanted prefix (top-bucket cap), never overshoot wildly.
    assert 0.50 <= rep["constructed_fraction_p50"] <= 0.65
    assert 0.80 <= rep["constructed_fraction_p95"] <= 0.92


def test_popularity_skew_exists():
    d = prof.sample(SPEC, 8_000, seed=9)
    pool = PrefixPool(seed=13)
    a = pool.assign(d["prefix_tokens"])
    rep = pool.structure_report(a, d["input_tokens"])
    # Zipf skew: the hottest doc should carry well above uniform share,
    # and plenty of distinct docs should still get used.
    assert rep["hottest_doc_share"] > 0.03
    assert rep["distinct_docs_used"] > 30


def test_prefix_never_exceeds_want_or_doc():
    d = prof.sample(SPEC, 3_000, seed=9)
    pool = PrefixPool(seed=13)
    a = pool.assign(d["prefix_tokens"])
    assert (a.prefix_tokens <= d["prefix_tokens"]).all()
    for i in range(len(a.doc_id)):
        if a.doc_id[i] >= 0:
            assert a.prefix_tokens[i] <= pool.doc_len[int(a.doc_id[i])]


def test_large_prefix_is_not_silently_clipped_to_40k():
    pool = PrefixPool(seed=13)
    wants = np.array([40_001, 99_999, 199_999])
    a = pool.assign(wants)
    assert np.array_equal(a.prefix_tokens, wants)
    assert all(pool.doc_len[int(doc)] >= want
               for doc, want in zip(a.doc_id, wants))


def test_out_of_range_prefix_is_rejected_not_misreported():
    pool = PrefixPool(seed=13)
    try:
        pool.assign(np.array([200_001]))
    except ValueError as exc:
        assert "outside pool range" in str(exc)
    else:
        raise AssertionError("out-of-range prefix was silently clipped")


def test_zero_prefix_handled():
    pool = PrefixPool(seed=13)
    a = pool.assign(np.array([0, 5_000, 0]))
    assert a.doc_id[0] == -1 and a.prefix_tokens[0] == 0
    assert a.doc_id[2] == -1 and a.prefix_tokens[2] == 0
    assert a.prefix_tokens[1] > 0
