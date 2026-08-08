"""Target evidence is captured before any inference traffic is sent."""
from __future__ import annotations

import pytest

from traffic_replay.runner import RunConfig, run


def test_endpoint_and_network_snapshots_precede_sizing_traffic(
        tmp_path, monkeypatch):
    events = []

    def fake_network(base_url):
        events.append("network")
        return {"endpoint_host": "example.invalid", "endpoint_ips": [],
                "tcp_connect_min_ms": 1.0,
                "tcp_connect_median_ms": 1.0, "samples": 1}

    def fake_metadata(base_url, path, token, timeout):
        events.append("metadata")
        return {"name": "endpoint-at-start"}

    def stop_at_sizing(*args, **kwargs):
        events.append("sizing")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(
        "traffic_replay.netpath.measure_network_path", fake_network)
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata", fake_metadata)
    monkeypatch.setattr(
        "traffic_replay.runner._size_for_concurrency", stop_at_sizing)

    rc = RunConfig(
        profile_path="configs/profile_validation_small.json",
        endpoint={"base_url": "https://example.invalid",
                  "path": "/serving-endpoints/example/invocations",
                  "auth_token_env": "UNSET_TEST_TOKEN"},
        sizing_concurrency=1, duration_s=1, out_dir=str(tmp_path / "runs"))
    with pytest.raises(RuntimeError, match="ordering assertion"):
        run(rc, quiet=True)
    assert events == ["network", "metadata", "sizing"]
