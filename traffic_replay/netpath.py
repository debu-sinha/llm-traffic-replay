"""Where the client sits relative to the endpoint, measured not assumed.

Every latency figure this harness reports contains at least one network
round trip: the request travels out and the first token travels back. Run
the generator in the wrong region and that round trip is silently added to
TTFT, to end-to-end, and to any SLA judgment made from them.

This was not hypothetical. A load test that produced TTFT p50 842 ms against
a 500 ms target was generated from a US east coast machine against an
endpoint in us-west-2, and 82 ms of that number was the width of the
country. The tool reported the latency and said nothing about the geography,
so the only reason it came to light was somebody asking.

The round trip is measured directly, as the minimum TCP connect time over a
few tries. Minimum rather than mean because a round trip has a hard floor
set by distance and speed of light, and everything above that floor is
queueing noise. Nothing here reaches a third party: no geolocation service,
no public-IP lookup. The endpoint's own address is resolved and connected to,
which is what the run is about to do a few thousand times anyway.

Stdlib only.
"""

from __future__ import annotations

import socket
import time
import urllib.parse


def measure_network_path(
    base_url: str, samples: int = 5, timeout: float = 5.0
) -> dict | None:
    """Resolve the endpoint and time the round trip to it.

    Returns None rather than raising: a benchmark should never fail because
    it could not describe its own network position.
    """
    try:
        u = urllib.parse.urlparse(base_url)
        host = u.hostname
        if not host:
            return None
        port = u.port or (443 if (u.scheme or "https") == "https" else 80)

        infos = socket.getaddrinfo(host, port, socket.AF_INET,
                                   socket.SOCK_STREAM)
        ips = sorted({i[4][0] for i in infos})
        if not ips:
            return None

        # the address this machine actually sources traffic from, taken from
        # the routing table rather than from a lookup service. a UDP connect
        # sends nothing, it just asks the kernel which interface it would
        # use for that destination.
        egress = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect((ips[0], port))
                egress = s.getsockname()[0]
            finally:
                s.close()
        except OSError:
            pass

        rtts: list[float] = []
        for i in range(max(1, samples)):
            ip = ips[i % len(ips)]
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            try:
                t0 = time.perf_counter()
                s.connect((ip, port))
                rtts.append((time.perf_counter() - t0) * 1000.0)
            except OSError:
                continue
            finally:
                s.close()
        if not rtts:
            return None

        return {
            "client_hostname": socket.gethostname(),
            "client_egress_ip": egress,
            "endpoint_host": host,
            "endpoint_ips": ips,
            "rtt_ms": round(min(rtts), 1),
            "rtt_median_ms": round(sorted(rtts)[len(rtts) // 2], 1),
            "samples": len(rtts),
            "note": (
                "round trip is the minimum TCP connect over "
                f"{len(rtts)} tries, which is the floor set by distance "
                "rather than an average carrying queueing noise. every "
                "latency figure in this report contains at least one of "
                "these, because the request has to reach the endpoint "
                "and the first token has to come back."
            ),
        }
    except Exception:
        return None
