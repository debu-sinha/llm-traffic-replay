"""Where the client sits relative to the endpoint, measured not assumed.

Every latency figure this harness reports includes network transit: the
request travels out and response bytes travel back. Run the generator in the
wrong region and that distance is silently folded into TTFT, end-to-end, and
any SLA judgment made from them.

This was not hypothetical. A load test that produced TTFT p50 842 ms against
a 500 ms target was generated from a US east coast machine against an
endpoint in us-west-2, and 82 ms of that number was the width of the
country. The tool reported the latency and said nothing about the geography,
so the only reason it came to light was somebody asking.

The diagnostic is the minimum TCP connect duration over a few tries. A TCP
connect generally needs one handshake round trip, but the duration is not an
exact RTT measurement and it cannot be subtracted from TTFT to recover
endpoint processing time. Minimum rather than mean gives a useful path floor
without presenting queueing noise as distance. Nothing here reaches a third
party: no geolocation service or public-IP lookup is used.

Stdlib only.
"""

from __future__ import annotations

import math
import socket
import statistics
import time

from .client import normalized_origin


def measure_network_path(
    base_url: str, samples: int = 5, timeout: float = 5.0
) -> dict | None:
    """Resolve the endpoint and time TCP connection establishment to it.

    Returns None rather than raising: a benchmark should never fail because
    it could not describe its own network position.
    """
    try:
        if isinstance(samples, bool) or not isinstance(samples, int) \
                or samples <= 0:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
                or not math.isfinite(float(timeout)) or timeout <= 0:
            return None
        _, host, port = normalized_origin(base_url)

        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC,
                                   socket.SOCK_STREAM)
        endpoints = []
        seen = set()
        for family, socktype, proto, _, address in infos:
            key = (family, address)
            if key not in seen:
                seen.add(key)
                endpoints.append((family, socktype, proto, address))
        endpoints.sort(key=lambda item: (item[0], item[3][0]))
        if not endpoints:
            return None
        ips = sorted({item[3][0] for item in endpoints})

        connect_times: list[float] = []
        for i in range(samples):
            family, socktype, proto, address = endpoints[i % len(endpoints)]
            s = socket.socket(family, socktype, proto)
            s.settimeout(timeout)
            try:
                t0 = time.perf_counter()
                s.connect(address)
                connect_times.append((time.perf_counter() - t0) * 1000.0)
            except OSError:
                continue
            finally:
                s.close()
        if not connect_times:
            return None

        return {
            "endpoint_host": host,
            "endpoint_ips": ips,
            "tcp_connect_min_ms": round(min(connect_times), 1),
            "tcp_connect_median_ms": round(
                statistics.median(connect_times), 1),
            "samples": len(connect_times),
            "note": (
                "minimum and median TCP connect duration over "
                f"{len(connect_times)} tries, with DNS lookup outside the "
                "timer. this is a network-path floor and location "
                "diagnostic, not an exact RTT or endpoint processing-time "
                "measurement. do not subtract it from TTFT."
            ),
        }
    except Exception:
        return None
