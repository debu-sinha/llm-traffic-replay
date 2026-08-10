"""Deadline-bounded DNS and TCP helpers for every stdlib HTTP path.

``socket`` timeouts do not bound ``getaddrinfo``.  A resolver can therefore
block before a socket exists, beyond both the configured timeout and any
socket watchdog.  Resolution runs in a daemon-only helper which is shared by
concurrent callers for the same target.  A caller stops waiting at its own
deadline; the helper does DNS only, so a late result can never open a socket
or emit an HTTP request.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass, field


_MAX_ACTIVE_DNS_LOOKUPS = 64
_DNS_CANCEL_POLL_S = 0.01


@dataclass
class _Lookup:
    done: threading.Event = field(default_factory=threading.Event)
    result: list | None = None
    error: BaseException | None = None


_lookups_lock = threading.Lock()
_active_lookups: dict[tuple, _Lookup] = {}


def _positive_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("network timeout must be positive and finite")
    return float(value)


def bounded_getaddrinfo(
        host: str, port: int, *, timeout: float,
        family: int = socket.AF_UNSPEC, socktype: int = socket.SOCK_STREAM,
        proto: int = 0, flags: int = 0,
        cancel_event: threading.Event | None = None) -> list:
    """Return ``getaddrinfo`` results without waiting beyond ``timeout``.

    At most one resolver thread is active for an identical lookup.  This is
    important under a resolver outage: hundreds of load workers may time out,
    but they do not create hundreds of stuck non-daemon executor threads.
    Abandoning a wait does not cancel the process-global resolver operation;
    its daemon helper is deliberately limited to DNS and discards a late
    result after waking any remaining waiters.
    """
    timeout_s = _positive_timeout(timeout)
    key = (host, port, family, socktype, proto, flags)

    with _lookups_lock:
        lookup = _active_lookups.get(key)
        if lookup is None:
            if len(_active_lookups) >= _MAX_ACTIVE_DNS_LOOKUPS:
                raise socket.gaierror(
                    socket.EAI_AGAIN,
                    "too many concurrent deadline-bounded DNS lookups")
            lookup = _Lookup()
            _active_lookups[key] = lookup

            def resolve() -> None:
                try:
                    lookup.result = socket.getaddrinfo(
                        host, port, family, socktype, proto, flags)
                except BaseException as exc:
                    lookup.error = exc
                finally:
                    lookup.done.set()
                    with _lookups_lock:
                        if _active_lookups.get(key) is lookup:
                            _active_lookups.pop(key, None)

            threading.Thread(
                target=resolve,
                name="traffic-replay-dns-resolver",
                daemon=True,
            ).start()

    deadline = time.monotonic() + timeout_s
    while not lookup.done.is_set():
        if cancel_event is not None and cancel_event.is_set():
            raise ConnectionAbortedError("DNS lookup cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("DNS lookup exceeded its deadline")
        lookup.done.wait(min(remaining, _DNS_CANCEL_POLL_S))

    # Do not accept a result that raced in after this caller's deadline.
    if time.monotonic() > deadline:
        raise socket.timeout("DNS lookup exceeded its deadline")
    if cancel_event is not None and cancel_event.is_set():
        raise ConnectionAbortedError("DNS lookup cancelled")
    if lookup.error is not None:
        if isinstance(lookup.error, Exception):
            raise lookup.error
        raise RuntimeError(
            "DNS resolver stopped with a non-standard base exception")
    if not isinstance(lookup.result, list) or not lookup.result:
        raise socket.gaierror(socket.EAI_NONAME, "DNS returned no addresses")
    return list(lookup.result)


def deadline_bounded_create_connection(
        address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address=None, *, all_errors: bool = False,
        cancel_event: threading.Event | None = None):
    """``socket.create_connection`` equivalent with a DNS-inclusive bound.

    The one timeout is an absolute budget shared by resolution and every
    resolved address attempt.  Only the caller opens sockets; the resolver
    helper cannot make a late connection after the caller has timed out.
    """
    if not isinstance(address, tuple) or len(address) != 2:
        raise ValueError("connection address must be a (host, port) pair")
    host, port = address
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
        raise ValueError("deadline-bounded connection requires a timeout")
    timeout_s = _positive_timeout(timeout)
    deadline = time.monotonic() + timeout_s
    infos = bounded_getaddrinfo(
        host, port, timeout=timeout_s, cancel_event=cancel_event)
    errors: list[OSError] = []
    for family, socktype, proto, _canonname, sockaddr in infos:
        if cancel_event is not None and cancel_event.is_set():
            raise ConnectionAbortedError("connection cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("connection exceeded its deadline")
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(remaining)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            errors.append(exc)
            if sock is not None:
                sock.close()
    if not errors:
        raise socket.gaierror(socket.EAI_NONAME, "DNS returned no addresses")
    if all_errors:
        # ``all_errors`` is not used by http.client, but accept the modern
        # socket.create_connection signature when running on Python 3.11+.
        try:
            exception_group = ExceptionGroup
        except NameError:  # pragma: no cover - Python 3.10 compatibility
            pass
        else:
            raise exception_group("all resolved addresses failed", errors)
    raise errors[-1]


def bind_deadline_bounded_dns(
        connection, *, cancel_event: threading.Event | None = None):
    """Make one ``http.client`` connection use the bounded TCP helper.

    ``HTTPConnection.connect`` delegates to its ``_create_connection``
    attribute.  Replacing only that hook preserves the original host for the
    HTTP Host header and HTTPS certificate/SNI verification.
    """
    def create(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
               source_address=None, *, all_errors: bool = False):
        return deadline_bounded_create_connection(
            address, timeout, source_address, all_errors=all_errors,
            cancel_event=cancel_event)

    connection._create_connection = create
    return connection


class AbsoluteHTTPDeadline:
    """Interrupt one stdlib HTTP connection at an absolute wall deadline.

    Socket timeouts are idle bounds: a peer can send one byte before each
    timeout forever. This daemon watchdog shuts down (but never closes) the
    attached socket at the command deadline. Keeping the socket object on the
    connection prevents ``http.client`` from auto-connecting and issuing a
    late request in a race with the watchdog.
    """

    def __init__(self, connection, timeout: float):
        self.connection = connection
        self.timeout = _positive_timeout(timeout)
        self.expired = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def watch() -> None:
            if self._stopped.wait(self.timeout):
                return
            self.expired.set()
            while not self._stopped.is_set():
                sock = getattr(self.connection, "sock", None)
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except (AttributeError, OSError, ValueError):
                        pass
                # Retry transient pre-connect/TLS shutdown failures until the
                # owner exits and closes its connection.
                self._stopped.wait(_DNS_CANCEL_POLL_S)

        self._thread = threading.Thread(
            target=watch,
            name="traffic-replay-http-deadline-watchdog",
            daemon=True,
        )
        self._thread.start()
        return self

    def raise_if_expired(self) -> None:
        if self.expired.is_set():
            raise socket.timeout("HTTP operation exceeded its deadline")

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)



__all__ = [
    "AbsoluteHTTPDeadline",
    "bind_deadline_bounded_dns",
    "bounded_getaddrinfo",
    "deadline_bounded_create_connection",
]
