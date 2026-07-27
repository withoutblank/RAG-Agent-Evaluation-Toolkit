"""Suite-wide pytest safeguards."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Generator
from typing import Any

import pytest

_NETWORK_BLOCK_MESSAGE = (
    "Outbound network access is disabled during tests. "
    "Inject a fake client or use a loopback address."
)


def _is_local_address(address: object) -> bool:
    """Return whether a socket address is local to the test process."""

    if isinstance(address, (str, bytes)):
        # String and bytes addresses represent local Unix-domain sockets.
        return True
    if not isinstance(address, tuple) or not address:
        return False

    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str):
        return False

    normalized_host = host.strip().lower()
    if normalized_host in {"localhost", "localhost.localdomain"}:
        return True
    if not normalized_host:
        return False

    # IPv6 scope identifiers (for example, "::1%1") are not accepted by
    # ipaddress but do not change whether the underlying address is loopback.
    normalized_host = normalized_host.split("%", maxsplit=1)[0]
    try:
        return ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_outbound_network(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Prevent accidental external network calls while allowing local sockets."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    original_sendto = socket.socket.sendto

    def require_local(address: object) -> None:
        if not _is_local_address(address):
            raise RuntimeError(_NETWORK_BLOCK_MESSAGE)

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        require_local(address)
        original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        require_local(address)
        return original_connect_ex(sock, address)

    def guarded_getaddrinfo(
        host: bytes | str | None,
        port: bytes | str | int | None,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        if host is not None:
            require_local((host, port))
        return original_getaddrinfo(host, port, *args, **kwargs)

    def guarded_sendto(
        sock: socket.socket,
        data: bytes,
        *args: Any,
    ) -> int:
        if not args:
            raise TypeError("sendto expected an address")
        require_local(args[-1])
        return original_sendto(sock, data, *args)

    replacements: tuple[tuple[object, str, Callable[..., object]], ...] = (
        (socket.socket, "connect", guarded_connect),
        (socket.socket, "connect_ex", guarded_connect_ex),
        (socket, "getaddrinfo", guarded_getaddrinfo),
        (socket.socket, "sendto", guarded_sendto),
    )
    for target, attribute, replacement in replacements:
        monkeypatch.setattr(target, attribute, replacement)

    yield
