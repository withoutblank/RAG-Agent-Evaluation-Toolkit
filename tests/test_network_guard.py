"""Regression tests for the suite-wide offline network boundary."""

from __future__ import annotations

import socket

import pytest


def test_external_socket_connections_are_blocked() -> None:
    """A numeric documentation address is rejected before a network call occurs."""

    with socket.socket() as client:
        with pytest.raises(RuntimeError, match="Outbound network access is disabled"):
            client.connect(("192.0.2.1", 443))


def test_external_dns_resolution_is_blocked() -> None:
    """External hostnames cannot trigger DNS lookups during tests."""

    with pytest.raises(RuntimeError, match="Outbound network access is disabled"):
        socket.getaddrinfo("example.com", 443)
