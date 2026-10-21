"""HTTP requests with a checked, pinned destination on every hop."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.parse
from contextlib import contextmanager


# Cover special assignments that older Python ipaddress tables call global,
# including IPv6 translation/tunnelling prefixes that can embed private IPv4.
_SPECIAL_V4 = tuple(map(ipaddress.ip_network, ("192.0.0.0/24", "192.88.99.0/24")))
_GLOBAL_V6 = ipaddress.ip_network("2000::/3")
_SPECIAL_V6 = tuple(map(ipaddress.ip_network, (
    "2001::/23", "2002::/16", "3fff::/20",
)))
_REDIRECTS = {301, 302, 303, 307, 308}


def _public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if address.version == 4:
        return not any(address in network for network in _SPECIAL_V4)
    return address in _GLOBAL_V6 and not any(
        address in network for network in _SPECIAL_V6
    )


def _connect(addresses: list, timeout: float) -> socket.socket:
    """Use the resolver's checked sockaddr directly, without resolving again."""
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
    raise OSError("Could not connect to dataset URL") from last_error


@contextmanager
def open_dataset_url(url: str, *, timeout: float, allow_internal: bool = False):
    """Download a dataset using the shared destination policy."""
    with open_http_url(url, timeout=timeout, allow_internal=allow_internal,
                       label="Dataset URL") as response:
        yield response


@contextmanager
def open_http_url(url: str, *, timeout: float, allow_internal: bool = False,
                  method: str = "GET", data: bytes | None = None,
                  headers: dict | None = None, label: str = "URL"):
    """Follow at most five redirects, checking all addresses before any connect.

    Direct connections deliberately ignore environment proxies. The HTTP host
    and HTTPS verification/SNI use the original hostname; only the socket is
    pinned to the checked resolver results.
    """
    for hop in range(6):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"{label} must be http(s) with a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{label} must not contain credentials")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(
            parsed.hostname, port, type=socket.SOCK_STREAM,
        )
        if not addresses:
            raise ValueError(f"{label} hostname has no addresses")
        if not allow_internal and any(
            not _public_address(item[4][0]) for item in addresses
        ):
            raise ValueError(f"Internal {label.lower()}s require an admin or operator")

        connection_type = (http.client.HTTPSConnection if parsed.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(parsed.hostname, port, timeout=timeout)
        # HTTPConnection and HTTPSConnection both use this factory. HTTPS still
        # wraps the resulting socket with its default verifying SSL context.
        connection._create_connection = lambda *args, **kwargs: _connect(addresses, timeout)
        try:
            path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection.request(method, path, body=data,
                               headers={"User-Agent": "AINode/1.0", **(headers or {})})
            response = connection.getresponse()
            try:
                if response.status in _REDIRECTS:
                    location = response.getheader("Location")
                    if not location or hop == 5:
                        raise ValueError(f"{label} redirect missing a location or limit exceeded")
                    next_url = urllib.parse.urljoin(url, location)
                    if method != "GET":
                        # Never turn a model POST into a GET or send its key and
                        # prompts to another origin. Same-origin 307/308 hops
                        # are still resolved, checked and pinned on the next loop.
                        target = urllib.parse.urlsplit(next_url)
                        target_port = target.port or (443 if target.scheme == "https" else 80)
                        if response.status not in (307, 308) or (
                            target.scheme, target.hostname, target_port
                        ) != (parsed.scheme, parsed.hostname, port):
                            raise ValueError(f"{label} POST redirect must preserve method and origin")
                    url = next_url
                    continue
                if not 200 <= response.status < 300:
                    raise ValueError(f"{label} returned HTTP {response.status}")
                yield response
                return
            finally:
                response.close()
        finally:
            connection.close()
