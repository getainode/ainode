"""Dataset URL destination policy, transport pinning, and caller authority."""

import asyncio
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ainode.datasets import url_import
from ainode.datasets.api_routes import handle_create
from ainode.datasets.manager import DatasetManager


class Request(dict):
    def __init__(self, manager, url, *, identity=None, enabled=True,
                 remote="198.51.100.5", field="url"):
        super().__init__(identity or {})
        self.app = {"dataset_manager": manager,
                    "auth_config": SimpleNamespace(enabled=enabled)}
        self.remote = remote
        # A body flag is never authority to reach internal destinations.
        self.body = {"source": "url", field: url, "allow_internal": True}

    async def json(self):
        return self.body


@pytest.fixture
def internal_server():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'{"text":"internal-proof"}\n'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/internal", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_ordinary_loopback_import_is_refused(tmp_path, internal_server):
    url, hits = internal_server
    mgr = DatasetManager(tmp_path)
    request = Request(mgr, url, identity={"authenticated": True,
                      "user": "reader", "user_role": "member"}, remote="127.0.0.1")
    response = await handle_create(request)
    assert response.status == 400
    assert "admin or operator" in response.text
    assert hits == []
    assert mgr.list() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("identity,enabled,remote,allowed", [
    ({"authenticated": True, "user": "reader", "user_role": "member"}, True, "127.0.0.1", False),
    ({"authenticated": True, "user": "owner", "user_role": "admin"}, True, "198.51.100.5", True),
    ({"authenticated": True, "api_key_id": "operator"}, True, "198.51.100.5", True),
    ({}, False, "198.51.100.5", False),
    ({}, False, "127.0.0.1", True),
])
async def test_internal_url_authority(tmp_path, internal_server, identity, enabled, remote, allowed):
    url, hits = internal_server
    mgr = DatasetManager(tmp_path)
    response = await handle_create(Request(mgr, url, identity=identity, enabled=enabled,
                                           remote=remote, field="path"))
    assert response.status == (201 if allowed else 400)
    assert hits == (["/internal"] if allowed else [])
    if allowed:
        dataset = json.loads(response.text)
        assert mgr.preview(dataset["id"])["samples"] == [{"text": "internal-proof"}]


def _address(ip, port=80):
    ipv6 = ":" in ip
    return (socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM,
            socket.IPPROTO_TCP, "", (ip, port, 0, 0) if ipv6 else (ip, port))


@pytest.fixture
def transport(monkeypatch):
    """Mock DNS and raw sockets so public-looking targets never leave the test."""
    resolver = Mock(return_value=[_address("8.8.8.8")])
    sockets = []
    replies = [b'HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{"text":"ok"}']

    def make_socket(*args):
        sock = Mock()
        sock.makefile.return_value = io.BytesIO(replies.pop(0))
        sockets.append(sock)
        return sock

    factory = Mock(side_effect=make_socket)
    monkeypatch.setattr(url_import, "socket", SimpleNamespace(
        SOCK_STREAM=socket.SOCK_STREAM, getaddrinfo=resolver, socket=factory,
    ))
    # Neither an environment proxy nor urllib's global opener gets a say.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    return SimpleNamespace(resolver=resolver, sockets=sockets, replies=replies, factory=factory)


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
    "100.64.0.1", "0.0.0.0", "192.0.0.8", "192.0.2.1", "192.88.99.1", "198.18.0.1",
    "224.0.0.1", "240.0.0.1", "255.255.255.255", "::", "::1", "fc00::1",
    "fe80::1", "fec0::1", "ff02::1", "2001:db8::1", "::ffff:127.0.0.1",
    "::ffff:8.8.8.8", "64:ff9b::7f00:1", "2002:7f00:1::", "2001::1", "3fff::1",
])
def test_special_use_addresses_refused(tmp_path, transport, ip):
    transport.resolver.return_value = [_address(ip)]
    mgr = DatasetManager(tmp_path)
    with pytest.raises(ValueError, match="admin or operator"):
        mgr.add_url("http://dataset.example/data.jsonl")
    transport.factory.assert_not_called()
    assert mgr.list() == []


@pytest.mark.parametrize("host", ["localhost", "2130706433", "0x7f000001", "0177.0.0.1"])
def test_alternate_host_spellings_resolved_and_refused(tmp_path, transport, host):
    transport.resolver.return_value = [_address("127.0.0.1")]
    with pytest.raises(ValueError, match="admin or operator"):
        DatasetManager(tmp_path).add_url(f"http://{host}/data")
    transport.factory.assert_not_called()


def test_every_resolved_address_is_checked(tmp_path, transport):
    transport.resolver.return_value = [_address("8.8.8.8"), _address("::1")]
    with pytest.raises(ValueError, match="admin or operator"):
        DatasetManager(tmp_path).add_url("http://dataset.example/data")
    transport.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ip", ["8.8.8.8", "2606:4700:4700::1111"])
async def test_public_url_is_allowed_and_pinned(tmp_path, transport, ip):
    # A second DNS lookup would see loopback. The raw connect must use the first.
    transport.resolver.side_effect = [[_address(ip)], [_address("127.0.0.1")]]
    mgr = DatasetManager(tmp_path)
    response = await handle_create(Request(mgr, "http://dataset.example/data.jsonl?download=1"))
    assert response.status == 201
    transport.resolver.assert_called_once_with("dataset.example", 80, type=socket.SOCK_STREAM)
    sock = transport.sockets[0]
    sock.connect.assert_called_once_with(_address(ip)[4])
    wire = b"".join(call.args[0] for call in sock.sendall.call_args_list)
    assert b"GET /data.jsonl?download=1 HTTP/1.1" in wire
    assert b"Host: dataset.example\r\n" in wire
    dataset = json.loads(response.text)
    assert mgr.preview(dataset["id"])["samples"] == [{"text": "ok"}]


@pytest.mark.parametrize("location", ["http://127.0.0.1/internal", "http://private.example/internal",
                                     "//private.example/internal"])
def test_redirect_to_internal_is_refused(tmp_path, transport, location):
    transport.replies[:] = [f"HTTP/1.1 302 Found\r\nLocation: {location}\r\n\r\n".encode()]
    transport.resolver.side_effect = [[_address("8.8.8.8")], [_address("127.0.0.1")]]
    mgr = DatasetManager(tmp_path)
    with pytest.raises(ValueError, match="admin or operator"):
        mgr.add_url("http://dataset.example/data")
    assert len(transport.sockets) == 1
    assert mgr.list() == []


def test_public_redirect_and_https_hostname_preserved(tmp_path, transport, monkeypatch):
    transport.replies.insert(0, b"HTTP/1.1 302 Found\r\nLocation: https://cdn.example/data\r\n\r\n")
    transport.resolver.side_effect = [[_address("8.8.8.8")], [_address("1.1.1.1", 443)]]
    context = Mock()
    context.wrap_socket.side_effect = lambda sock, **kwargs: sock
    monkeypatch.setattr(url_import.http.client.ssl, "_create_default_https_context",
                        Mock(return_value=context))
    mgr = DatasetManager(tmp_path)
    ds = mgr.add_url("http://dataset.example/data")
    assert mgr.preview(ds.id)["samples"] == [{"text": "ok"}]
    context.wrap_socket.assert_called_once_with(transport.sockets[1], server_hostname="cdn.example")
    transport.sockets[1].connect.assert_called_once_with(("1.1.1.1", 443))


def test_redirect_limit(tmp_path, transport):
    transport.replies[:] = [b"HTTP/1.1 302 Found\r\nLocation: /again\r\n\r\n"] * 6
    with pytest.raises(ValueError, match="limit exceeded"):
        DatasetManager(tmp_path).add_url("http://dataset.example/data")
    assert len(transport.sockets) == 6


@pytest.mark.parametrize("location", ["file:///etc/passwd", "ftp://dataset.example/data"])
def test_redirect_scheme_is_checked(tmp_path, transport, location):
    transport.replies[:] = [f"HTTP/1.1 302 Found\r\nLocation: {location}\r\n\r\n".encode()]
    with pytest.raises(ValueError, match="http"):
        DatasetManager(tmp_path).add_url("http://dataset.example/data")
    assert len(transport.sockets) == 1


def test_size_limit_still_applies(tmp_path, transport, monkeypatch):
    monkeypatch.setattr("ainode.datasets.manager.MAX_UPLOAD_BYTES", 5)
    mgr = DatasetManager(tmp_path)
    with pytest.raises(ValueError, match="max size"):
        mgr.add_url("http://dataset.example/data")
    assert mgr.list() == []
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.asyncio
async def test_concurrent_downloads_register_on_event_loop(tmp_path, transport, monkeypatch):
    mgr = DatasetManager(tmp_path)
    prepare = mgr._prepare_url
    barrier = threading.Barrier(2, timeout=5)
    loop_thread = threading.get_ident()
    prepare_threads = []
    save_threads = []
    save = mgr._save

    def prepare_together(*args, **kwargs):
        prepare_threads.append(threading.get_ident())
        barrier.wait()
        return prepare(*args, **kwargs)

    def record_save():
        save_threads.append(threading.get_ident())
        save()

    transport.replies *= 2
    monkeypatch.setattr(mgr, "_prepare_url", prepare_together)
    monkeypatch.setattr(mgr, "_save", record_save)
    responses = await asyncio.gather(*[
        handle_create(Request(mgr, f"http://dataset.example/{number}.jsonl"))
        for number in range(2)
    ])
    assert [response.status for response in responses] == [201, 201]
    assert all(thread != loop_thread for thread in prepare_threads)
    assert save_threads == [loop_thread, loop_thread]
    assert len(DatasetManager(tmp_path).list()) == 2
