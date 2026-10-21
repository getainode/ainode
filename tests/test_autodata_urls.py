"""AutoData shares dataset destination checks and retains original caller authority."""

import asyncio
import importlib
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ainode.datasets import url_import
from ainode.training.api_routes import handle_autodata_run
from ainode.training.autodata import core, meta


ROLES = ("challenger", "weak", "strong", "judge")


def config(url="http://model.example/v1"):
    return {"task_spec": "x", "gen_prompt": "x", "retries": 0,
            **{role: {"url": url, "model": role} for role in ROLES}}


class Request(dict):
    def __init__(self, tmp_path, body, *, identity=None, enabled=True,
                 remote="198.51.100.5"):
        super().__init__(identity or {})
        self.app = {"dataset_manager": SimpleNamespace(root=tmp_path),
                    "auth_config": SimpleNamespace(enabled=enabled), "autodata_runs": {}}
        self.remote = remote
        self.body = body

    async def json(self):
        return self.body


async def completed(request):
    for _ in range(200):
        result = next(iter(request.app["autodata_runs"].values()))
        if result["status"] != "running":
            return result
        await asyncio.sleep(0.01)
    pytest.fail("AutoData did not finish")


@pytest.fixture
def local_engine():
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            hits.append((self.path, body, self.headers.get("Authorization")))
            reply = b'{"choices":[{"message":{"content":"ok"}}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_original_confused_post_never_reaches_host_route(local_engine):
    url, hits = local_engine
    with pytest.raises(ValueError):
        core._http_chat(core.Endpoint(url + "/api/engine/update?", "model"), [])
    assert hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("is_meta", [False, True])
@pytest.mark.parametrize("identity,enabled,remote,allowed", [
    ({"authenticated": True, "user": "reader", "user_role": "member"}, True, "127.0.0.1", False),
    ({"authenticated": True, "user": "owner", "user_role": "admin"}, True, "198.51.100.5", True),
    ({"authenticated": True, "api_key_id": "operator"}, True, "198.51.100.5", True),
    ({}, False, "198.51.100.5", False),
    ({}, False, "127.0.0.1", True),
])
async def test_every_role_uses_original_authority(tmp_path, monkeypatch, local_engine,
                                               role, is_meta, identity, enabled, remote, allowed):
    url, hits = local_engine

    def exercise(cfg, **kwargs):
        assert core._http_chat(getattr(cfg, role), []) == "ok"
        return {"report": {}, "dataset": [], "best_yield": 0, "rounds": []}

    monkeypatch.setattr(core, "run", exercise)
    monkeypatch.setattr(meta, "meta_optimize", exercise)
    request = Request(tmp_path, {"config": config(url + "/v1"), "meta": is_meta,
                                 "allow_internal": True},
                      identity=identity, enabled=enabled, remote=remote)
    response = await handle_autodata_run(request)
    assert response.status == 202
    result = await completed(request)
    assert result["status"] == ("completed" if allowed else "failed"), result
    if allowed:
        assert len(hits) == 1
        assert hits[0][0] == "/v1/chat/completions"
        assert hits[0][1]["model"] == role
    else:
        assert "admin or operator" in result["error"]
        assert hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("url", [
    "http://model.example/api/engine/update?", "http://model.example/api/engine/update?x=",
    "http://model.example/api/engine/update#", "http://model.example/api/engine/update#x",
    "http://user:pass@model.example/v1", "http://user@model.example/v1",
    "file:///etc/passwd", "u",
])
async def test_bad_base_urls_rejected_before_queue(tmp_path, role, url):
    cfg = config()
    cfg[role]["url"] = url
    request = Request(tmp_path, {"config": cfg})
    response = await handle_autodata_run(request)
    assert response.status == 400
    assert request.app["autodata_runs"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ROLES)
async def test_endpoint_json_cannot_grant_internal_authority(tmp_path, role):
    cfg = config()
    cfg[role]["allow_internal"] = True
    request = Request(tmp_path, {"config": cfg})
    response = await handle_autodata_run(request)
    assert response.status == 400
    assert request.app["autodata_runs"] == {}


def address(ip, port=80):
    ipv6 = ":" in ip
    return (socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM,
            socket.IPPROTO_TCP, "", (ip, port, 0, 0) if ipv6 else (ip, port))


@pytest.fixture
def transport(monkeypatch):
    resolver = Mock(return_value=[address("8.8.8.8")])
    sockets = []
    body = b'{"choices":[{"message":{"content":"ok"}}]}'
    replies = [b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body]

    def make_socket(*args):
        sock = Mock()
        sock.makefile.return_value = io.BytesIO(replies.pop(0))
        sockets.append(sock)
        return sock

    factory = Mock(side_effect=make_socket)
    monkeypatch.setattr(url_import, "socket", SimpleNamespace(
        SOCK_STREAM=socket.SOCK_STREAM, getaddrinfo=resolver, socket=factory))
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    return SimpleNamespace(resolver=resolver, sockets=sockets, replies=replies, factory=factory)


@pytest.mark.parametrize("ip", ["8.8.8.8", "2606:4700:4700::1111"])
def test_public_model_post_is_pinned(transport, ip):
    transport.resolver.side_effect = [[address(ip)], [address("127.0.0.1")]]
    ep = core.Endpoint("http://model.example/v1/", "strong", api_key="synthetic-key")
    assert core._http_chat(ep, [{"role": "user", "content": "hello"}], True) == "ok"
    transport.resolver.assert_called_once_with("model.example", 80, type=socket.SOCK_STREAM)
    transport.sockets[0].connect.assert_called_once_with(address(ip)[4])
    wire = b"".join(call.args[0] for call in transport.sockets[0].sendall.call_args_list)
    assert b"POST /v1/chat/completions HTTP/1.1\r\n" in wire
    assert b"Host: model.example\r\n" in wire
    assert b"Authorization: Bearer synthetic-key\r\n" in wire
    body = json.loads(wire.split(b"\r\n\r\n", 1)[1])
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_ordinary_public_api_endpoint_works(tmp_path, transport, monkeypatch):
    def exercise(cfg):
        assert core._http_chat(cfg.challenger, []) == "ok"
        return {"report": {}}

    monkeypatch.setattr(core, "run", exercise)
    request = Request(tmp_path, {"config": config()}, enabled=False)
    assert (await handle_autodata_run(request)).status == 202
    assert (await completed(request))["status"] == "completed"
    transport.sockets[0].connect.assert_called_once_with(("8.8.8.8", 80))


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
    "224.0.0.1", "192.0.2.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1",
])
def test_internal_model_destinations_refused(transport, ip):
    transport.resolver.return_value = [address(ip)]
    with pytest.raises(ValueError, match="admin or operator"):
        core._http_chat(core.Endpoint("http://model.example/v1", "model"), [])
    transport.factory.assert_not_called()


def test_mixed_public_private_dns_refused(transport):
    transport.resolver.return_value = [address("8.8.8.8"), address("::1")]
    with pytest.raises(ValueError, match="admin or operator"):
        core._http_chat(core.Endpoint("http://model.example/v1", "model"), [])
    transport.factory.assert_not_called()


@pytest.mark.parametrize("location", [
    "http://127.0.0.1/api/engine/update?", "http://private.example/internal",
    "//private.example/internal", "https://other-public.example/v1/chat/completions",
])
def test_cross_origin_model_redirect_refused(transport, location):
    transport.replies[:] = [f"HTTP/1.1 307 Redirect\r\nLocation: {location}\r\n\r\n".encode()]
    with pytest.raises(ValueError, match="origin"):
        core._http_chat(core.Endpoint("http://model.example/v1", "model"), [])
    assert len(transport.sockets) == 1


@pytest.mark.parametrize("status", [301, 302, 303])
def test_redirect_cannot_change_model_method(transport, status):
    transport.replies[:] = [f"HTTP/1.1 {status} Redirect\r\nLocation: /other\r\n\r\n".encode()]
    with pytest.raises(ValueError, match="method"):
        core._http_chat(core.Endpoint("http://model.example/v1", "model"), [])
    assert len(transport.sockets) == 1


def test_same_origin_redirect_revalidates_dns(transport):
    transport.replies[:] = [b"HTTP/1.1 307 Redirect\r\nLocation: /other\r\n\r\n"]
    transport.resolver.side_effect = [[address("8.8.8.8")], [address("127.0.0.1")]]
    with pytest.raises(ValueError, match="admin or operator"):
        core._http_chat(core.Endpoint("http://model.example/v1", "model"), [])
    assert len(transport.sockets) == 1


def test_same_origin_https_redirect_preserves_post_key_and_sni(transport, monkeypatch):
    transport.replies.insert(0, b"HTTP/1.1 308 Redirect\r\nLocation: /other\r\n\r\n")
    transport.resolver.side_effect = [[address("8.8.8.8", 443)], [address("1.1.1.1", 443)]]
    context = Mock()
    context.wrap_socket.side_effect = lambda sock, **kwargs: sock
    monkeypatch.setattr(url_import.http.client.ssl, "_create_default_https_context",
                        Mock(return_value=context))
    assert core._http_chat(core.Endpoint("https://model.example/v1", "model"), []) == "ok"
    assert context.wrap_socket.call_count == 2
    assert all(call.kwargs == {"server_hostname": "model.example"}
               for call in context.wrap_socket.call_args_list)
    transport.sockets[1].connect.assert_called_once_with(("1.1.1.1", 443))
    wire = b"".join(call.args[0] for call in transport.sockets[1].sendall.call_args_list)
    assert b"POST /other HTTP/1.1\r\n" in wire
    assert b"Authorization: Bearer EMPTY\r\n" in wire


@pytest.mark.parametrize("is_meta", [False, True])
def test_local_cli_retains_operator_access(tmp_path, monkeypatch, local_engine, is_meta):
    cli = importlib.import_module("ainode.training.autodata.run")

    url, hits = local_engine
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(url + "/v1")))

    def exercise(cfg, **kwargs):
        assert core._http_chat(cfg.challenger, []) == "ok"
        return {"report": {"kept": 0, "total": 0, "yield_pct": 0}, "out": "",
                "objective": "yield", "best_score": 0, "best_yield": 0,
                "best_lift": None, "best_prompt": "x", "rounds": [], "dataset": []}

    monkeypatch.setattr(cli, "run", exercise)
    monkeypatch.setattr(meta, "meta_optimize", exercise)
    monkeypatch.setattr("sys.argv", ["autodata", "--config", str(path)] + (["--meta"] if is_meta else []))
    cli.main()
    assert hits[0][0] == "/v1/chat/completions"
