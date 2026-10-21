"""Public ingress boundaries, with observable sinks and real auth transport."""

import argparse
import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.api import server
from ainode.auth import middleware as auth
from ainode.auth.accounts import UsersStore, may_control_host
from ainode.auth.api_routes import register_auth_routes
from ainode.auth.fleet import fleet_key
from ainode.core.config import NodeConfig
from ainode.engine import maintenance
from ainode.models.api_routes import parse_launch_overrides
from ainode.secrets.api_routes import register_secrets_routes


class Request(dict):
    def __init__(self, body=None, *, authority="member", enabled=True, remote="198.51.100.9"):
        identities = {
            "member": {"authenticated": True, "user": "member", "user_role": "member"},
            "admin": {"authenticated": True, "user": "owner", "user_role": "admin"},
            "operator": {"authenticated": True, "api_key_id": "operator"},
            "fleet": {"authenticated": True, "api_key_id": "fleet"},
            "anonymous": {},
        }
        super().__init__(identities[authority])
        self.body = body or {}
        self.method = "POST"
        self.remote = remote
        self.headers = {"X-AINode-Client": "dashboard"}
        self.query = {}
        self.app = {"config": NodeConfig(node_id="local", cluster_secret="synthetic-fleet-secret"),
                    "auth_config": SimpleNamespace(enabled=enabled)}

    async def json(self):
        return self.body


BAD_ALIASES = ["--trust-remote-code", "--host=0.0.0.0", "--api-key=synthetic",
               "--allowed-local-media-path=/", "-x", "", " ", "a\nb", "a\tb", "a\x00b",
               "a b", "a\\b", "a=flag", "a" * 257, "é", 1, None]


@pytest.mark.parametrize("alias", BAD_ALIASES)
def test_alias_parser_refuses_every_option_or_malformed_value(alias):
    overrides, error = parse_launch_overrides({"served_model_name": ["normal", alias]}, privileged=True)
    assert error and "served_model_name" in error
    assert not overrides


@pytest.mark.parametrize("value", [[], {}, None, 3, ["a"] * 33])
def test_alias_container_is_strict(value):
    assert parse_launch_overrides({"served_model_name": value}, privileged=True)[1]


def test_valid_aliases_stay_values_in_real_multivalue_parser():
    names = ["org/model", "Aegis-14B", "model.v1:latest", "a_b@v+1", "a" * 256]
    overrides, error = parse_launch_overrides({"served_model_name": names}, privileged=True)
    assert error is None and overrides["served_model_name"] == names
    parser = argparse.ArgumentParser()
    parser.add_argument("--served-model-name", nargs="+")
    parser.add_argument("--trust-remote-code", action="store_true")
    result = parser.parse_args(["--served-model-name", *overrides["served_model_name"]])
    assert result.served_model_name == names and result.trust_remote_code is False


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["solo", "stacked", "sharded", "fleet-local", "fleet-remote"])
@pytest.mark.parametrize("authority,alias",
    [("member", alias) for alias in BAD_ALIASES[:4] + ["ordinary-alias"]]
    + [("operator", alias) for alias in BAD_ALIASES[:4]])
async def test_unprivileged_alias_never_reaches_launch_or_forward(path, alias, authority, monkeypatch):
    from ainode.models import api_routes as models
    from ainode.engine.sharding_routes import handle_sharding_launch
    req = Request({"model": "synthetic/model", "served_model_name": ["normal", alias],
                   "min_nodes": 2}, authority=authority)
    req.app["client_session"] = Mock()
    req.app["engine"] = Mock()
    req.app["instances"] = Mock() if path == "stacked" else None
    backend = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr("ainode.engine.backends.get_backend", backend)
    if path.startswith("fleet"):
        req.body["node_id"] = "local" if path.endswith("local") else "remote"
        response = await server.handle_cluster_load(req)
    elif path == "sharded":
        response = await handle_sharding_launch(req)
    else:
        response = await models.handle_model_load(req)
    assert response.status in (400, 403)
    req.app["client_session"].post.assert_not_called()
    backend.assert_not_called()
    req.app["engine"].stop.assert_not_called()


@pytest.mark.parametrize("origin", ["http://localhost.evil.example", "http://127.0.0.1.evil.example",
    "http://localhost@evil.example", "http://evil@localhost", "http://localhost/path",
    "http://localhost?x=1", "http://localhost#x", "http://localhost:bad",
    "http://localhost:65536", "http://[::1].evil.example", "null", "http://localhost\n"])
def test_origin_is_a_whole_parsed_origin(origin):
    assert server.cors_allowed_origin(NodeConfig(), origin) == ""


@pytest.mark.parametrize("origin", ["http://localhost:3000", "https://localhost:4443",
    "http://127.0.0.1:50123", "http://127.0.0.2", "http://[::1]:3000"])
def test_exact_loopback_origins_keep_ports_and_ipv6(origin):
    assert server.cors_allowed_origin(NodeConfig(), origin) == origin


@pytest.mark.asyncio
async def test_open_loopback_key_mint_requires_unsafelisted_header(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_FILE", tmp_path / "auth.json")
    cfg = auth.AuthConfig(enabled=False)
    app = web.Application(middlewares=[server.cors_middleware, auth.auth_middleware])
    app["auth_config"] = cfg
    app["config"] = NodeConfig(cors_origins="*")
    register_auth_routes(app)
    async with TestClient(TestServer(app)) as client:
        for origin in ("http://localhost.evil.example", "http://localhost:9123", "https://evil.example"):
            preflight = await client.options("/api/auth/keys", headers={"Origin": origin,
                "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-AINode-Client"})
            assert "x-ainode-client" not in preflight.headers["Access-Control-Allow-Headers"].lower()
            denied = await client.post("/api/auth/keys", json={}, headers={"Origin": origin})
            assert denied.status == 403 and not cfg.api_keys
        accepted = await client.post("/api/auth/keys", json={}, headers={"X-AINode-Client": "dashboard"})
        assert accepted.status == 200 and len(cfg.api_keys) == 1


@pytest.mark.parametrize("remote", ["198.51.100.9", "127.0.0.1", "::1"])
def test_loopback_trust_ignores_forwarded_addresses_and_needs_csrf(remote):
    req = Request(authority="anonymous", enabled=False, remote=remote)
    req.headers = {"X-Forwarded-For": "127.0.0.1", "Forwarded": "for=127.0.0.1"}
    assert may_control_host(req) is False
    req.headers["X-AINode-Client"] = "dashboard"
    assert may_control_host(req) is (remote != "198.51.100.9")


SECRET_ROUTES = [("GET", "/api/secrets"), ("GET", "/api/secrets/custom"),
    ("PUT", "/api/secrets/huggingface_token"), ("DELETE", "/api/secrets/huggingface_token"),
    ("GET", "/api/secrets/huggingface_token/test"),
    ("PUT", "/api/secrets/custom/example"), ("DELETE", "/api/secrets/custom/example")]


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["member", "operator", "admin", "fleet", "remote-open", "loopback-open"])
async def test_every_secret_route_checks_real_identity_before_manager(authority, tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_FILE", tmp_path / "auth.json")
    cfg = auth.AuthConfig(enabled=not authority.endswith("open"), api_keys=[{
        "id": "operator", "key_hash": hashlib.sha256(b"synthetic-operator").hexdigest()}])
    users = UsersStore(tmp_path / "users.json")
    users.add_user("person", "synthetic password", role="admin" if authority == "admin" else "member")
    cookie, _ = users.create_session("person")
    @web.middleware
    async def transport(request, handler):
        # Model a transport peer; never trust a forwarded-address header.
        if authority == "remote-open":
            request = request.clone(remote="198.51.100.9")
        return await handler(request)
    app = web.Application(middlewares=[transport, auth.auth_middleware])
    app["config"] = NodeConfig(cluster_secret="synthetic-fleet-secret")
    app["auth_config"] = cfg
    app["users_store"] = users
    mgr = Mock()
    mgr.all.return_value = {"huggingface_token": {"masked": "masked"}}
    mgr.custom_all.return_value = {"example": {"masked": "masked"}}
    mgr.get.return_value = None  # validation must stop before external I/O
    app["secrets_manager"] = mgr
    register_secrets_routes(app)
    headers = {"X-AINode-Client": "dashboard", "X-Forwarded-For": "127.0.0.1",
               "X-AINode-Launch-Authorized": "true", "X-AINode-Maintenance-Authorized": "true"}
    if authority in ("admin", "member"):
        headers["Cookie"] = f"ainode_session={cookie}"
    elif authority in ("operator", "fleet"):
        token = "synthetic-operator" if authority == "operator" else fleet_key("synthetic-fleet-secret")
        headers["Authorization"] = f"Bearer {token}"
    allowed = authority not in ("member", "remote-open")
    async with TestClient(TestServer(app)) as client:
        for method, path in SECRET_ROUTES:
            mgr.reset_mock()
            response = await client.request(method, path, json={"value": "synthetic-value"}, headers=headers)
            assert (response.status != 403) is allowed, (authority, method, path, await response.text())
            assert bool(mgr.mock_calls) is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "load", "unload", "sharding", "cluster-hold", "cluster-load", "cluster-unload"])
@pytest.mark.parametrize("authority,enabled,remote", [("member", True, "127.0.0.1"),
    ("anonymous", False, "198.51.100.9"), ("fleet", True, "198.51.100.9")])
async def test_hold_denial_precedes_every_local_or_fleet_effect(path, authority, enabled, remote, tmp_path, monkeypatch):
    from ainode.models import api_routes as models
    from ainode.engine.sharding_routes import handle_sharding_launch
    monkeypatch.setattr(maintenance, "hold_path", lambda: tmp_path / "maintenance.json")
    req = Request({"model": "synthetic/model", "node_id": "remote", "min_nodes": 2,
        "maintenance": {"reason": "synthetic hold", "until": time.time() + 3600}},
        authority=authority, enabled=enabled, remote=remote)
    if authority != "fleet":
        req.headers["X-AINode-Maintenance-Authorized"] = "true"  # forged marker is no authority
    req.app["client_session"] = Mock()
    req.app["engine"] = Mock()
    handlers = {"direct": maintenance.handle_maintenance, "load": models.handle_model_load,
                "unload": models.handle_model_unload, "sharding": handle_sharding_launch,
                "cluster-hold": server.handle_cluster_hold, "cluster-load": server.handle_cluster_load,
                "cluster-unload": server.handle_cluster_unload}
    response = await handlers[path](req)
    assert response.status == 403
    assert not maintenance.hold_path().exists()
    req.app["client_session"].post.assert_not_called()
    req.app["engine"].stop.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["admin", "operator", "loopback", "fleet"])
async def test_authorized_hold_set_and_clear(authority, tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, "hold_path", lambda: tmp_path / "maintenance.json")
    req = Request({"reason": "synthetic hold", "until": time.time() + 60},
        authority="anonymous" if authority == "loopback" else authority,
        enabled=authority != "loopback", remote="127.0.0.1")
    if authority == "fleet":
        req.headers["X-AINode-Maintenance-Authorized"] = "true"
    assert (await maintenance.handle_maintenance(req)).status == 200
    assert maintenance.held()
    req.body = {"maintenance": None}
    assert (await maintenance.handle_maintenance(req)).status == 200
    assert maintenance.maintenance() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/engine/hold", "/api/models/load", "/api/models/unload"])
async def test_fleet_hold_marker_is_stamped_after_original_authority(path, monkeypatch):
    req = Request({"node_id": "remote", "model": "synthetic/model", "maintenance": None}, authority="operator")
    peer = SimpleNamespace(web_port=3000)
    req.app["cluster_state"] = Mock(get_node=Mock(return_value=peer))
    upstream = SimpleNamespace(status=200, headers={}, read=AsyncMock(return_value=b"{}"))
    context = Mock(__aenter__=AsyncMock(return_value=upstream), __aexit__=AsyncMock())
    req.app["client_session"] = Mock(post=Mock(return_value=context))
    monkeypatch.setattr(server, "best_host", lambda *a: "198.51.100.8")
    assert (await server._cluster_dispatch(req, path)).status == 200
    sent = req.app["client_session"].post.call_args.kwargs
    assert sent["headers"]["X-AINode-Maintenance-Authorized"] == "true"
    assert sent["headers"]["Authorization"] == "Bearer " + fleet_key("synthetic-fleet-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["local", "peer"])
@pytest.mark.parametrize("credential", ["operator", "cookie", "fleet"])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses",
                                  "/v1/audio/transcriptions", "/v1/embeddings"])
async def test_protocol_reaches_engine_without_node_credentials(path, credential, placement, tmp_path, monkeypatch):
    import aiohttp
    from ainode.discovery.broadcast import NodeStatus
    from ainode.discovery.cluster import ClusterNode, ClusterState
    from ainode.embeddings.api_routes import handle_v1_embeddings
    seen = []
    async def engine(request):
        seen.append((dict(request.headers), await request.read(), request.path_qs))
        if path == "/v1/responses":
            return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")
        return web.json_response({"synthetic": True})
    engine_app = web.Application()
    engine_app.router.add_post("/{path:.*}", engine)
    monkeypatch.setattr(auth, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.delenv("AINODE_ENGINE_API_KEY", raising=False)
    async with TestServer(engine_app) as upstream, aiohttp.ClientSession() as session:
        cluster = ClusterState()
        node_id = "local" if placement == "local" else "peer"
        cluster.add_node(ClusterNode(node_id=node_id, node_name=node_id, gpu_name="synthetic",
            gpu_memory_gb=1, unified_memory=True, model="synthetic/model", status=NodeStatus.ONLINE,
            api_port=upstream.port, web_port=3000, last_seen=0, fabric_ip="127.0.0.1"))
        app = web.Application(middlewares=[auth.auth_middleware])
        app["config"] = NodeConfig(node_id="local", cluster_secret="synthetic-fleet-secret", api_port=upstream.port)
        app["cluster_state"] = cluster
        app["client_session"] = session
        app["metrics_collector"] = Mock()
        app["embedding_manager"] = Mock()
        app["auth_config"] = auth.AuthConfig(enabled=True, api_keys=[{
            "id": "operator", "key_hash": hashlib.sha256(b"synthetic-operator").hexdigest()}])
        store = UsersStore(tmp_path / "users.json")
        store.add_user("owner", "synthetic password", role="admin")
        cookie, _ = store.create_session("owner")
        app["users_store"] = store
        handler = handle_v1_embeddings if path == "/v1/embeddings" else server.proxy_to_vllm
        app.router.add_post(path, handler)
        headers = {"X-AINode-Client": "dashboard", "X-AINode-Forwarded-By": "forged",
            "X-AINode-Launch-Authorized": "true", "X-AINode-Maintenance-Authorized": "true",
            "X-API-Key": "synthetic-alternative", "Forwarded": "for=127.0.0.1",
            "X-Forwarded-For": "127.0.0.1", "Anthropic-Version": "2023-06-01",
            "Anthropic-Beta": "synthetic-beta", "OpenAI-Beta": "responses=v1"}
        if credential == "cookie":
            headers["Cookie"] = f"ainode_session={cookie}"
        else:
            token = "synthetic-operator" if credential == "operator" else fleet_key("synthetic-fleet-secret")
            headers["Authorization"] = f"Bearer {token}"
        if "audio" in path:
            headers["Content-Type"] = "multipart/form-data; boundary=synthetic-boundary"
            body = (b'--synthetic-boundary\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
                    b'synthetic/model\r\n--synthetic-boundary\r\nContent-Disposition: form-data; name="file"; '
                    b'filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\nSYNTHETIC\r\n--synthetic-boundary--\r\n')
        else:
            headers["Content-Type"] = "application/json"
            body = b'{"model":"synthetic/model","input":"synthetic"}'
        async with TestClient(TestServer(app)) as client:
            response = await client.post(path + "?beta=true", data=body, headers=headers)
            assert response.status == 200, await response.text()
            text = await response.text()
            assert "DONE" in text if path == "/v1/responses" else "synthetic" in text
        assert len(seen) == 1
        received = {key.lower(): value for key, value in seen[0][0].items()}
        for key in ("authorization", "cookie", "x-api-key", "x-ainode-client", "x-ainode-forwarded-by",
                    "x-ainode-launch-authorized", "x-ainode-maintenance-authorized", "forwarded", "x-forwarded-for"):
            assert key not in received
        assert received["anthropic-version"] == "2023-06-01"
        assert received["anthropic-beta"] == "synthetic-beta"
        assert received["openai-beta"] == "responses=v1"
        assert received["content-type"] == headers["Content-Type"]
        assert seen[0][1] == body
        if path != "/v1/embeddings":
            assert seen[0][2].endswith("?beta=true")


def test_upstream_engine_key_is_separate_from_all_caller_headers(monkeypatch):
    from ainode.api.engine_headers import engine_headers
    req = Request()
    req.headers.update({"Authorization": "Bearer synthetic-node-key", "Cookie": "synthetic-cookie",
                        "x-api-key": "synthetic-client-key", "Content-Type": "application/json"})
    monkeypatch.setenv("AINODE_ENGINE_API_KEY", "synthetic-engine-key")
    sent = engine_headers(req)
    assert sent == {"Content-Type": "application/json", "Authorization": "Bearer synthetic-engine-key"}


@pytest.mark.asyncio
async def test_authorized_alias_fleet_transport_keeps_alias_and_launch_approval(monkeypatch):
    req = Request({"node_id": "remote", "model": "synthetic/model", "served_model_name": "valid-alias"}, authority="operator")
    req.app["cluster_state"] = Mock(get_node=Mock(return_value=SimpleNamespace(web_port=3000)))
    upstream = SimpleNamespace(status=200, headers={}, read=AsyncMock(return_value=b"{}"))
    context = Mock(__aenter__=AsyncMock(return_value=upstream), __aexit__=AsyncMock())
    req.app["client_session"] = Mock(post=Mock(return_value=context))
    monkeypatch.setattr(server, "best_host", lambda *a: "198.51.100.8")
    assert (await server.handle_cluster_load(req)).status == 200
    sent = req.app["client_session"].post.call_args.kwargs
    assert sent["json"]["served_model_name"] == "valid-alias"
    assert sent["headers"]["X-AINode-Launch-Authorized"] == "true"


@pytest.mark.asyncio
async def test_sharding_solo_delegation_preserves_validated_controls(monkeypatch):
    from ainode.engine.sharding_routes import handle_sharding_launch
    req = Request({"model": "synthetic/model", "served_model_name": "valid-alias",
                   "maintenance": None, "min_nodes": 1}, authority="operator")
    req.app["cluster_state"] = Mock()
    received = []
    async def load(request):
        received.append((await request.json(), request.get("api_key_id")))
        return web.json_response({"ok": True})
    monkeypatch.setattr("ainode.models.api_routes.handle_model_load", load)
    assert (await handle_sharding_launch(req)).status == 200
    assert received == [(req.body, "operator")]


def test_alias_host_control_requires_authority_even_when_parser_safe():
    overrides, error = parse_launch_overrides({"served_model_name": "ordinary-alias"}, privileged=False)
    assert overrides == {}
    assert "admin or operator" in error


@pytest.mark.asyncio
async def test_member_cannot_forward_hold_to_an_existing_peer(monkeypatch):
    req = Request({"node_id": "remote", "maintenance": None})
    req.app["cluster_state"] = Mock(get_node=Mock(return_value=SimpleNamespace(web_port=3000)))
    upstream = SimpleNamespace(status=200, headers={}, read=AsyncMock(return_value=b"{}"))
    context = Mock(__aenter__=AsyncMock(return_value=upstream), __aexit__=AsyncMock())
    req.app["client_session"] = Mock(post=Mock(return_value=context))
    monkeypatch.setattr(server, "best_host", lambda *a: "198.51.100.8")
    assert (await server.handle_cluster_hold(req)).status == 403
    req.app["client_session"].post.assert_not_called()
