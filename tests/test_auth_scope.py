"""Persisted decision keys are credentials with exactly two allowed routes.

All HTTP tests use real auth middleware and loopback servers. The recording
handler makes a refusal observable independently of its status code.
"""

import hashlib
import json
import stat
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from yarl import URL

from ainode.auth import middleware as auth
from ainode.auth.accounts import (
    UsersStore, may_control_host, may_supply_launch_fields, require_admin,
)
from ainode.auth.api_routes import register_auth_routes
from ainode.auth.fleet import FORWARDED_BY_HEADER, fleet_key
from ainode.auth.session_routes import register_session_routes
from ainode.core.config import NodeConfig
from ainode.ratelimit.middleware import is_forwarded_by_fleet


SECRET = "synthetic-scope-fleet-secret"
SCOPED_TOKEN = "synthetic-decide-token"
OPERATOR_TOKEN = "synthetic-operator-token"
BAD_SCOPES = [None, "", "unknown", "Decide", " decide", [], {}, 1, False]


def _entry(token, key_id, **claims):
    return {"id": key_id, "key_hash": hashlib.sha256(token.encode()).hexdigest(),
            "name": key_id, **claims}


def _headers(token=SCOPED_TOKEN, **extra):
    return {"Authorization": f"Bearer {token}", **extra}


@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AINODE_HOME", tmp_path)
    monkeypatch.setattr(auth, "AUTH_FILE", tmp_path / "auth.json")
    cfg = auth.AuthConfig(enabled=True, api_keys=[
        _entry(SCOPED_TOKEN, "decision-client", scope="decide"),
        _entry(OPERATOR_TOKEN, "old-operator"),
    ])
    cfg.save()
    return tmp_path


@pytest_asyncio.fixture
async def node(auth_home):
    seen = []

    async def record(request):
        seen.append((request.method, request.path))
        return web.json_response({
            "id": request["api_key_id"], "scope": request["api_key_scope"],
            "admin": require_admin(request), "host": may_control_host(request),
            "launch": may_supply_launch_fields(request),
            "forwarded": is_forwarded_by_fleet(request),
            "user": request["user"],
        })

    app = web.Application(middlewares=[auth.auth_middleware])
    app["config"] = NodeConfig(node_id="test-node", cluster_secret=SECRET)
    app["auth_config"] = auth.AuthConfig.load()
    app["users_store"] = UsersStore.load()
    app.router.add_route("*", "/{path:.*}", record)
    async with TestClient(TestServer(app)) as client:
        yield SimpleNamespace(client=client, app=app, seen=seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("POST", "/v1/decide"), ("POST", "/v1/decide?trace=synthetic"),
    ("GET", "/api/health"), ("GET", "/api/health?probe=1"),
])
async def test_only_the_two_method_path_pairs_reach_the_handler(node, method, path):
    response = await node.client.request(method, path, headers=_headers())
    assert response.status == 200
    data = await response.json()
    assert data["id"] == "decision-client" and data["scope"] == "decide"
    assert data["admin"] is data["host"] is data["launch"] is False
    assert node.seen == [(method, path.split("?")[0])]


DENIED = [
    ("GET", "/api/nodes"), ("GET", "/api/status"), ("GET", "/api/config"),
    ("POST", "/api/config"), ("PATCH", "/api/config"),
    ("GET", "/api/auth/keys"), ("POST", "/api/auth/keys"),
    ("DELETE", "/api/auth/keys/old-operator"),
    ("POST", "/api/auth/enable"), ("POST", "/api/auth/disable"),
    ("GET", "/api/auth/users"), ("POST", "/api/auth/users"),
    ("DELETE", "/api/auth/users/example"),
    ("POST", "/api/auth/users/sync"), ("GET", "/api/auth/users/export"),
    ("GET", "/api/auth/sessions"), ("POST", "/api/auth/password"),
    ("POST", "/api/models/load"), ("POST", "/api/models/unload"),
    ("POST", "/api/sharding/launch"), ("POST", "/api/training/submit"),
    ("GET", "/api/training/jobs"), ("POST", "/api/cluster/join"),
    ("GET", "/api/cluster/endpoint"), ("GET", "/api/auth/status"),
    ("POST", "/api/auth/login"), ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"), ("GET", "/"), ("GET", "/static/app.js"),
    ("GET", "/v1/models"),
    *[("POST", path) for path in (
        "/v1/chat/completions", "/v1/completions", "/v1/messages",
        "/v1/messages/count_tokens", "/v1/responses", "/v1/embeddings",
        "/v1/rerank", "/v1/score", "/v1/systemone",
        "/v1/audio/transcriptions", "/v1/audio/translations",
        "/tokenize", "/detokenize",
    )],
    *[(method, "/v1/decide") for method in ("GET", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS")],
    *[(method, "/api/health") for method in ("POST", "HEAD", "PUT", "DELETE", "OPTIONS")],
    ("POST", "/v1/decide/"), ("POST", "/v1/decide/extra"),
    ("POST", "/v1/decide-extra"), ("POST", "/v1/Decide"),
    ("POST", "/v1//decide"), ("POST", "/v1/decide%2fextra"),
    ("POST", "/v1/%2564ecide"), ("POST", "/v1/decide%00"),
    ("GET", "/api/health/"), ("GET", "/api/health%2fextra"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", DENIED)
async def test_refused_routes_never_run_the_handler(node, method, path):
    response = await node.client.request(method, URL(path, encoded=True), headers=_headers())
    assert response.status == 403
    assert node.seen == []
    assert SCOPED_TOKEN not in await response.text()


@pytest.mark.asyncio
async def test_health_is_still_keyless_and_unknown_keys_are_not_operators(node):
    assert (await node.client.get("/api/health")).status == 200
    assert (await node.client.get("/api/health", headers=_headers("unknown"))).status == 200
    node.seen.clear()
    for headers in ({}, _headers("unknown")):
        assert (await node.client.post("/v1/decide", headers=headers)).status == 401
    assert node.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", BAD_SCOPES)
async def test_invalid_scope_is_live_and_never_falls_back_to_an_admin_cookie(node, scope):
    store = node.app["users_store"]
    store.add_user("example", "synthetic password", role="admin")
    cookie, _ = store.create_session("example")
    cfg = auth.AuthConfig.load()
    cfg.api_keys[0]["scope"] = scope
    cfg.save()
    headers = _headers(Cookie=f"{auth.SESSION_COOKIE}={cookie}", **{
        auth.CLIENT_HEADER: auth.DASHBOARD_CLIENT,
    })
    for method, path in [("POST", "/v1/decide"), ("GET", "/api/health"),
                         ("POST", "/api/auth/login"), ("GET", "/api/auth/me"),
                         ("GET", "/api/config")]:
        response = await node.client.request(method, path, headers=headers)
        assert response.status == 403
        assert SCOPED_TOKEN not in await response.text()
    assert node.seen == []
    assert not node.app["auth_config"].validate_token(SCOPED_TOKEN)


@pytest.mark.asyncio
async def test_scoped_key_wins_over_admin_cookie_but_stale_key_keeps_existing_fallback(node):
    store = node.app["users_store"]
    store.add_user("example", "synthetic password", role="admin")
    cookie, _ = store.create_session("example")
    extra = {"Cookie": f"{auth.SESSION_COOKIE}={cookie}",
             auth.CLIENT_HEADER: auth.DASHBOARD_CLIENT}
    response = await node.client.post("/v1/decide", headers=_headers(**extra))
    assert (await response.json())["admin"] is False
    node.seen.clear()
    for path in ("/api/config", "/api/auth/me", "/api/auth/login", "/api/auth/keys"):
        assert (await node.client.post(path, headers=_headers(**extra))).status == 403
    assert node.seen == []
    response = await node.client.get("/api/config", headers=_headers("stale", **extra))
    data = await response.json()
    assert data["admin"] is True and data["user"] == "example"


@pytest.mark.asyncio
@pytest.mark.parametrize("token,key_id", [(OPERATOR_TOKEN, "old-operator"),
                                           (fleet_key(SECRET), "fleet")])
async def test_old_operator_and_fleet_keep_their_authority(node, token, key_id):
    for method, path in DENIED:
        response = await node.client.request(method, URL(path, encoded=True), headers=_headers(token))
        if method == "OPTIONS":
            assert response.status == 204
            continue
        assert response.status == 200
        if method != "HEAD":
            data = await response.json()
            assert data["admin"] is data["host"] is True
            assert data["id"] == key_id
    assert len(node.seen) == sum(method != "OPTIONS" for method, _ in DENIED)


@pytest.mark.asyncio
async def test_scope_changes_and_revocation_are_live(node):
    headers = _headers(OPERATOR_TOKEN)
    assert (await node.client.get("/api/config", headers=headers)).status == 200
    cfg = auth.AuthConfig.load()
    cfg.api_keys[1]["scope"] = "decide"
    cfg.save()
    node.seen.clear()
    assert (await node.client.get("/api/config", headers=headers)).status == 403
    assert node.seen == []
    assert (await node.client.post("/v1/decide", headers=headers)).status == 200
    assert auth.AuthConfig.load().revoke_key("old-operator") is True
    node.seen.clear()
    assert (await node.client.post("/v1/decide", headers=headers)).status == 401
    assert node.seen == []
    assert (await node.client.get("/api/health")).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("damaged", ['{"enabled":', '{}', '[]',
    '{"enabled": false, "api_keys": {}}', '{"enabled": "false", "api_keys": []}',
    '{"enabled": false, "api_keys": [null]}', None])
async def test_damaged_or_missing_reload_preserves_the_scope_and_auth_wall(node, auth_home, damaged):
    path = auth_home / "auth.json"
    if damaged is None:
        path.unlink()
    else:
        path.write_text(damaged)
    assert (await node.client.get("/api/config", headers=_headers())).status == 403
    assert (await node.client.get("/api/config")).status == 401
    assert node.seen == []
    assert (await node.client.post("/v1/decide", headers=_headers())).status == 200


@pytest.mark.asyncio
async def test_scope_is_enforced_on_open_nodes_too(node):
    auth.AuthConfig.load().disable()
    assert (await node.client.get("/api/config", headers=_headers())).status == 403
    assert node.seen == []
    assert (await node.client.get("/api/config")).status == 200
    assert (await node.client.post("/v1/decide", headers=_headers())).status == 200


@pytest.mark.asyncio
async def test_forwarding_header_does_not_turn_a_scoped_key_into_fleet(node):
    headers = _headers(**{FORWARDED_BY_HEADER: "synthetic-entry",
                         "X-AINode-Launch-Authorized": "true"})
    response = await node.client.post("/v1/decide", headers=headers)
    data = await response.json()
    assert data["forwarded"] is data["admin"] is data["launch"] is False
    node.seen.clear()
    assert (await node.client.post("/v1/systemone", headers=headers)).status == 403
    assert node.seen == []
    response = await node.client.post("/v1/decide", headers=_headers(
        fleet_key(SECRET), **{FORWARDED_BY_HEADER: "synthetic-entry"}))
    assert (await response.json())["forwarded"] is True


def test_scope_is_persisted_beside_only_a_hash(auth_home):
    cfg = auth.AuthConfig.load()
    entry = cfg.generate_key("decision client", scope="decide")
    text = (auth_home / "auth.json").read_text()
    assert entry["key"] not in text
    loaded = auth.AuthConfig.load()
    assert loaded.api_keys[-1]["scope"] == "decide"
    assert loaded.identify_token_claims(entry["key"]) == (True, entry["id"], "decide")
    assert loaded.key_ids()[-1]["scope"] == "decide"
    assert entry["key"] not in json.dumps(loaded.key_ids())
    assert loaded.api_keys[-1]["key_hash"] not in json.dumps(loaded.key_ids())
    assert stat.S_IMODE((auth_home / "auth.json").stat().st_mode) == 0o600
    assert not (auth_home / "auth.json.tmp").exists()


@pytest.mark.parametrize("scope", BAD_SCOPES)
def test_generation_rejects_invalid_claims_before_minting(auth_home, scope):
    cfg = auth.AuthConfig.load()
    before = (auth_home / "auth.json").read_text()
    with pytest.raises(ValueError, match="Scope must be operator or decide"):
        cfg.generate_key("decision client", scope=scope)
    assert (auth_home / "auth.json").read_text() == before


def test_loopback_stubs_keep_the_existing_open_node_authority():
    request = SimpleNamespace(app={}, remote="127.0.0.1", headers={})
    assert may_control_host(request) is True
    assert may_supply_launch_fields(request) is True
    request.remote = "192.0.2.10"
    assert may_control_host(request) is False
    assert may_supply_launch_fields(request) is False


def test_invalid_scope_values_are_never_reflected_in_key_lists(auth_home):
    cfg = auth.AuthConfig.load()
    cfg.api_keys[0]["scope"] = "synthetic-untrusted-scope-value"
    public = json.dumps(cfg.key_ids())
    assert "synthetic-untrusted-scope-value" not in public
    assert cfg.key_ids()[0]["scope"] == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/%64ecide", "/%76%31/decide"])
async def test_scope_uses_the_normalized_path(node, path):
    response = await node.client.post(URL(path, encoded=True), headers=_headers())
    assert response.status == 200
    assert node.seen == [("POST", "/v1/decide")]


def _cli(*argv):
    from ainode.cli.main import main
    with patch.object(sys, "argv", ["ainode", *argv]):
        main()


def test_cli_creates_lists_and_revokes_a_named_decide_key(auth_home, capsys):
    _cli("auth", "key", "create", "--name", "decision-client", "--scope", "decide")
    output = capsys.readouterr().out
    assert "decide" in output and "decision-client" in output
    assert "POST /v1/decide" in output and "GET /api/health" in output
    assert "Dashboard: paste" not in output
    cfg = auth.AuthConfig.load()
    entry = cfg.api_keys[-1]
    assert entry["scope"] == "decide"
    plaintext = output.split("API key: ", 1)[1].split()[0]
    assert cfg.validate_token(plaintext)
    _cli("auth", "key", "list")
    output = capsys.readouterr().out
    assert "decide" in output and "operator" in output
    assert plaintext not in output and entry["key_hash"] not in output
    _cli("auth", "key", "revoke", entry["id"])
    output = capsys.readouterr().out
    assert plaintext not in output and entry["key_hash"] not in output
    assert not auth.AuthConfig.load().validate_token(plaintext)


def test_cli_default_is_operator_and_unknown_scope_mints_nothing(auth_home, capsys):
    _cli("auth", "key", "create", "--name", "operator-client")
    capsys.readouterr()
    assert auth.AuthConfig.load().api_keys[-1]["scope"] == "operator"
    before = (auth_home / "auth.json").read_text()
    with pytest.raises(SystemExit) as exc:
        _cli("auth", "key", "create", "--scope", "unknown")
    assert exc.value.code == 2
    assert (auth_home / "auth.json").read_text() == before


@pytest.mark.asyncio
async def test_real_management_handlers_never_mint_or_log_in_with_a_scoped_key(auth_home):
    app = web.Application(middlewares=[auth.auth_middleware])
    app["auth_config"] = auth.AuthConfig.load()
    app["users_store"] = UsersStore.load()
    app["users_store"].add_user("example", "synthetic password", role="admin")
    register_auth_routes(app)
    register_session_routes(app)
    async with TestClient(TestServer(app)) as client:
        before = (auth_home / "auth.json").read_text()
        for path, body in [("/api/auth/keys", {"scope": "operator"}),
                           ("/api/auth/login", {"name": "example", "password": "synthetic password"})]:
            response = await client.post(path, headers=_headers(), json=body)
            assert response.status == 403 and not response.cookies
        assert (auth_home / "auth.json").read_text() == before
        assert app["users_store"].sessions == []
        for scope in BAD_SCOPES:
            response = await client.post("/api/auth/keys", headers=_headers(OPERATOR_TOKEN),
                                         json={"scope": scope})
            assert response.status == 400
            assert (auth_home / "auth.json").read_text() == before
        response = await client.post("/api/auth/keys", headers=_headers(OPERATOR_TOKEN),
                                     json={"name": "example", "scope": "decide"})
        assert response.status == 200
        assert auth.AuthConfig.load().api_keys[-1]["scope"] == "decide"


@pytest.mark.asyncio
async def test_production_decide_handler_forwards_with_fleet_without_widening_scope(auth_home, monkeypatch):
    from ainode.api import decide, server

    owner_seen = []

    async def owner_handler(request):
        owner_seen.append((request.path, request["api_key_id"], await request.json()))
        assert is_forwarded_by_fleet(request)
        return web.json_response({"model": "synthetic-model", "decisions": {}})

    owner = web.Application(middlewares=[auth.auth_middleware])
    owner["config"] = NodeConfig(cluster_secret=SECRET)
    owner["auth_config"] = auth.AuthConfig(enabled=True)
    owner.router.add_post("/v1/decide", owner_handler)
    called = []
    real_handler = decide.handle_decide

    async def record(request):
        called.append(request.path)
        return await real_handler(request)

    monkeypatch.setattr(server, "handle_decide", record)
    monkeypatch.setattr(decide, "candidates_for", lambda *a: [("127.0.0.1", 9999)])
    async with TestServer(owner) as owner_server:
        monkeypatch.setattr(decide, "owner_web_ports", lambda *a: [("127.0.0.1", owner_server.port)])
        app = server.create_app(NodeConfig(node_id="entry", cluster_secret=SECRET,
                                           onboarded=True))
        async with TestClient(TestServer(app)) as client:
            body = {"model": "synthetic-model", "state": {"message": "synthetic"},
                    "questions": {"ok": {"question": "Proceed?", "type": "boolean"}}}
            response = await client.post("/v1/decide", headers=_headers(), json=body)
            assert response.status == 200
            assert called == ["/v1/decide"]
            assert owner_seen == [("/v1/decide", "fleet", body)]
            response = await client.post("/v1/systemone", headers=_headers(), json=body)
            assert response.status == 403
            assert called == ["/v1/decide"] and len(owner_seen) == 1
            assert (await client.get("/api/health")).status == 200


@pytest.mark.asyncio
async def test_production_preflight_checks_scope_and_preserves_cors(auth_home):
    from ainode.api.server import create_app

    app = create_app(NodeConfig(node_id="test-node", cluster_secret=SECRET, onboarded=True))
    seen = []

    async def handler(request):
        seen.append(request.path)
        return web.Response(status=200)

    app.router.add_route("OPTIONS", "/v1/decide", handler)
    app.router.add_route("OPTIONS", "/api/health", handler)
    async with TestClient(TestServer(app)) as client:
        # This is what a browser really sends, without a Bearer credential.
        preflight = {"Origin": "http://localhost:3000",
                     "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "authorization,content-type"}
        for path in ("/v1/decide", "/api/health"):
            response = await client.options(path, headers=preflight)
            assert response.status == 204
            cors = {key: value for key, value in response.headers.items()
                    if key.startswith("Access-Control-")}
            assert cors["Access-Control-Allow-Origin"] == preflight["Origin"]
            assert "Authorization" in cors["Access-Control-Allow-Headers"]
            for token in (OPERATOR_TOKEN, fleet_key(SECRET)):
                response = await client.options(path, headers={**preflight, **_headers(token)})
                assert response.status == 204
                assert all(response.headers[key] == value for key, value in cors.items())
            response = await client.options(path, headers={**preflight, **_headers()})
            assert response.status == 403
            assert all(response.headers[key] == value for key, value in cors.items())
            response = await client.options(path, headers={**preflight, **_headers(),
                                                            "Origin": "https://untrusted.example"})
            assert response.status == 403
            assert response.headers["Access-Control-Allow-Origin"] != "https://untrusted.example"
        assert seen == []
