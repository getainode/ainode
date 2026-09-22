"""With auth on, every route needs the key except the few that cannot.

The rule this file pins (#168): when auth is enabled, every path under ``/api``
and ``/v1`` requires a credential, except ``/api/health``, ``/api/auth/status``,
``/api/auth/login``, ``/api/auth/me``, ``/api/cluster/endpoint``,
``POST /api/cluster/join`` and the static shell. It is checked against the REAL
route table (``create_app``), not a list of paths typed out here, because a list
is what drifts: the next mutating route somebody adds is covered by the rule and
by this test on the day it is registered.

"A credential" is three things since #261: an operator API key, the fleet key, or
a login session cookie. This file drives the first; ``tests/test_session_routes.py``
drives the third, including the CSRF header a cookie-authenticated write needs.

The second half pins the ``trust_remote_code`` rule, which holds even when auth
is switched off: setting it needs a key, or a curated catalog entry that already
declares it.
"""

import socket

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import (
    AUTHENTICATED_ONLY_CONFIG_FIELDS,
    auth_status_fields,
    create_app,
)
from ainode.core.config import NodeConfig
from ainode.models.api_routes import catalog_recipe, parse_launch_overrides


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    """Every file this app writes goes into tmp_path, not the operator's ~/.ainode.

    The app writes auth.json, config.json, instances.json and launch-times.json
    just by starting, and this file drives real handlers. Most paths resolve
    AINODE_HOME at call time, which the first patch covers; AUTH_FILE,
    CONFIG_FILE and SECRETS_FILE are computed at import and need their own.
    """
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.secrets.manager.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.secrets.manager.SECRETS_FILE", tmp_path / "secrets.json")
    return tmp_path


@pytest.fixture
def config():
    # A port nothing listens on: the vllm proxy falls back to localhost:<api_port>
    # and a real service there would answer instead of failing the way we expect.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    return NodeConfig(node_id="auth-node", node_name="AuthNode",
                      api_port=free_port, onboarded=True)


@pytest.fixture
def app(config, auth_home):
    return create_app(config=config, engine=None)


@pytest_asyncio.fixture
async def open_client(app):
    """Auth off: today's default."""
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest_asyncio.fixture
async def keyed(app):
    """Auth on, plus the plaintext key, which exists only at this moment."""
    entry = app["auth_config"].enable()
    async with TestClient(TestServer(app)) as c:
        yield c, entry["key"]


# =============================================================================
# The rule, walked over the real route table
# =============================================================================

# The only paths that answer without a key when auth is on, and why:
#   /api/health        a liveness probe has no key
#   /api/auth/status   so the dashboard can say a key is wanted, not render blank
#   /                  the static shell, which is what asks for the key
#   /api/cluster/endpoint  addresses only: a client whose node is down has to be
#                      able to ask a reachable node where the fleet is
#                      (tests/test_failover_endpoint.py holds it to addresses)
#   /api/cluster/join  a node joining this cluster does not hold this cluster's
#                      key yet, so a single-use expiring join token is the
#                      credential and the handler rate limits per source IP
#                      (api/cluster_join.py). Covered by tests/test_join_flow.py.
#   /api/auth/login    the door: the caller with no credential is exactly who
#                      posts to it. It has its own per-address throttle on FAILED
#                      attempts instead (tests/test_session_routes.py).
#   /api/auth/me       answers {"user": null} to a caller it does not recognise,
#                      so the dashboard can draw the login page instead of
#                      guessing. It tells a stranger nothing else.
OPEN_WITH_AUTH_ON = {"/", "/api/health", "/api/auth/status",
                     "/api/auth/login", "/api/auth/me",
                     "/api/cluster/endpoint", "/api/cluster/join"}

# Routes whose path carries a variable. Filled in with something harmless: the
# request must be refused before the handler ever looks at it.
PATH_PARAMS = {
    "key_id": "some-key",
    "model_id": "some-model",
    "job_id": "some-job",
    "run_id": "some-run",
    "dataset_id": "some-dataset",
    "filename": "out.bin",
    "name": "some-name",
    "key": "HF_TOKEN",
    "id": "some-id",
}


def _concrete_paths(app):
    """Every registered (method, path) with the variables filled in."""
    out = []
    for route in app.router.routes():
        info = route.resource.get_info() if route.resource else {}
        path = info.get("path") or info.get("formatter") or ""
        if not path or info.get("prefix"):
            continue  # static route: served from disk, not a handler
        if route.method in ("OPTIONS", "HEAD"):
            continue
        for param, value in PATH_PARAMS.items():
            path = path.replace("{" + param + "}", value)
        # {model_id:.+} and friends keep their regex in the formatter.
        for param, value in PATH_PARAMS.items():
            path = path.replace("{" + param + ":.+}", value)
        if "{" in path:
            continue  # a pattern this test does not know how to fill
        out.append((route.method, path))
    return sorted(set(out))


@pytest.mark.asyncio
async def test_every_route_needs_the_key_when_auth_is_on(keyed):
    """No route answers without the key, bar the three that cannot want one."""
    client, _key = keyed
    checked = 0
    for method, path in _concrete_paths(client.app):
        if path in OPEN_WITH_AUTH_ON:
            continue
        resp = await client.request(method, path)
        assert resp.status == 401, f"{method} {path} answered {resp.status} with no key"
        body = await resp.json()
        assert body["error"]["type"] == "auth_error"
        checked += 1
    # A route table that suddenly has nothing in it would pass every assert above.
    assert checked > 40, f"only {checked} routes checked: is the route table wired?"


# Every mutating route #168 named, by name rather than by walking the table, so
# the list this file is read against is the list in the issue.
MUTATING_ROUTES = [
    ("POST", "/api/models/load", {"model": "x"}),
    ("POST", "/api/models/unload", {}),
    ("POST", "/api/models/download-repo", {"repo": "x"}),
    ("POST", "/api/models/delete-repo", {"repo": "x"}),
    ("POST", "/api/models/some-model/download", None),
    ("DELETE", "/api/models/some-model", None),
    ("PATCH", "/api/config", {"node_name": "renamed"}),
    ("POST", "/api/cluster/load", {"model": "x"}),
    ("POST", "/api/cluster/unload", {}),
    ("POST", "/api/cluster/role", {"role": "auto"}),
    ("POST", "/api/cluster/id", {"cluster_id": "other"}),
    ("POST", "/api/cluster/update-all", {}),
    ("POST", "/api/sharding/launch", {"model": "x"}),
    ("POST", "/api/engine/set-model", {"model": "x"}),
    ("POST", "/api/engine/update", {}),
    ("POST", "/api/training/jobs", {}),
    ("DELETE", "/api/training/jobs/some-job", None),
    ("POST", "/api/datasets", {}),
    ("POST", "/api/datasets/upload", None),
    ("DELETE", "/api/datasets/some-dataset", None),
    ("PUT", "/api/secrets/HF_TOKEN", {"value": "x"}),
    ("DELETE", "/api/secrets/HF_TOKEN", None),
    ("GET", "/api/secrets", None),
    ("POST", "/api/embeddings/models/some-model/load", {}),
    ("POST", "/api/bench/runs", {}),
    ("DELETE", "/api/bench/runs/some-run", None),
    ("POST", "/api/server/models/some-model/eject", None),
    ("DELETE", "/api/server/logs", None),
    ("POST", "/api/auth/keys", None),
    ("POST", "/api/auth/disable", None),
]

# Routes whose handler is cheap and local, so the "the key is accepted" half can
# actually call them. The rest are checked for the 401 only: they pull images,
# start engines, restart the fleet or delete weights, and a unit test must not
# do any of that on the machine it runs on. The middleware is one function with
# no per-route logic, so the key being accepted here is the key being accepted
# everywhere; the fleet proof in the PR exercises unload end to end.
KEY_ACCEPTED_ROUTES = {
    ("POST", "/api/models/unload"),
    ("PATCH", "/api/config"),
    ("POST", "/api/cluster/role"),
    ("POST", "/api/cluster/id"),
    ("POST", "/api/training/jobs"),
    ("POST", "/api/datasets"),
    ("GET", "/api/secrets"),
    ("POST", "/api/bench/runs"),
    ("DELETE", "/api/server/logs"),
    ("POST", "/api/auth/keys"),
}


@pytest.mark.asyncio
async def test_mutating_routes_refuse_a_request_with_no_key(keyed):
    client, _key = keyed
    for method, path, body in MUTATING_ROUTES:
        kwargs = {"json": body} if body is not None else {}
        resp = await client.request(method, path, **kwargs)
        assert resp.status == 401, f"{method} {path} answered {resp.status} with no key"


@pytest.mark.asyncio
async def test_mutating_routes_accept_the_key(keyed):
    """The key gets the request past the middleware and into the handler.

    What the handler then says (400, 404, 409, 503 on a node with no engine) is
    its own business; a 401 would mean the key was not accepted.
    """
    client, key = keyed
    for method, path, body in MUTATING_ROUTES:
        if (method, path) not in KEY_ACCEPTED_ROUTES:
            continue
        kwargs = {"json": body} if body is not None else {}
        resp = await client.request(method, path,
                                   headers={"Authorization": f"Bearer {key}"}, **kwargs)
        assert resp.status != 401, f"{method} {path} refused a valid key"


@pytest.mark.asyncio
async def test_the_shell_still_loads_with_auth_on(keyed):
    """The dashboard's own HTML, its assets and the two open API routes."""
    client, _ = keyed
    resp = await client.get("/")
    assert resp.status == 200
    assert "AINode" in await resp.text()

    resp = await client.get("/static/js/auth.js")
    assert resp.status == 200

    resp = await client.get("/api/health")
    assert resp.status == 200

    resp = await client.get("/api/auth/status")
    assert resp.status == 200
    status = await resp.json()
    assert status["enabled"] is True
    # No key on this request, so the UI knows to ask for one.
    assert status["authenticated"] is False


@pytest.mark.asyncio
async def test_auth_status_reports_a_good_key_as_authenticated(keyed):
    client, key = keyed
    resp = await client.get("/api/auth/status",
                            headers={"Authorization": f"Bearer {key}"})
    status = await resp.json()
    assert status["authenticated"] is True


# =============================================================================
# trust_remote_code
# =============================================================================

@pytest.mark.asyncio
async def test_patch_config_refuses_trust_remote_code_unauthenticated(open_client):
    """Auth off, so no request is authenticated: the field cannot be set."""
    resp = await open_client.patch("/api/config", json={"trust_remote_code": True})
    assert resp.status == 200
    data = await resp.json()
    assert "trust_remote_code" in data["rejected"]
    assert "trust_remote_code" not in data["applied"]
    reason = data["rejected_reasons"]["trust_remote_code"]
    # The rule is in the error text, not only in a comment in the source.
    assert "API key" in reason
    assert "curated catalog" in reason
    assert open_client.app["config"].trust_remote_code is False


@pytest.mark.asyncio
async def test_patch_config_still_applies_the_other_fields(open_client):
    """One refused field must not take the rest of the patch down with it."""
    resp = await open_client.patch(
        "/api/config", json={"trust_remote_code": True, "node_name": "Renamed"})
    data = await resp.json()
    assert data["applied"]["node_name"] == "Renamed"
    assert data["rejected"] == ["trust_remote_code"]


@pytest.mark.asyncio
async def test_patch_config_allows_trust_remote_code_with_a_key(keyed):
    client, key = keyed
    resp = await client.patch("/api/config", json={"trust_remote_code": True},
                             headers={"Authorization": f"Bearer {key}"})
    assert resp.status == 200
    data = await resp.json()
    assert data["applied"]["trust_remote_code"] is True
    assert data["rejected"] == []


@pytest.mark.asyncio
async def test_clearing_trust_remote_code_never_needs_a_key(open_client):
    """A de-escalation stays open, so a client can always put the node back."""
    open_client.app["config"].trust_remote_code = True
    resp = await open_client.patch("/api/config", json={"trust_remote_code": False})
    data = await resp.json()
    assert data["applied"]["trust_remote_code"] is False
    assert open_client.app["config"].trust_remote_code is False


def test_trust_remote_code_is_the_whole_authenticated_only_list():
    assert AUTHENTICATED_ONLY_CONFIG_FIELDS == {"trust_remote_code"}


# -- the load path -------------------------------------------------------------

def _curated_trc_model() -> str:
    """A catalog id whose recipe declares trust_remote_code, or skip."""
    from ainode.models.registry import CURATED_CLUSTER_MODELS
    for info in CURATED_CLUSTER_MODELS.values():
        if getattr(info, "trust_remote_code", False):
            return info.id
    pytest.skip("no curated entry declares trust_remote_code")


def test_load_refuses_trust_remote_code_on_an_arbitrary_repo():
    overrides, err = parse_launch_overrides(
        {"trust_remote_code": True}, authenticated=False, model="attacker/evil-repo")
    assert overrides == {}
    assert err and "API key" in err
    assert "attacker/evil-repo" in err


def test_load_allows_trust_remote_code_for_a_curated_entry():
    model = _curated_trc_model()
    assert catalog_recipe(model)["trust_remote_code"] is True
    overrides, err = parse_launch_overrides(
        {"trust_remote_code": True}, authenticated=False, model=model)
    assert err is None
    assert overrides["trust_remote_code"] is True


def test_load_allows_trust_remote_code_for_an_authenticated_caller():
    overrides, err = parse_launch_overrides(
        {"trust_remote_code": True}, authenticated=True, model="someone/their-repo")
    assert err is None
    assert overrides["trust_remote_code"] is True


def test_load_ignores_a_falsy_trust_remote_code_from_anyone():
    overrides, err = parse_launch_overrides(
        {"trust_remote_code": False}, authenticated=False, model="attacker/evil-repo")
    assert err is None
    assert overrides["trust_remote_code"] is False


@pytest.mark.asyncio
async def test_load_route_answers_400_with_the_rule(open_client):
    """End to end: the refusal a curl user actually sees on an open node.

    Refused in parse_launch_overrides, which runs before anything is launched, so
    this does not start an engine.
    """
    resp = await open_client.post(
        "/api/models/load", json={"model": "attacker/evil-repo", "trust_remote_code": True})
    assert resp.status == 400
    data = await resp.json()
    assert "API key" in data["error"]
    assert "curated catalog" in data["error"]


# =============================================================================
# The truthful default
# =============================================================================

@pytest.mark.asyncio
async def test_status_says_the_api_is_open(open_client):
    resp = await open_client.get("/api/status")
    data = await resp.json()
    assert data["auth"] == {"enabled": False, "key_count": 0,
                            "label": "API open, no key set"}


@pytest.mark.asyncio
async def test_status_says_a_key_is_required(keyed):
    client, key = keyed
    resp = await client.get("/api/status", headers={"Authorization": f"Bearer {key}"})
    data = await resp.json()
    assert data["auth"]["enabled"] is True
    assert data["auth"]["key_count"] == 1
    assert data["auth"]["label"] == "API key required"


def test_auth_status_fields_reports_a_key_that_is_not_required(auth_home):
    from ainode.auth.middleware import AuthConfig
    cfg = AuthConfig()
    cfg.generate_key()
    assert auth_status_fields({"auth_config": cfg})["label"] == (
        "API open, key set but not required")


def test_auth_status_fields_survives_a_node_with_no_auth_config():
    assert auth_status_fields({})["label"] == "API open, no key set"


def test_the_installer_prints_the_same_words():
    """One fact, one wording: the banner and /api/status must not disagree."""
    from pathlib import Path
    script = (Path(__file__).parent.parent / "scripts" / "install.sh").read_text()
    assert "API open, no key set" in script
    assert auth_status_fields({})["label"] in script
