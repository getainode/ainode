"""The login routes, driven through the real middleware (#261).

Same shape as ``tests/test_auth.py``: a small aiohttp app carrying the auth
middleware, the real auth and session routes, and a couple of protected routes to
aim a credential at. What it pins:

* **The door.** A good login sets one cookie with exactly the attributes the
  browser needs, a bad one says "Wrong name or password" whether or not the name
  exists, a node with no accounts says how to make the first admin, and an address
  that keeps guessing is throttled.
* **The cookie is a third credential**, equal to a key on a GET and refused on a
  write that does not carry ``X-AINode-Client: dashboard``. That header is the
  whole CSRF defence, so it is checked in both directions.
* **A Bearer token still wins.** The dashboard, the bench and the desktop app
  keep working exactly as they did.
* **Roles mean something.** A member session is refused by every account route,
  by the key routes and by the auth switch; an admin session, an operator key and
  the fleet key are not. The last enabled admin cannot be removed or disabled.
* **Replication is the fleet's alone.** Export and sync answer 403 to an admin
  and to an operator key, because they move password hashes and the credential for
  that is membership of the cluster.
* **The keyless list is exactly two longer.** ``SKIP_PATHS`` is asserted here as a
  literal set, because it is the one list in the codebase where a typo is an open
  door.
"""

import asyncio
import time

import pytest
import pytest_asyncio
from aiohttp import DummyCookieJar, web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth.accounts import ROLE_ADMIN, ROLE_MEMBER, UsersStore
from ainode.auth.api_routes import register_auth_routes
from ainode.auth.fleet import fleet_key
from ainode.auth.middleware import (
    CLIENT_HEADER,
    MISSING_KEY_MESSAGE,
    SESSION_COOKIE,
    SKIP_PATHS,
    AuthConfig,
    auth_middleware,
)
from ainode.auth.session_routes import (
    COOKIE_MAX_AGE,
    LOGIN_RATE_LIMIT,
    LOGIN_RATE_STATE_KEY,
    LOGIN_RATE_WINDOW,
    NO_ACCOUNTS_MESSAGE,
    clear_login_failures,
    login_locked_out,
    record_login_failure,
    register_session_routes,
    session_cookie,
)
from ainode.core.config import NodeConfig


SECRET = "0123456789abcdef0123456789abcdef"
ADMIN_PASSWORD = "hunter2hunter2"
MEMBER_PASSWORD = "correct horse battery"
DASH = {CLIENT_HEADER: "dashboard"}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def stores(tmp_path, monkeypatch):
    """A fresh ``auth.json`` and ``users.json`` in a temp home."""
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.accounts.USERS_FILE", tmp_path / "users.json")
    users = UsersStore()
    users.add_user("jason", ADMIN_PASSWORD, ROLE_ADMIN)
    users.add_user("sam", MEMBER_PASSWORD, ROLE_MEMBER)
    return AuthConfig(), users


async def _whoami(request):
    """Echoes what the middleware stamped, which is the contract under test."""
    return web.json_response({
        "authenticated": request["authenticated"],
        "api_key_id": request["api_key_id"],
        "user": request["user"],
        "user_role": request["user_role"],
    })


async def _ok(_request):
    return web.json_response({"ok": True})


def _make_app(auth_config, users, secret=SECRET):
    app = web.Application(middlewares=[auth_middleware])
    app["config"] = NodeConfig(node_id="login-node", node_name="Login",
                               cluster_secret=secret, onboarded=True)
    app["auth_config"] = auth_config
    app["users_store"] = users
    app.router.add_get("/api/whoami", _whoami)
    app.router.add_post("/api/whoami", _whoami)
    app.router.add_post("/api/models/unload", _ok)
    register_auth_routes(app)
    register_session_routes(app)
    return app


def _client(app):
    """A client with NO cookie jar, so every cookie in this file is explicit.

    A jar would re-send whichever session logged in last, which is exactly what a
    browser does and exactly what makes a multi-account test lie: "the member
    cannot see the admin's sessions" passes for the wrong reason when the jar
    quietly upgraded the request to the admin's cookie. Every test here names the
    session it is using in a ``Cookie`` header instead.
    """
    return TestClient(TestServer(app), cookie_jar=DummyCookieJar())


@pytest_asyncio.fixture
async def open_client(stores):
    """Auth off, which is still the default on an installed node before 0.5.30."""
    auth_config, users = stores
    async with _client(_make_app(auth_config, users)) as client:
        yield client


@pytest_asyncio.fixture
async def protected(stores):
    """Auth on, with one operator key whose plaintext exists only here."""
    auth_config, users = stores
    entry = auth_config.enable("test")
    async with _client(_make_app(auth_config, users)) as client:
        yield client, entry["key"]


async def _login(client, name, password, **kwargs):
    return await client.post("/api/auth/login",
                             json={"name": name, "password": password}, **kwargs)


async def _cookie_of(client, name, password):
    """Log in and hand back the raw token, for a test that sets Cookie itself."""
    resp = await _login(client, name, password)
    assert resp.status == 200, await resp.text()
    header = resp.headers["Set-Cookie"]
    return header.split(";")[0].split("=", 1)[1]


def _jar(token):
    return {"Cookie": f"{SESSION_COOKIE}={token}"}


# =============================================================================
# SKIP_PATHS
# =============================================================================

def test_the_keyless_list_is_exactly_these_seven():
    """The one list where a typo is an open door, so it is typed out in full."""
    assert SKIP_PATHS == {
        "/",
        "/api/health",
        "/api/auth/status",
        "/api/auth/login",
        "/api/auth/me",
        "/api/cluster/endpoint",
        "/api/cluster/join",
    }, ("SKIP_PATHS changed. Every path here answers with NO credential on a node "
        "with auth on, so a new entry needs a reason in the middleware docstring, "
        "a sentence in the dashboard's API access panel "
        "(tests/test_fleet_auth.py::test_the_dashboard_panel_lists_every_keyless_path) "
        "and a line in tests/test_auth_gate.py::OPEN_WITH_AUTH_ON.")


def test_logout_is_not_keyless():
    """Deliberate: logout revokes something, so it is a write like any other.

    A caller with no credential at all has no session to end, and on a node with
    auth on it gets the middleware's 401. A dashboard reads that as "already
    logged out" and clears its own state.
    """
    assert "/api/auth/logout" not in SKIP_PATHS


# =============================================================================
# The door
# =============================================================================

class TestLogin:
    @pytest.mark.asyncio
    async def test_a_good_login_answers_the_account_and_one_cookie(self, protected):
        client, _ = protected
        resp = await _login(client, "jason", ADMIN_PASSWORD)
        assert resp.status == 200
        body = await resp.json()
        assert body["user"] == {"name": "jason", "role": ROLE_ADMIN}
        assert body["session_id"]

        cookie = resp.headers["Set-Cookie"]
        assert cookie.startswith(f"{SESSION_COOKIE}=")
        assert "; Path=/" in cookie
        assert "; HttpOnly" in cookie
        assert "; SameSite=Lax" in cookie
        assert f"; Max-Age={COOKIE_MAX_AGE}" in cookie
        assert "Secure" not in cookie, "a plain-HTTP node must not set Secure"
        token = cookie.split(";")[0].split("=", 1)[1]
        assert token not in await resp.text(), "the token is in the body too"

    @pytest.mark.asyncio
    async def test_the_name_is_case_folded(self, protected):
        client, _ = protected
        resp = await _login(client, "  JASON ", ADMIN_PASSWORD)
        assert resp.status == 200
        assert (await resp.json())["user"]["name"] == "jason"

    @pytest.mark.asyncio
    async def test_https_adds_secure(self, protected):
        """Behind a terminating proxy the node sees HTTP plus the header."""
        client, _ = protected
        resp = await client.post("/api/auth/login",
                                 json={"name": "jason", "password": ADMIN_PASSWORD},
                                 headers={"X-Forwarded-Proto": "https"})
        assert "; Secure" in resp.headers["Set-Cookie"]

    @pytest.mark.asyncio
    async def test_a_wrong_password_and_an_unknown_name_answer_the_same(self,
                                                                       protected):
        client, _ = protected
        wrong = await _login(client, "jason", "not the password")
        missing = await _login(client, "nobody", ADMIN_PASSWORD)
        assert wrong.status == missing.status == 401
        assert (await wrong.json()) == (await missing.json())
        assert (await wrong.json())["error"] == {"message": "Wrong name or password",
                                                 "type": "auth_error"}

    @pytest.mark.asyncio
    async def test_a_body_with_nothing_in_it_is_a_400(self, protected):
        client, _ = protected
        resp = await client.post("/api/auth/login", json={})
        assert resp.status == 400
        assert "name and a password" in (await resp.json())["error"]["message"]

    @pytest.mark.asyncio
    async def test_a_disabled_account_cannot_log_in(self, protected):
        client, _ = protected
        client.app["users_store"].set_disabled("sam", True)
        resp = await _login(client, "sam", MEMBER_PASSWORD)
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_a_node_with_no_accounts_says_how_to_make_the_first(self, stores):
        auth_config, users = stores
        users.import_users([])
        auth_config.enable("test")
        async with _client(_make_app(auth_config, users)) as client:
            resp = await _login(client, "jason", ADMIN_PASSWORD)
            assert resp.status == 409
            body = await resp.json()
            assert body["error"]["message"] == NO_ACCOUNTS_MESSAGE
            assert "ainode auth user add" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_login_is_open_with_no_credential_on_a_protected_node(self,
                                                                       protected):
        """The whole point of the exemption: no key, and the door still opens."""
        client, _ = protected
        resp = await client.get("/api/whoami")
        assert resp.status == 401
        assert (await _login(client, "jason", ADMIN_PASSWORD)).status == 200

    @pytest.mark.asyncio
    async def test_the_cli_can_name_itself(self, protected):
        client, _ = protected
        resp = await client.post("/api/auth/login",
                                 json={"name": "jason", "password": ADMIN_PASSWORD,
                                       "client": "cli"})
        assert resp.status == 200
        sessions = client.app["users_store"].sessions_for("jason")
        assert [s["client"] for s in sessions] == ["cli"]


class TestThrottle:
    def test_the_counter_only_counts_failures(self):
        state, now = {}, 1000.0
        assert login_locked_out(state, "1.2.3.4", now) is False
        for n in range(LOGIN_RATE_LIMIT):
            assert login_locked_out(state, "1.2.3.4", now) is False
            record_login_failure(state, "1.2.3.4", now)
        assert login_locked_out(state, "1.2.3.4", now) is True
        assert login_locked_out(state, "5.6.7.8", now) is False, "one address, one bucket"

    def test_the_window_expires(self):
        state = {}
        for _ in range(LOGIN_RATE_LIMIT):
            record_login_failure(state, "1.2.3.4", 1000.0)
        assert login_locked_out(state, "1.2.3.4", 1000.0) is True
        assert login_locked_out(state, "1.2.3.4", 1000.0 + LOGIN_RATE_WINDOW) is False
        assert "1.2.3.4" not in state, "the log did not trim itself"

    def test_a_success_clears_the_slate(self):
        state = {}
        record_login_failure(state, "1.2.3.4", 1000.0)
        clear_login_failures(state, "1.2.3.4")
        assert state == {}

    def test_the_log_does_not_grow_without_bound(self):
        state = {}
        for n in range(40):
            record_login_failure(state, f"10.0.0.{n}", 1000.0 + n, max_sources=8)
        assert len(state) <= 8

    @pytest.mark.asyncio
    async def test_a_guessing_address_gets_429_with_retry_after(self, protected):
        client, _ = protected
        resp = await _login(client, "jason", "wrong")
        assert resp.status == 401
        state = client.app[LOGIN_RATE_STATE_KEY]
        source = next(iter(state))
        assert len(state[source]) == 1, "a failure was not recorded"

        state[source] = [time.time()] * LOGIN_RATE_LIMIT
        resp = await _login(client, "jason", ADMIN_PASSWORD)
        assert resp.status == 429
        assert resp.headers["Retry-After"] == str(int(LOGIN_RATE_WINDOW))
        assert (await resp.json())["error"]["type"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_a_good_login_clears_the_failures(self, protected):
        client, _ = protected
        await _login(client, "jason", "wrong")
        state = client.app[LOGIN_RATE_STATE_KEY]
        assert state
        assert (await _login(client, "jason", ADMIN_PASSWORD)).status == 200
        assert state == {}

    @pytest.mark.asyncio
    async def test_a_node_with_no_accounts_does_not_count_against_the_address(
            self, stores):
        auth_config, users = stores
        users.import_users([])
        async with _client(_make_app(auth_config, users)) as client:
            for _ in range(LOGIN_RATE_LIMIT + 2):
                assert (await _login(client, "jason", ADMIN_PASSWORD)).status == 409
            assert client.app[LOGIN_RATE_STATE_KEY] == {}


# =============================================================================
# The cookie as a credential
# =============================================================================

class TestCookieAuth:
    @pytest.mark.asyncio
    async def test_a_session_gets_a_get_past_the_wall(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.get("/api/whoami", headers=_jar(token))
        assert resp.status == 200
        assert await resp.json() == {"authenticated": True,
                                     "api_key_id": "user:jason",
                                     "user": "jason", "user_role": ROLE_ADMIN}

    @pytest.mark.asyncio
    async def test_a_write_needs_the_client_header(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)

        resp = await client.post("/api/whoami", headers=_jar(token))
        assert resp.status == 401
        message = (await resp.json())["error"]["message"]
        assert CLIENT_HEADER in message, "the 401 does not name the missing header"
        assert "Wrong name or password" not in message

        resp = await client.post("/api/whoami", headers={**_jar(token), **DASH})
        assert resp.status == 200
        assert (await resp.json())["user"] == "jason"

    @pytest.mark.asyncio
    async def test_a_wrong_client_header_is_no_header(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.post("/api/whoami",
                                 headers={**_jar(token), CLIENT_HEADER: "someone"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_the_header_is_case_insensitive_in_its_value(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.post("/api/whoami",
                                 headers={**_jar(token), CLIENT_HEADER: "Dashboard"})
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_a_stale_cookie_is_simply_unauthenticated(self, protected):
        client, _ = protected
        resp = await client.get("/api/whoami", headers=_jar("not-a-token"))
        assert resp.status == 401
        assert (await resp.json())["error"]["message"] == MISSING_KEY_MESSAGE

    @pytest.mark.asyncio
    async def test_a_revoked_session_stops_working_at_once(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 200
        client.app["users_store"].revoke_sessions_for("jason")
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 401

    @pytest.mark.asyncio
    async def test_disabling_an_account_ends_its_session(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 200
        client.app["users_store"].set_disabled("sam", True)
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 401

    @pytest.mark.asyncio
    async def test_a_bearer_key_wins_over_a_cookie(self, protected):
        client, key = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.get("/api/whoami",
                                headers={**_jar(token),
                                         "Authorization": f"Bearer {key}"})
        body = await resp.json()
        assert body["user"] == "", "the cookie overrode the key"
        assert body["api_key_id"] not in ("", "user:jason")

    @pytest.mark.asyncio
    async def test_a_stale_key_does_not_cost_a_good_session_its_request(self,
                                                                       protected):
        """A browser that logged in with an old key still in localStorage."""
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.get("/api/whoami",
                                headers={**_jar(token),
                                         "Authorization": "Bearer stale-key"})
        assert resp.status == 200
        assert (await resp.json())["user"] == "jason"

    @pytest.mark.asyncio
    async def test_a_key_still_needs_no_client_header(self, protected):
        """Nothing about #261 changes what the bench and the desktop app send."""
        client, key = protected
        resp = await client.post("/api/models/unload",
                                 headers={"Authorization": f"Bearer {key}"})
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_a_member_session_reads_as_a_member(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        body = await (await client.get("/api/whoami", headers=_jar(token))).json()
        assert body["user_role"] == ROLE_MEMBER
        assert body["api_key_id"] == "user:sam"

    @pytest.mark.asyncio
    async def test_a_session_authenticates_on_an_open_node_too(self, open_client):
        """``is_authenticated`` gates trust_remote_code even with auth off."""
        token = await _cookie_of(open_client, "jason", ADMIN_PASSWORD)
        body = await (await open_client.get("/api/whoami",
                                           headers=_jar(token))).json()
        assert body["authenticated"] is True
        assert body["user"] == "jason"

    def test_the_cookie_builder_is_the_one_spelling(self):
        plain = session_cookie("tok", False)
        assert plain == (f"{SESSION_COOKIE}=tok; Path=/; HttpOnly; SameSite=Lax; "
                         f"Max-Age={COOKIE_MAX_AGE}")
        assert session_cookie("tok", True).endswith("; Secure")
        assert session_cookie("", False, max_age=0).startswith(
            f"{SESSION_COOKIE}=; ")


# =============================================================================
# logout and me
# =============================================================================

class TestLogoutAndMe:
    @pytest.mark.asyncio
    async def test_logout_revokes_the_session_and_clears_the_cookie(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.post("/api/auth/logout",
                                 headers={**_jar(token), **DASH})
        assert resp.status == 200
        assert (await resp.json()) == {"ok": True, "revoked": True}
        assert "Max-Age=0" in resp.headers["Set-Cookie"]
        assert client.app["users_store"].sessions_for("jason") == []
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 401

    @pytest.mark.asyncio
    async def test_logout_with_no_session_is_still_200(self, open_client):
        resp = await open_client.post("/api/auth/logout", headers=DASH)
        assert resp.status == 200
        assert (await resp.json()) == {"ok": True, "revoked": False}
        assert "Max-Age=0" in resp.headers["Set-Cookie"]

    @pytest.mark.asyncio
    async def test_logout_only_ends_the_session_that_asked(self, protected):
        client, _ = protected
        mine = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        await _cookie_of(client, "jason", ADMIN_PASSWORD)
        assert len(client.app["users_store"].sessions_for("jason")) == 2
        await client.post("/api/auth/logout", headers={**_jar(mine), **DASH})
        assert len(client.app["users_store"].sessions_for("jason")) == 1

    @pytest.mark.asyncio
    async def test_me_answers_a_stranger_with_null_and_nothing_else(self, protected):
        client, _ = protected
        resp = await client.get("/api/auth/me")
        assert resp.status == 200
        assert await resp.json() == {"user": None, "auth_enabled": True,
                                     "has_users": True}

    @pytest.mark.asyncio
    async def test_me_answers_a_session_with_its_account_and_session(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        body = await (await client.get("/api/auth/me", headers=_jar(token))).json()
        assert body["auth_enabled"] is True and body["has_users"] is True
        assert body["user"]["name"] == "jason"
        assert body["user"]["role"] == ROLE_ADMIN
        assert set(body["user"]["session"]) >= {"id", "created_at", "last_seen"}
        assert "token_hash" not in body["user"]["session"]

    @pytest.mark.asyncio
    async def test_me_says_an_open_node_with_no_accounts_is_both(self, stores):
        auth_config, users = stores
        users.import_users([])
        async with _client(_make_app(auth_config, users)) as client:
            assert await (await client.get("/api/auth/me")).json() == {
                "user": None, "auth_enabled": False, "has_users": False}

    @pytest.mark.asyncio
    async def test_auth_status_counts_a_session_as_authenticated(self, protected):
        """The dashboard shows "stop requiring a key" only to an authenticated
        caller (#262), and a person who has logged in is one."""
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        body = await (await client.get("/api/auth/status",
                                       headers=_jar(token))).json()
        assert body["authenticated"] is True
        anonymous = await (await client.get("/api/auth/status")).json()
        assert anonymous["authenticated"] is False

    @pytest.mark.asyncio
    async def test_an_api_key_is_not_a_person(self, protected):
        client, key = protected
        body = await (await client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {key}"})).json()
        assert body["user"] is None


# =============================================================================
# Sessions and the self-service password change
# =============================================================================

class TestSessionRoutes:
    @pytest.mark.asyncio
    async def test_a_member_sees_its_own_sessions_only(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        await _cookie_of(client, "jason", ADMIN_PASSWORD)

        resp = await client.get("/api/auth/sessions", headers=_jar(token))
        body = await resp.json()
        assert body["user"] == "sam"
        assert [s["user"] for s in body["sessions"]] == ["sam"]

        resp = await client.get("/api/auth/sessions?user=jason", headers=_jar(token))
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_an_admin_may_read_another_account(self, protected):
        client, _ = protected
        await _cookie_of(client, "sam", MEMBER_PASSWORD)
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        body = await (await client.get("/api/auth/sessions?user=sam",
                                       headers=_jar(token))).json()
        assert [s["user"] for s in body["sessions"]] == ["sam"]

    @pytest.mark.asyncio
    async def test_a_bad_name_in_the_query_is_a_400(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        resp = await client.get("/api/auth/sessions?user=not%20a%20name",
                                headers=_jar(token))
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_a_key_with_no_query_sees_no_sessions(self, protected):
        client, key = protected
        await _cookie_of(client, "jason", ADMIN_PASSWORD)
        body = await (await client.get(
            "/api/auth/sessions",
            headers={"Authorization": f"Bearer {key}"})).json()
        assert body == {"user": None, "sessions": []}

    @pytest.mark.asyncio
    async def test_a_member_revokes_its_own_session_and_not_another(self, protected):
        client, _ = protected
        member = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        await _cookie_of(client, "jason", ADMIN_PASSWORD)
        store = client.app["users_store"]
        theirs = store.sessions_for("jason")[0]["id"]
        mine = store.sessions_for("sam")[0]["id"]

        resp = await client.delete(f"/api/auth/sessions/{theirs}",
                                   headers={**_jar(member), **DASH})
        assert resp.status == 404, "a member learned that somebody else's id exists"

        resp = await client.delete(f"/api/auth/sessions/{mine}",
                                   headers={**_jar(member), **DASH})
        assert resp.status == 200
        assert store.sessions_for("sam") == []

    @pytest.mark.asyncio
    async def test_an_admin_revokes_anybodys_session(self, protected):
        client, _ = protected
        await _cookie_of(client, "sam", MEMBER_PASSWORD)
        admin = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        store = client.app["users_store"]
        theirs = store.sessions_for("sam")[0]["id"]
        resp = await client.delete(f"/api/auth/sessions/{theirs}",
                                   headers={**_jar(admin), **DASH})
        assert resp.status == 200
        assert store.sessions_for("sam") == []

    @pytest.mark.asyncio
    async def test_a_password_change_keeps_this_browser_and_drops_the_others(
            self, protected):
        client, _ = protected
        keeping = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        elsewhere = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        resp = await client.post("/api/auth/password",
                                 json={"current": MEMBER_PASSWORD,
                                       "new": "a brand new password"},
                                 headers={**_jar(keeping), **DASH})
        assert resp.status == 200
        fresh = resp.headers["Set-Cookie"].split(";")[0].split("=", 1)[1]
        assert fresh != keeping

        store = client.app["users_store"]
        assert store.verify_password("sam", "a brand new password") is True
        assert len(store.sessions_for("sam")) == 1
        assert (await client.get("/api/whoami", headers=_jar(elsewhere))).status == 401
        assert (await client.get("/api/whoami", headers=_jar(fresh))).status == 200

    @pytest.mark.asyncio
    async def test_a_wrong_current_password_is_a_403(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        resp = await client.post("/api/auth/password",
                                 json={"current": "nope", "new": "a new password"},
                                 headers={**_jar(token), **DASH})
        assert resp.status == 403
        assert client.app["users_store"].verify_password(
            "sam", MEMBER_PASSWORD) is True

    @pytest.mark.asyncio
    async def test_a_short_new_password_is_refused(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        resp = await client.post("/api/auth/password",
                                 json={"current": MEMBER_PASSWORD, "new": "short"},
                                 headers={**_jar(token), **DASH})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_a_key_cannot_change_a_password_through_the_self_route(self,
                                                                        protected):
        client, key = protected
        resp = await client.post("/api/auth/password",
                                 json={"current": "x", "new": "a new password"},
                                 headers={"Authorization": f"Bearer {key}"})
        assert resp.status == 403
        assert "/api/auth/users/<name>/password" in (
            await resp.json())["error"]["message"]


# =============================================================================
# Accounts, and who may manage them
# =============================================================================

class TestUserRoutes:
    @pytest.mark.asyncio
    async def test_an_admin_session_manages_accounts(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        headers = {**_jar(token), **DASH}

        resp = await client.post("/api/auth/users",
                                 json={"name": "kim", "password": "a good password",
                                       "role": "member"}, headers=headers)
        assert resp.status == 201
        assert (await resp.json())["user"] == {
            "name": "kim", "role": ROLE_MEMBER,
            "created_at": (await resp.json())["user"]["created_at"],
            "disabled": False}

        body = await (await client.get("/api/auth/users", headers=headers)).json()
        assert {u["name"] for u in body["users"]} == {"jason", "sam", "kim"}
        assert body["admin_count"] == 1
        assert all("password_hash" not in u for u in body["users"])

        resp = await client.delete("/api/auth/users/kim", headers=headers)
        assert resp.status == 200
        assert client.app["users_store"].find_user("kim") is None

    @pytest.mark.asyncio
    async def test_a_member_session_is_refused_everywhere(self, protected):
        client, _ = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        headers = {**_jar(token), **DASH}
        refused = [
            ("GET", "/api/auth/users", None),
            ("POST", "/api/auth/users", {"name": "kim", "password": "a password"}),
            ("DELETE", "/api/auth/users/jason", None),
            ("POST", "/api/auth/users/jason/password", {"password": "a password"}),
            ("POST", "/api/auth/users/jason/disable", {}),
            ("POST", "/api/auth/users/jason/enable", {}),
            ("GET", "/api/auth/keys", None),
            ("POST", "/api/auth/keys", {}),
            ("DELETE", "/api/auth/keys/whatever", None),
            ("POST", "/api/auth/enable", {}),
            ("POST", "/api/auth/disable", {}),
        ]
        for method, path, body in refused:
            kwargs = {"json": body} if body is not None else {}
            resp = await client.request(method, path, headers=headers, **kwargs)
            assert resp.status == 403, f"{method} {path} answered {resp.status}"
            assert (await resp.json())["error"]["type"] == "forbidden"
        assert client.app["auth_config"].enabled is True, "a member turned auth off"

    @pytest.mark.asyncio
    async def test_an_operator_key_manages_accounts(self, protected):
        """How the CLI works, and how the first admin is created."""
        client, key = protected
        headers = {"Authorization": f"Bearer {key}"}
        resp = await client.post("/api/auth/users",
                                 json={"name": "kim", "password": "a good password",
                                       "role": "admin"}, headers=headers)
        assert resp.status == 201
        assert client.app["users_store"].admin_count() == 2
        # A browser holding a key and no account still gets the list, not a 403:
        # the dashboard renders 403 as "only an admin can manage accounts".
        resp = await client.get("/api/auth/users", headers=headers)
        assert resp.status == 200
        assert {u["name"] for u in (await resp.json())["users"]} == {
            "jason", "sam", "kim"}

    @pytest.mark.asyncio
    async def test_the_fleet_key_manages_accounts(self, protected):
        client, _ = protected
        headers = {"Authorization": f"Bearer {fleet_key(SECRET)}"}
        resp = await client.get("/api/auth/users", headers=headers)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_the_first_admin_can_be_created_on_an_open_node(self, stores):
        """A fresh install with auth off and nobody to authorise anything."""
        auth_config, users = stores
        users.import_users([])
        async with _client(_make_app(auth_config, users)) as client:
            resp = await client.post("/api/auth/users",
                                     json={"name": "jason", "role": "admin",
                                           "password": "a good password"})
            assert resp.status == 201
            assert users.admin_count() == 1

    @pytest.mark.asyncio
    async def test_a_bad_account_is_a_400_carrying_the_rule(self, protected):
        client, key = protected
        headers = {"Authorization": f"Bearer {key}"}
        for body, expect in (
            ({"name": "has space", "password": "a good password"}, "1 to 64"),
            ({"name": "kim", "password": "short"}, "at least 8"),
            ({"name": "kim", "password": "a good password", "role": "root"},
             "Role must be one of"),
            ({"name": "jason", "password": "a good password"}, "already an account"),
        ):
            resp = await client.post("/api/auth/users", json=body, headers=headers)
            assert resp.status == 400, body
            assert expect in (await resp.json())["error"]["message"]

    @pytest.mark.asyncio
    async def test_the_last_admin_cannot_be_removed_or_disabled(self, protected):
        client, key = protected
        headers = {"Authorization": f"Bearer {key}"}
        for method, path in (("DELETE", "/api/auth/users/jason"),
                             ("POST", "/api/auth/users/jason/disable")):
            resp = await client.request(method, path, headers=headers)
            assert resp.status == 409, f"{method} {path} answered {resp.status}"
            assert "only admin" in (await resp.json())["error"]["message"]
        assert client.app["users_store"].admin_count() == 1

    @pytest.mark.asyncio
    async def test_disable_and_enable_round_trip(self, protected):
        client, key = protected
        headers = {"Authorization": f"Bearer {key}"}
        resp = await client.post("/api/auth/users/sam/disable", headers=headers)
        assert resp.status == 200
        assert (await resp.json())["user"]["disabled"] is True
        resp = await client.post("/api/auth/users/sam/enable", headers=headers)
        assert (await resp.json())["user"]["disabled"] is False

    @pytest.mark.asyncio
    async def test_an_admin_resets_a_password_and_logs_that_account_out(self,
                                                                       protected):
        client, key = protected
        token = await _cookie_of(client, "sam", MEMBER_PASSWORD)
        resp = await client.post("/api/auth/users/sam/password",
                                 json={"password": "a reset password"},
                                 headers={"Authorization": f"Bearer {key}"})
        assert resp.status == 200
        assert (await client.get("/api/whoami", headers=_jar(token))).status == 401
        assert (await _login(client, "sam", "a reset password")).status == 200

    @pytest.mark.asyncio
    async def test_an_unknown_account_is_a_404(self, protected):
        client, key = protected
        headers = {"Authorization": f"Bearer {key}"}
        for method, path, body in (
            ("DELETE", "/api/auth/users/nobody", None),
            ("POST", "/api/auth/users/nobody/password", {"password": "a password"}),
            ("POST", "/api/auth/users/nobody/disable", {}),
            ("POST", "/api/auth/users/nobody/enable", {}),
        ):
            kwargs = {"json": body} if body is not None else {}
            resp = await client.request(method, path, headers=headers, **kwargs)
            assert resp.status == 404, f"{method} {path} answered {resp.status}"

    @pytest.mark.asyncio
    async def test_every_mutation_tells_the_broadcaster(self, stores):
        """The hook the CLI and fleet half registers, so a head knows to replicate."""
        auth_config, users = stores
        entry = auth_config.enable("test")
        app = _make_app(auth_config, users)
        calls = []
        app["users_changed"] = lambda: calls.append(1)
        headers = {"Authorization": f"Bearer {entry['key']}"}
        async with _client(app) as client:
            await client.post("/api/auth/users",
                              json={"name": "kim", "password": "a good password"},
                              headers=headers)
            await client.post("/api/auth/users/kim/password",
                              json={"password": "another password"}, headers=headers)
            await client.post("/api/auth/users/kim/disable", headers=headers)
            await client.post("/api/auth/users/kim/enable", headers=headers)
            await client.delete("/api/auth/users/kim", headers=headers)
        assert len(calls) == 5

    @pytest.mark.asyncio
    async def test_an_async_broadcaster_is_scheduled_and_not_awaited(self, stores):
        """A fan-out to every peer must not be inside the operator's 201."""
        auth_config, users = stores
        entry = auth_config.enable("test")
        app = _make_app(auth_config, users)
        ran = []

        async def broadcast():
            ran.append(1)

        app["users_changed"] = broadcast
        async with _client(app) as client:
            resp = await client.post(
                "/api/auth/users",
                json={"name": "kim", "password": "a good password"},
                headers={"Authorization": f"Bearer {entry['key']}"})
            assert resp.status == 201
            await asyncio.sleep(0)
        assert ran == [1]

    @pytest.mark.asyncio
    async def test_a_broken_broadcaster_does_not_fail_the_mutation(self, stores):
        auth_config, users = stores
        entry = auth_config.enable("test")
        app = _make_app(auth_config, users)

        def boom():
            raise RuntimeError("no peers today")

        app["users_changed"] = boom
        async with _client(app) as client:
            resp = await client.post(
                "/api/auth/users",
                json={"name": "kim", "password": "a good password"},
                headers={"Authorization": f"Bearer {entry['key']}"})
            assert resp.status == 201


# =============================================================================
# Fleet replication
# =============================================================================

class TestReplicationRoutes:
    @pytest.mark.asyncio
    async def test_export_and_sync_are_the_fleet_key_alone(self, protected):
        client, key = protected
        admin = await _cookie_of(client, "jason", ADMIN_PASSWORD)
        for headers in ({"Authorization": f"Bearer {key}"},
                        {**_jar(admin), **DASH}):
            resp = await client.get("/api/auth/users/export", headers=headers)
            assert resp.status == 403, "an operator read the password hashes"
            assert "fleet key" in (await resp.json())["error"]["message"]
            resp = await client.post("/api/auth/users/sync", json={"users": []},
                                     headers=headers)
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_the_fleet_key_exports_the_hashes_with_a_stamp(self, protected):
        client, _ = protected
        headers = {"Authorization": f"Bearer {fleet_key(SECRET)}"}
        body = await (await client.get("/api/auth/users/export",
                                       headers=headers)).json()
        assert [u["name"] for u in body["users"]] == ["jason", "sam"]
        assert body["users"][0]["password_hash"]
        assert body["stamp"] == client.app["users_store"].export_stamp()

    @pytest.mark.asyncio
    async def test_sync_adopts_a_list_once(self, protected):
        client, _ = protected
        headers = {"Authorization": f"Bearer {fleet_key(SECRET)}"}
        exported = client.app["users_store"].export_users()
        incoming = [u for u in exported if u["name"] == "jason"]

        resp = await client.post("/api/auth/users/sync", json={"users": incoming},
                                 headers=headers)
        assert await resp.json() == {"changed": True, "count": 1}

        resp = await client.post("/api/auth/users/sync", json={"users": incoming},
                                 headers=headers)
        assert await resp.json() == {"changed": False, "count": 1}
        assert client.app["users_store"].verify_password(
            "jason", ADMIN_PASSWORD) is True

    @pytest.mark.asyncio
    async def test_a_malformed_sync_leaves_the_node_alone(self, protected):
        client, _ = protected
        headers = {"Authorization": f"Bearer {fleet_key(SECRET)}"}
        before = client.app["users_store"].export_users()

        resp = await client.post("/api/auth/users/sync", json={}, headers=headers)
        assert resp.status == 400
        assert "'users' list" in (await resp.json())["error"]["message"]

        resp = await client.post("/api/auth/users/sync",
                                 json={"users": [{"name": "no hash"}]},
                                 headers=headers)
        assert resp.status == 400
        assert client.app["users_store"].export_users() == before

    @pytest.mark.asyncio
    async def test_export_is_not_swallowed_by_the_name_route(self, protected):
        """``/users/export`` must not resolve as ``/users/{name}``."""
        client, _ = protected
        resp = await client.get("/api/auth/users/export",
                                headers={"Authorization": f"Bearer {fleet_key(SECRET)}"})
        assert resp.status == 200
        assert "users" in await resp.json()
