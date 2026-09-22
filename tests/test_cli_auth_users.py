"""``ainode auth user`` and ``ainode auth session``: the operator's side of the login.

The dashboard login (#261) has no sign-up page, and must not: a route that mints the
first admin over HTTP is a route anybody who can reach port 3000 calls before the
operator does. So the first account is made on the box, and these commands are the
whole path to having one. What is pinned here:

1. **The argparse wiring**, because a subcommand that is not wired is a typo an
   operator discovers instead of a test.
2. **Both ways of supplying a password.** Interactive is two ``getpass`` prompts
   that have to agree; ``--password-stdin`` is the answer for anything with no
   terminal, which includes ``ssh host ainode ...`` and every script, because
   getpass needs a TTY. The installer's host wrapper had to stop hardcoding
   ``docker exec -it`` for that flag to work at all, which is pinned at the bottom.
3. **A short password is refused** before anything is written.
4. **The last admin cannot be removed or disabled.** With auth on and no admin, the
   dashboard can only be opened by pasting an API key, which is the state
   ``ainode doctor``'s Login check FAILs on.
5. **Replication is triggered on a master and warned about on a worker**, and it
   carries the fleet key. A change made on the wrong node is still made, because
   refusing would leave an operator with neither an account nor an explanation.
6. **Sessions are never replicated**, and ``ainode auth status`` counts them beside
   the keys.

The account store lands on its own branch (``ainode/auth/accounts.py``), so the
store here is :class:`StubStore` and the one guarded import the package makes
(``replication.open_store``) is what the tests replace.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ainode.auth import replication as rep
from ainode.auth.fleet import fleet_key
from ainode.cli import main as cli

SECRET = "0123456789abcdef0123456789abcdef"


class StubStore:
    """``UsersStore`` as the CLI uses it, in memory."""

    MIN_PASSWORD_LENGTH = 8

    def __init__(self, users=None, sessions=None, can_disable=True):
        self.users = [dict(u) for u in (users or [])]
        self.sessions = {k: [dict(s) for s in v] for k, v in (sessions or {}).items()}
        self.saved = 0
        self.can_disable = can_disable

    # -- opening ---------------------------------------------------------
    def load(self):
        return self

    def save(self):
        self.saved += 1

    # -- accounts --------------------------------------------------------
    def add_user(self, name, password, role="member"):
        if any(u["name"] == name for u in self.users):
            raise ValueError(f"an account named {name} already exists")
        if len(password) < self.MIN_PASSWORD_LENGTH:
            raise ValueError("password too short")
        self.users.append({"name": name, "role": role, "disabled": False,
                           "password_hash": f"hash-of-{password}"})
        return True

    def remove_user(self, name):
        before = len(self.users)
        self.users = [u for u in self.users if u["name"] != name]
        self.sessions.pop(name, None)
        return len(self.users) != before

    def set_password(self, name, password):
        for user in self.users:
            if user["name"] == name:
                user["password_hash"] = f"hash-of-{password}"
                return True
        return False

    def set_enabled(self, name, enabled):
        if not self.can_disable:
            raise AttributeError("no such method")
        for user in self.users:
            if user["name"] == name:
                user["disabled"] = not enabled
                return True
        return False

    def list_users(self):
        return [{"name": u["name"], "role": u["role"],
                 "disabled": u.get("disabled", False)} for u in self.users]

    def admin_count(self):
        return sum(1 for u in self.users
                   if u["role"] == "admin" and not u.get("disabled"))

    def has_users(self):
        return bool(self.users)

    def export_users(self):
        return [dict(u) for u in self.users]

    # -- sessions --------------------------------------------------------
    def sessions_for(self, name):
        return [dict(s) for s in self.sessions.get(name, [])]

    def revoke_session(self, session_id, user=None):
        for name, rows in self.sessions.items():
            if user and name != user:
                continue
            kept = [s for s in rows if s.get("id") != session_id]
            if len(kept) != len(rows):
                self.sessions[name] = kept
                return True
        return False


class NoDisableStore(StubStore):
    """A store with no disable of its own: the one method the contract left open."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    set_enabled = None  # type: ignore[assignment]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Every file these commands touch, inside tmp_path.

    ``AUTH_FILE`` and ``CONFIG_FILE`` are computed at import, so the redirect has to
    name them as well as the environment variable (the same shape
    ``tests/test_fleet_auth.py`` uses).
    """
    monkeypatch.setenv("AINODE_HOME", str(tmp_path))
    monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("ainode.auth.middleware.AINODE_HOME", tmp_path)
    monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
    return tmp_path


@pytest.fixture
def store(monkeypatch):
    """One store for the run, in place of the guarded import."""
    stub = StubStore()
    monkeypatch.setattr(rep, "open_store", lambda app=None: stub)
    return stub


@pytest.fixture
def no_peer_calls(monkeypatch):
    """No replication request leaves the test unless a test asks for one.

    ``/api/cluster/info`` on 127.0.0.1 is a real request otherwise, and on the
    operator's own machine it would reach their own node.
    """
    calls: list[dict] = []

    def fake_http_json(url, payload=None, headers=None, timeout=10.0):
        calls.append({"url": url, "payload": payload, "headers": headers or {}})
        return 0, None

    monkeypatch.setattr(rep, "http_json", fake_http_json)
    return calls


def run(*argv, stdin: str = ""):
    """``ainode <argv>``, with stdin fed from a string. Returns (exit code, output)."""
    from rich.console import Console

    buffer = io.StringIO()
    with patch.object(cli, "console", Console(file=buffer, width=100,
                                             force_terminal=False)):
        with patch.object(sys, "argv", ["ainode", *argv]):
            with patch.object(sys, "stdin", io.StringIO(stdin)):
                code = 0
                try:
                    cli.main()
                except SystemExit as exc:
                    code = exc.code or 0
    return code, buffer.getvalue()


def _config(home, **keys):
    import json

    payload = {"node_id": "n1", "node_name": "n1", "cluster_secret": SECRET,
               "onboarded": True}
    payload.update(keys)
    (home / "config.json").write_text(json.dumps(payload))


# =============================================================================
# 1. The wiring
# =============================================================================

def test_every_account_subcommand_is_wired(home, store, no_peer_calls):
    """A subcommand argparse does not know about exits 2 with a usage message, and
    nothing in the product would notice."""
    _config(home)
    for argv in (("auth", "user", "list"),
                 ("auth", "session", "list"),
                 ("auth", "status")):
        code, out = run(*argv)
        assert code == 0, f"{argv} exited {code}: {out}"


def test_the_usage_line_names_the_new_commands(home, store, no_peer_calls):
    _config(home)
    code, out = run("auth")
    assert "user" in out and "session" in out
    assert "--password-stdin" in out


def test_password_stdin_is_documented_in_the_help(capsys):
    """getpass needs a terminal, and the flag is the documented way to work without
    one, so the help text has to say so where an operator will read it."""
    with patch.object(sys, "argv", ["ainode", "auth", "user", "add", "--help"]):
        with pytest.raises(SystemExit):
            cli.main()
    out = capsys.readouterr().out
    assert "--password-stdin" in out
    assert "TTY" in out or "tty" in out


def test_the_key_commands_still_work(home, monkeypatch):
    """`ainode auth key ...` and `auth status|enable|disable` are not disturbed."""
    _config(home)
    code, out = run("auth", "key", "create", "--name", "laptop")
    assert code == 0 and "New API key created" in out
    code, out = run("auth", "key", "list")
    assert code == 0 and "laptop" in out
    code, out = run("auth", "enable")
    assert code == 0 and "Auth enabled" in out
    code, out = run("auth", "disable")
    assert code == 0 and "Auth disabled" in out


# =============================================================================
# 2. Adding an account
# =============================================================================

def test_add_prompts_twice_and_writes_the_account(home, store, no_peer_calls,
                                                  monkeypatch):
    _config(home)
    asked = []

    def fake_getpass(prompt=""):
        asked.append(prompt)
        return "a-good-password"

    monkeypatch.setattr("getpass.getpass", fake_getpass)

    code, out = run("auth", "user", "add", "jason", "--admin")

    assert code == 0, out
    assert len(asked) == 2, "a new password is typed twice or it is a typo"
    assert store.users == [{"name": "jason", "role": "admin", "disabled": False,
                            "password_hash": "hash-of-a-good-password"}]
    assert store.saved >= 1, "the account has to be on disk, not just in memory"
    assert "added" in out and "admin" in out
    assert "re-reads users.json" in out, "the running node picks the file up live"
    assert "Made in Texas" in out


def test_add_refuses_when_the_two_prompts_disagree(home, store, no_peer_calls,
                                                   monkeypatch):
    _config(home)
    answers = iter(["first-password", "second-password"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    code, out = run("auth", "user", "add", "jason")

    assert code == 2
    assert "do not match" in out
    assert store.users == []


def test_add_reads_the_password_from_stdin_with_password_stdin(home, store,
                                                              no_peer_calls,
                                                              monkeypatch):
    _config(home)
    monkeypatch.setattr("getpass.getpass",
                        lambda prompt="": pytest.fail("must not prompt with a pipe"))

    code, out = run("auth", "user", "add", "jason", "--admin", "--password-stdin",
                    stdin="piped-password\n")

    assert code == 0, out
    assert store.users[0]["password_hash"] == "hash-of-piped-password", \
        "only the newline the shell added is stripped"


def test_add_refuses_a_short_password(home, store, no_peer_calls):
    _config(home)
    code, out = run("auth", "user", "add", "jason", "--password-stdin", stdin="short\n")

    assert code == 2
    assert "too short" in out and "8" in out
    assert store.users == []


def test_add_refuses_a_duplicate_name_with_the_stores_own_message(home, store,
                                                                  no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"}]

    code, out = run("auth", "user", "add", "jason", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 2
    assert "already exists" in out


def test_a_name_typed_with_capitals_is_folded_to_the_contracts_alphabet(
        home, store, no_peer_calls):
    _config(home)
    code, out = run("auth", "user", "add", "Jason", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 0, out
    assert store.users[0]["name"] == "jason"
    assert "jason" in out


def test_a_node_with_no_account_store_says_so_instead_of_raising(home, monkeypatch,
                                                                no_peer_calls):
    _config(home)
    monkeypatch.setattr(rep, "open_store", lambda app=None: None)

    code, out = run("auth", "user", "list")

    assert code == 2
    assert "no account store" in out
    assert "auth key create" in out, "the key path still works, so it is named"


# =============================================================================
# 3. The last admin
# =============================================================================

def test_removing_the_last_admin_is_refused(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]

    code, out = run("auth", "user", "remove", "jason")

    assert code == 2
    assert "only admin" in out
    assert [u["name"] for u in store.users] == ["jason", "ops"]


def test_disabling_the_last_admin_is_refused_too(home, store, no_peer_calls):
    """A disabled admin cannot sign in, so it locks the dashboard exactly as hard
    as a deleted one."""
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"}]

    code, out = run("auth", "user", "disable", "jason")

    assert code == 2
    assert "only admin" in out
    assert store.users[0]["disabled"] is False


def test_a_second_admin_makes_the_first_removable(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "sem", "role": "admin", "disabled": False,
                    "password_hash": "y"}]

    code, out = run("auth", "user", "remove", "jason")

    assert code == 0, out
    assert [u["name"] for u in store.users] == ["sem"]
    assert "removed" in out


def test_removing_a_member_is_never_refused(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]

    code, out = run("auth", "user", "remove", "ops")

    assert code == 0, out
    assert [u["name"] for u in store.users] == ["jason"]


def test_an_unknown_account_is_a_refusal_and_not_a_traceback(home, store,
                                                             no_peer_calls):
    _config(home)
    for action in ("remove", "passwd", "disable", "enable"):
        code, out = run("auth", "user", action, "nobody")
        assert code == 2, action
        assert "No account named 'nobody'" in out


# =============================================================================
# 4. The rest of the account commands
# =============================================================================

def test_list_shows_the_role_the_state_and_the_session_count(home, store,
                                                             no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "THE-STORED-HASH"},
                   {"name": "ops", "role": "member", "disabled": True,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1"}, {"id": "s2"}]}

    code, out = run("auth", "user", "list")

    assert code == 0, out
    assert "jason" in out and "admin" in out and "disabled" in out
    assert "not replicated" in out, "sessions are per node and the list says so"
    assert "THE-STORED-HASH" not in out, "no password hash is ever printed"


def test_passwd_changes_the_hash_and_names_the_sign_out_command(home, store,
                                                                no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "old"}]

    code, out = run("auth", "user", "passwd", "jason", "--password-stdin",
                    stdin="a-new-password\n")

    assert code == 0, out
    assert store.users[0]["password_hash"] == "hash-of-a-new-password"
    assert "session clear --user jason" in out


def test_disable_then_enable_round_trips(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]

    code, out = run("auth", "user", "disable", "ops")
    assert code == 0, out
    assert store.users[1]["disabled"] is True
    assert "cannot sign in" in out

    code, out = run("auth", "user", "enable", "ops")
    assert code == 0, out
    assert store.users[1]["disabled"] is False


def test_a_store_with_no_disable_of_its_own_points_at_the_route(home, monkeypatch,
                                                               no_peer_calls):
    """The one method the account contract did not name. The CLI asks for the three
    spellings a store of this shape would use, and says so plainly when it has none
    rather than writing a field name into somebody else's file."""
    _config(home)
    stub = NoDisableStore([{"name": "jason", "role": "admin", "disabled": False,
                            "password_hash": "x"},
                           {"name": "ops", "role": "member", "disabled": False,
                            "password_hash": "y"}])
    monkeypatch.setattr(rep, "open_store", lambda app=None: stub)

    code, out = run("auth", "user", "disable", "ops")

    assert code == 2
    assert "/api/auth/users/ops/disable" in out


# =============================================================================
# 5. Sessions, which are per node
# =============================================================================

def test_session_list_shows_every_session_on_this_node(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1", "created_at": "2026-09-21T10:00:00Z"}],
                      "ops": [{"id": "s2"}]}

    code, out = run("auth", "session", "list")

    assert code == 0, out
    assert "s1" in out and "s2" in out
    assert "per node" in out


def test_session_list_can_be_narrowed_to_one_user(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1"}], "ops": [{"id": "s2"}]}

    code, out = run("auth", "session", "list", "--user", "jason")

    assert code == 0, out
    assert "s1" in out and "s2" not in out


def test_session_revoke_takes_one_session_away(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"}]
    store.sessions = {"jason": [{"id": "s1"}, {"id": "s2"}]}

    code, out = run("auth", "session", "revoke", "s1")

    assert code == 0, out
    assert [s["id"] for s in store.sessions["jason"]] == ["s2"]


def test_session_revoke_of_an_unknown_id_is_a_refusal(home, store, no_peer_calls):
    _config(home)
    code, out = run("auth", "session", "revoke", "nope")

    assert code == 2
    assert "No session 'nope'" in out


def test_session_clear_signs_everybody_out(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1"}, {"id": "s2"}], "ops": [{"id": "s3"}]}

    code, out = run("auth", "session", "clear")

    assert code == 0, out
    assert store.sessions == {"jason": [], "ops": []}
    assert "3 session(s)" in out


def test_session_clear_can_be_narrowed_to_one_user(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1"}], "ops": [{"id": "s3"}]}

    code, out = run("auth", "session", "clear", "--user", "ops")

    assert code == 0, out
    assert [s["id"] for s in store.sessions["jason"]] == ["s1"]
    assert store.sessions["ops"] == []


def test_a_session_command_never_replicates_anything(home, store, no_peer_calls):
    """Sessions are per node: there is nothing here to fan out."""
    _config(home, cluster_role="master", peer_ips=["10.0.0.2"])
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"}]
    store.sessions = {"jason": [{"id": "s1"}]}

    run("auth", "session", "clear")

    assert no_peer_calls == []


# =============================================================================
# 6. Replication, from the CLI
# =============================================================================

def test_a_master_pushes_the_change_to_its_peers_with_the_fleet_key(
        home, store, no_peer_calls):
    _config(home, cluster_role="master", peer_ips=["10.0.0.2"])

    code, out = run("auth", "user", "add", "jason", "--admin", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 0, out
    posts = [c for c in no_peer_calls if c["payload"] is not None]
    assert [c["url"] for c in posts] == ["http://10.0.0.2:3000/api/auth/users/sync"]
    assert posts[0]["headers"]["Authorization"] == f"Bearer {fleet_key(SECRET)}"
    assert list(posts[0]["payload"]) == ["users"], "sessions are never replicated"
    assert "10.0.0.2 did not take the update" in out, \
        "the stub answers nothing, so the operator is told and the loop retries"


def test_a_worker_writes_the_change_and_warns_that_it_will_be_overwritten(
        home, store, no_peer_calls):
    _config(home, cluster_role="worker", distributed_mode="member",
            master_address="10.0.0.1:3000")

    code, out = run("auth", "user", "add", "jason", "--admin", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 0, out
    assert store.users[0]["name"] == "jason", "the change is made, not refused"
    assert "managed on the master" in out
    assert "overwritten by the next sync" in out
    assert [c for c in no_peer_calls if c["payload"] is not None] == [], \
        "a worker does not push its own accounts at the fleet"


def test_a_solo_node_says_nothing_about_replication(home, store, no_peer_calls):
    _config(home)

    code, out = run("auth", "user", "add", "jason", "--admin", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 0, out
    assert "master" not in out
    assert [c for c in no_peer_calls if c["payload"] is not None] == []


def test_the_role_comes_from_the_running_node_when_it_answers(home, store,
                                                              monkeypatch):
    """`/api/cluster/info` is the live answer: a node whose config says nothing is
    still a worker if the cluster elected somebody else."""
    _config(home)
    calls: list[dict] = []

    def fake_http_json(url, payload=None, headers=None, timeout=10.0):
        calls.append({"url": url, "payload": payload})
        if rep.CLUSTER_INFO_PATH in url:
            return 200, {"my_role": "worker", "my_node_id": "n1",
                         "master_address": "10.0.0.1:3000",
                         "members": [{"node_id": "m1"}, {"node_id": "n1"}]}
        return 200, {"changed": True}

    monkeypatch.setattr(rep, "http_json", fake_http_json)

    code, out = run("auth", "user", "add", "jason", "--admin", "--password-stdin",
                    stdin="a-good-password\n")

    assert code == 0, out
    assert "managed on the master" in out
    assert [c["url"] for c in calls if c["payload"] is not None] == []


def test_every_account_mutation_replicates_and_no_read_does(home, store,
                                                            no_peer_calls):
    _config(home, cluster_role="master", peer_ips=["10.0.0.2"])
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]

    for argv, stdin in ((("auth", "user", "passwd", "ops"), "a-good-password\n"),
                        (("auth", "user", "disable", "ops"), ""),
                        (("auth", "user", "enable", "ops"), ""),
                        (("auth", "user", "remove", "ops"), "")):
        no_peer_calls.clear()
        argv = argv + ("--password-stdin",) if stdin else argv
        code, out = run(*argv, stdin=stdin)
        assert code == 0, out
        assert [c for c in no_peer_calls if c["payload"] is not None], \
            f"{argv} did not replicate"

    no_peer_calls.clear()
    run("auth", "user", "list")
    assert no_peer_calls == [], "reading the list changes nothing, so it pushes nothing"


# =============================================================================
# 7. `ainode auth status`
# =============================================================================

def test_status_counts_users_admins_and_sessions(home, store, no_peer_calls):
    _config(home)
    store.users = [{"name": "jason", "role": "admin", "disabled": False,
                    "password_hash": "x"},
                   {"name": "ops", "role": "member", "disabled": False,
                    "password_hash": "y"}]
    store.sessions = {"jason": [{"id": "s1"}, {"id": "s2"}], "ops": [{"id": "s3"}]}

    code, out = run("auth", "status")

    assert code == 0, out
    assert "Users: 2 (1 admin)" in out
    assert "Sessions: 3" in out
    assert "never" in out and "replicated" in out


def test_status_says_a_protected_node_with_no_account_can_only_take_a_key(
        home, store, no_peer_calls):
    _config(home)
    run("auth", "enable")

    code, out = run("auth", "status")

    assert code == 0, out
    assert "Users: 0" in out
    assert "only be opened with an API key" in out
    assert "auth user add" in out


def test_status_still_works_on_a_build_with_no_account_store(home, monkeypatch,
                                                            no_peer_calls):
    _config(home)
    monkeypatch.setattr(rep, "open_store", lambda app=None: None)

    code, out = run("auth", "status")

    assert code == 0, out
    assert "Auth:" in out and "Users:" not in out


# =============================================================================
# 8. The installer, and the wrapper the flag has to survive
# =============================================================================

INSTALL_SH = Path(__file__).resolve().parent.parent / "scripts" / "install.sh"


def _render_install(tmp_path, env_extra=None):
    """The real installer in --dry-run against a throwaway HOME.

    The same helper ``tests/test_fresh_install.py`` and ``tests/test_fleet_auth.py``
    use, kept here rather than imported so the three files cannot break each other.
    """
    home = tmp_path / "install-home"
    ainode_home = home / ".ainode"
    home.mkdir(parents=True, exist_ok=True)
    sysfs = tmp_path / "sys-class-net"
    sysfs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HOME=str(home), AINODE_HOME=str(ainode_home),
               AINODE_IMAGE="ghcr.io/getainode/ainode:9.9.9",
               SYS_CLASS_NET=str(sysfs))
    for key in ("AINODE_PEERS", "HF_TOKEN", "AINODE_AUTH"):
        env.pop(key, None)
    env.update(env_extra or {})
    proc = subprocess.run(["bash", str(INSTALL_SH), "--dry-run"],
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return ainode_home, proc


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_protected_install_prints_the_two_lines_an_operator_needs(tmp_path):
    _, proc = _render_install(tmp_path)

    assert "ainode auth enable" in proc.stdout
    assert "ainode auth user add <name> --admin" in proc.stdout
    assert "--password-stdin" in proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_open_install_says_nothing_about_a_login(tmp_path):
    """With auth off the dashboard opens without one, so these lines would be
    advice about a problem nobody has."""
    _, proc = _render_install(tmp_path, env_extra={"AINODE_AUTH": "off"})

    assert "auth user add" not in proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_wrapper_asks_for_a_tty_only_when_it_has_one(tmp_path):
    """``--password-stdin`` is the flag for a pipe, and a hardcoded ``docker exec
    -it`` would kill the pipe before the CLI ran: docker answers "the input device
    is not a TTY" and exits. Same failure ``ainode doctor --peer`` routes around."""
    ainode_home, _ = _render_install(tmp_path)
    wrapper = (ainode_home / "ainode-wrapper").read_text()

    assert "docker exec -it" not in wrapper
    assert '[ -t 0 ]' in wrapper
    forward = wrapper[wrapper.index("forward_to_container()"):]
    assert 'exec docker exec $exec_tty' in forward
