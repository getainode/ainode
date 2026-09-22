"""The account and session store: what ``users.json`` is allowed to contain.

Five things are pinned here, and each of them is a promise the login page makes
on behalf of this file (#261):

1. **A password never reaches disk.** What lands is scrypt over a per-password
   salt, and the check is constant time. An unknown or disabled name pays the same
   scrypt cost as a real one, so neither the answer nor the clock says which
   accounts exist.
2. **A session token never reaches disk either**, only its SHA-256, and it is
   returned exactly once. A stolen ``users.json`` logs nobody in.
3. **A session lives until it is revoked**, which is Jason's rule for this
   feature, so the tests here are about REVOCATION being complete: a password
   change, a disable and a removal all take the open browsers with them.
4. **The file is 0600**, like ``auth.json``, and re-read when it changes, so the
   CLI and the server never disagree about who may log in.
5. **Replication is whole records or nothing.** ``import_users`` either adopts a
   valid list or leaves this node exactly as it was, because a half-adopted list
   is a node whose logins quietly differ from the rest of the cluster.
"""

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from ainode.auth.accounts import (
    LAST_SEEN_REFRESH_SECONDS,
    MAX_SESSIONS_PER_USER,
    MIN_PASSWORD_LENGTH,
    ROLE_ADMIN,
    ROLE_MEMBER,
    UsersStore,
    hash_password,
    hash_token,
    normalize_name,
    public_session,
    public_user,
    require_admin,
    verify_hash,
)
from ainode.auth.fleet import FLEET_KEY_ID


GOOD = "hunter2hunter2"
OTHER = "correct horse battery"


@pytest.fixture
def users_file(tmp_path, monkeypatch):
    """Point the module constant at a temp file, the way AUTH_FILE is redirected."""
    path = tmp_path / "users.json"
    monkeypatch.setattr("ainode.auth.accounts.USERS_FILE", path)
    return path


@pytest.fixture
def store(users_file):
    return UsersStore()


@pytest.fixture
def admin_store(store):
    store.add_user("jason", GOOD, ROLE_ADMIN)
    return store


# =============================================================================
# Hashing
# =============================================================================

class TestHashing:
    def test_a_password_hash_is_hex_and_salted(self):
        first_hash, first_salt = hash_password(GOOD)
        second_hash, second_salt = hash_password(GOOD)
        assert len(first_hash) == 64 and len(first_salt) == 32
        bytes.fromhex(first_hash)
        assert first_salt != second_salt, "the salt is not per password"
        assert first_hash != second_hash, "two salts produced one hash"

    def test_verify_accepts_the_password_and_nothing_else(self):
        password_hash, salt = hash_password(GOOD)
        assert verify_hash(GOOD, password_hash, salt) is True
        assert verify_hash(OTHER, password_hash, salt) is False
        assert verify_hash(GOOD, password_hash, "00" * 16) is False
        assert verify_hash(GOOD, "", salt) is False
        assert verify_hash(GOOD, password_hash, "not hex") is False

    def test_a_token_is_stored_as_its_sha256(self):
        assert len(hash_token("abc")) == 64
        assert hash_token("abc") == hash_token("abc")
        assert hash_token("abc") != hash_token("abd")

    def test_names_are_case_folded_and_bounded(self):
        assert normalize_name("Jason") == "jason"
        assert normalize_name("  JASON  ") == "jason"
        assert normalize_name("a.b_c-1") == "a.b_c-1"
        assert normalize_name("") == ""
        assert normalize_name("has space") == ""
        assert normalize_name("bang!") == ""
        assert normalize_name("a" * 64) == "a" * 64
        assert normalize_name("a" * 65) == ""


# =============================================================================
# Accounts
# =============================================================================

class TestAccounts:
    def test_an_account_round_trips_through_the_file(self, store, users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        loaded = UsersStore.load()
        assert loaded.has_users() is True
        assert loaded.admin_count() == 1
        assert loaded.verify_password("jason", GOOD) is True

    def test_the_file_is_0600(self, store, users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        mode = stat.S_IMODE(os.stat(users_file).st_mode)
        assert mode == 0o600, f"users.json is {oct(mode)}"

    def test_a_wide_file_is_tightened_on_load(self, store, users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        os.chmod(users_file, 0o644)
        UsersStore.load()
        assert stat.S_IMODE(os.stat(users_file).st_mode) == 0o600

    def test_the_file_holds_no_plaintext(self, store, users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        token, _ = store.create_session("jason")
        text = users_file.read_text()
        assert GOOD not in text
        assert token not in text
        assert hash_token(token) in text

    def test_list_users_carries_no_hash_and_no_salt(self, admin_store):
        row = admin_store.list_users()[0]
        assert row == {"name": "jason", "role": ROLE_ADMIN,
                       "created_at": row["created_at"], "disabled": False,
                       "sessions": 0}

    def test_a_name_is_normalised_on_the_way_in(self, store):
        store.add_user("  JaSoN ", GOOD)
        assert store.find_user("jason")["name"] == "jason"
        assert store.verify_password("JASON", GOOD) is True

    @pytest.mark.parametrize("name", ["", "has space", "bang!", "a" * 65])
    def test_a_bad_name_is_refused_with_the_rule(self, store, name):
        with pytest.raises(ValueError, match="1 to 64 characters"):
            store.add_user(name, GOOD)

    def test_a_short_password_is_refused_with_the_rule(self, store):
        with pytest.raises(ValueError, match="at least 8 characters"):
            store.add_user("jason", "a" * (MIN_PASSWORD_LENGTH - 1))

    def test_an_unknown_role_is_refused(self, store):
        with pytest.raises(ValueError, match="Role must be one of"):
            store.add_user("jason", GOOD, "superuser")

    def test_a_duplicate_name_is_refused(self, admin_store):
        with pytest.raises(ValueError, match="already an account"):
            admin_store.add_user("JASON", OTHER)

    def test_verify_is_false_for_unknown_and_disabled(self, admin_store):
        admin_store.add_user("sam", GOOD)
        assert admin_store.verify_password("sam", GOOD) is True
        assert admin_store.verify_password("sam", OTHER) is False
        assert admin_store.verify_password("nobody", GOOD) is False
        admin_store.set_disabled("sam", True)
        assert admin_store.verify_password("sam", GOOD) is False
        assert admin_store.role_of("sam") == ""
        admin_store.set_disabled("sam", False)
        assert admin_store.verify_password("sam", GOOD) is True
        assert admin_store.role_of("sam") == ROLE_MEMBER

    def test_a_password_change_revokes_every_session(self, admin_store):
        admin_store.create_session("jason")
        admin_store.create_session("jason")
        assert len(admin_store.sessions_for("jason")) == 2
        assert admin_store.set_password("jason", OTHER) is True
        assert admin_store.sessions_for("jason") == []
        assert admin_store.verify_password("jason", OTHER) is True
        assert admin_store.verify_password("jason", GOOD) is False

    def test_a_password_change_on_an_unknown_account_is_false(self, store):
        assert store.set_password("nobody", GOOD) is False

    def test_a_short_new_password_is_refused(self, admin_store):
        with pytest.raises(ValueError, match="at least 8 characters"):
            admin_store.set_password("jason", "short")

    def test_disabling_revokes_the_sessions_it_leaves_behind(self, admin_store):
        admin_store.add_user("sam", GOOD)
        token, _ = admin_store.create_session("sam")
        assert admin_store.session_for_token(token) is not None
        admin_store.set_disabled("sam", True)
        assert admin_store.sessions_for("sam") == []
        assert admin_store.session_for_token(token) is None

    def test_removing_an_account_takes_its_sessions(self, admin_store):
        admin_store.add_user("sam", GOOD)
        token, _ = admin_store.create_session("sam")
        assert admin_store.remove_user("sam") is True
        assert admin_store.session_for_token(token) is None
        assert admin_store.sessions == []

    def test_removing_an_unknown_account_is_false(self, store):
        assert store.remove_user("nobody") is False

    def test_the_last_admin_cannot_be_removed_or_disabled(self, admin_store):
        assert admin_store.is_last_admin("jason") is True
        with pytest.raises(ValueError, match="only admin"):
            admin_store.remove_user("jason")
        with pytest.raises(ValueError, match="only admin"):
            admin_store.set_disabled("jason", True)
        assert admin_store.admin_count() == 1

    def test_a_second_admin_unlocks_the_first(self, admin_store):
        admin_store.add_user("sam", GOOD, ROLE_ADMIN)
        assert admin_store.admin_count() == 2
        assert admin_store.is_last_admin("jason") is False
        assert admin_store.remove_user("jason") is True
        assert admin_store.is_last_admin("sam") is True

    def test_a_member_is_never_the_last_admin(self, admin_store):
        admin_store.add_user("sam", GOOD)
        assert admin_store.is_last_admin("sam") is False
        assert admin_store.remove_user("sam") is True

    def test_a_disabled_admin_is_not_counted(self, admin_store):
        admin_store.add_user("sam", GOOD, ROLE_ADMIN)
        admin_store.set_disabled("sam", True)
        assert admin_store.admin_count() == 1
        assert admin_store.is_last_admin("sam") is False


# =============================================================================
# Sessions
# =============================================================================

class TestSessions:
    def test_a_session_is_found_by_its_token_and_by_nothing_else(self, admin_store):
        token, session = admin_store.create_session("jason", client="cli",
                                                    agent="curl/8")
        assert session["user"] == "jason"
        assert session["client"] == "cli"
        assert session["token_hash"] == hash_token(token)
        assert admin_store.session_for_token(token)["id"] == session["id"]
        assert admin_store.session_for_token(token + "x") is None
        assert admin_store.session_for_token("") is None

    def test_the_agent_is_cut_to_80_characters(self, admin_store):
        _, session = admin_store.create_session("jason", agent="M" * 300)
        assert len(session["agent"]) == 80

    def test_a_session_cannot_be_minted_for_an_unknown_or_disabled_account(
            self, admin_store):
        admin_store.add_user("sam", GOOD)
        admin_store.set_disabled("sam", True)
        with pytest.raises(ValueError, match="No such account"):
            admin_store.create_session("nobody")
        with pytest.raises(ValueError, match="No such account"):
            admin_store.create_session("sam")

    def test_sessions_for_hides_the_token_hash(self, admin_store):
        admin_store.create_session("jason")
        row = admin_store.sessions_for("jason")[0]
        assert "token_hash" not in row
        assert set(row) == {"id", "user", "client", "agent", "created_at",
                            "last_seen"}

    def test_last_seen_is_refreshed_at_most_once_a_minute(self, admin_store):
        token, session = admin_store.create_session("jason")
        first = session["last_seen"]
        admin_store.session_for_token(token)
        assert session["last_seen"] == first, "a second lookup rewrote last_seen"

        stale = (datetime.now(timezone.utc)
                 - timedelta(seconds=LAST_SEEN_REFRESH_SECONDS + 5)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
        session["last_seen"] = stale
        admin_store.session_for_token(token)
        assert session["last_seen"] > stale, "a stale session was never refreshed"
        # Back to "about now", which within one test run is the stamp the session
        # was minted with; the point is that it moved off the stale value.
        assert session["last_seen"] >= first

    def test_a_refused_lookup_does_not_count_as_activity(self, admin_store):
        token, session = admin_store.create_session("jason")
        session["last_seen"] = "2020-01-01T00:00:00Z"
        assert admin_store.session_for_token(token, touch=False) is not None
        assert session["last_seen"] == "2020-01-01T00:00:00Z"

    def test_the_cap_evicts_the_oldest(self, admin_store):
        ids = []
        for _ in range(MAX_SESSIONS_PER_USER + 3):
            _, session = admin_store.create_session("jason")
            ids.append(session["id"])
        live = [row["id"] for row in admin_store.sessions_for("jason")]
        assert len(live) == MAX_SESSIONS_PER_USER
        assert live == ids[3:], "the cap dropped something other than the oldest"

    def test_the_cap_is_per_account(self, admin_store):
        admin_store.add_user("sam", GOOD)
        for _ in range(MAX_SESSIONS_PER_USER + 2):
            admin_store.create_session("jason")
        admin_store.create_session("sam")
        assert len(admin_store.sessions_for("jason")) == MAX_SESSIONS_PER_USER
        assert len(admin_store.sessions_for("sam")) == 1

    def test_revoking_is_scoped_to_its_owner_when_asked(self, admin_store):
        admin_store.add_user("sam", GOOD)
        _, mine = admin_store.create_session("jason")
        _, theirs = admin_store.create_session("sam")
        assert admin_store.revoke_session(theirs["id"], user="jason") is False
        assert admin_store.revoke_session(theirs["id"], user="sam") is True
        assert admin_store.revoke_session(mine["id"]) is True
        assert admin_store.sessions == []
        assert admin_store.revoke_session(mine["id"]) is False
        assert admin_store.revoke_session("") is False

    def test_revoking_every_session_of_an_account(self, admin_store):
        admin_store.add_user("sam", GOOD)
        admin_store.create_session("jason")
        admin_store.create_session("jason")
        admin_store.create_session("sam")
        assert admin_store.revoke_sessions_for("jason") == 2
        assert admin_store.revoke_sessions_for("jason") == 0
        assert admin_store.revoke_sessions_for("nobody") == 0
        assert len(admin_store.sessions_for("sam")) == 1

    def test_a_session_survives_a_restart(self, admin_store, users_file):
        token, session = admin_store.create_session("jason")
        reopened = UsersStore.load()
        found = reopened.session_for_token(token)
        assert found is not None and found["id"] == session["id"]

    def test_public_helpers_never_leak(self):
        assert "token_hash" not in public_session({"token_hash": "x", "id": "1"})
        assert set(public_user({"name": "a"})) == {"name", "role", "created_at",
                                                  "disabled"}


# =============================================================================
# Live reload
# =============================================================================

class TestReload:
    def test_a_change_on_disk_is_adopted(self, store, users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        reader = UsersStore.load()
        assert reader.reload_if_changed() is False

        store.add_user("sam", GOOD)
        assert reader.reload_if_changed() is True
        assert {u["name"] for u in reader.list_users()} == {"jason", "sam"}
        assert reader.reload_if_changed() is False

    def test_a_missing_or_broken_file_keeps_the_state_in_memory(self, store,
                                                                users_file):
        store.add_user("jason", GOOD, ROLE_ADMIN)
        reader = UsersStore.load()

        users_file.write_text("{ not json")
        assert reader.reload_if_changed() is False
        assert reader.has_users() is True

        users_file.write_text('["a list, not an object"]')
        assert reader.reload_if_changed() is False
        assert reader.has_users() is True

        users_file.unlink()
        assert reader.reload_if_changed() is False
        assert reader.has_users() is True

    def test_load_raises_on_a_malformed_file(self, users_file):
        users_file.write_text("{ not json")
        with pytest.raises(ValueError):
            UsersStore.load()

    def test_load_of_a_missing_file_is_an_empty_store(self, users_file):
        empty = UsersStore.load()
        assert empty.has_users() is False
        assert empty.admin_count() == 0
        assert empty.list_users() == []


# =============================================================================
# Fleet replication
# =============================================================================

class TestReplication:
    def test_export_carries_the_hashes_and_is_sorted(self, admin_store):
        admin_store.add_user("sam", GOOD)
        rows = admin_store.export_users()
        assert [r["name"] for r in rows] == ["jason", "sam"]
        assert set(rows[0]) == {"name", "password_hash", "salt", "role",
                               "created_at", "disabled"}
        assert rows[0]["password_hash"] and rows[0]["salt"]

    def test_the_stamp_follows_the_content_and_not_the_file(self, admin_store,
                                                            users_file):
        stamp = admin_store.export_stamp()
        assert UsersStore.load().export_stamp() == stamp
        admin_store.add_user("sam", GOOD)
        assert admin_store.export_stamp() != stamp

    def test_import_replaces_the_list_and_lets_a_login_work(self, admin_store,
                                                            users_file,
                                                            tmp_path, monkeypatch):
        exported = admin_store.export_users()

        other = tmp_path / "peer-users.json"
        monkeypatch.setattr("ainode.auth.accounts.USERS_FILE", other)
        peer = UsersStore()
        assert peer.import_users(exported) is True
        assert peer.verify_password("jason", GOOD) is True, (
            "a replicated hash must verify without asking the head")
        assert peer.admin_count() == 1

    def test_import_of_the_same_list_changes_nothing(self, admin_store):
        assert admin_store.import_users(admin_store.export_users()) is False

    def test_import_keeps_the_sessions_whose_account_survived(self, admin_store):
        admin_store.add_user("sam", GOOD)
        mine, _ = admin_store.create_session("jason")
        theirs, _ = admin_store.create_session("sam")
        without_sam = [r for r in admin_store.export_users() if r["name"] != "sam"]
        assert admin_store.import_users(without_sam) is True
        assert admin_store.session_for_token(mine) is not None
        assert admin_store.session_for_token(theirs) is None

    @pytest.mark.parametrize("payload, match", [
        ("not a list", "must be a list"),
        (["not an object"], "JSON object"),
        ([{"name": "has space", "password_hash": "a", "salt": "b"}], "1 to 64"),
        ([{"name": "jason", "salt": "b"}], "no password hash"),
        ([{"name": "jason", "password_hash": "a"}], "no password hash"),
        ([{"name": "jason", "password_hash": "a", "salt": "b", "role": "root"}],
         "unknown role"),
        ([{"name": "jason", "password_hash": "a", "salt": "b"},
          {"name": "JASON", "password_hash": "c", "salt": "d"}], "twice"),
    ])
    def test_a_malformed_import_leaves_the_node_alone(self, admin_store, payload,
                                                     match):
        before = admin_store.export_users()
        with pytest.raises(ValueError, match=match):
            admin_store.import_users(payload)
        assert admin_store.export_users() == before

    def test_an_import_lands_on_disk(self, admin_store, users_file):
        admin_store.import_users([
            {"name": "sam", "password_hash": "a" * 64, "salt": "b" * 32,
             "role": ROLE_ADMIN, "created_at": "2026-01-01T00:00:00Z"},
        ])
        data = json.loads(users_file.read_text())
        assert [u["name"] for u in data["users"]] == ["sam"]


# =============================================================================
# require_admin
# =============================================================================

class TestRequireAdmin:
    def test_an_admin_session_may_administer(self):
        assert require_admin({"user": "jason", "user_role": ROLE_ADMIN,
                              "authenticated": True,
                              "api_key_id": "user:jason"}) is True

    def test_a_member_session_may_not(self):
        assert require_admin({"user": "sam", "user_role": ROLE_MEMBER,
                              "authenticated": True,
                              "api_key_id": "user:sam"}) is False

    def test_an_operator_key_may(self):
        """How the CLI works, and how the FIRST admin gets created."""
        assert require_admin({"user": "", "user_role": "",
                              "authenticated": True, "api_key_id": "abc123"}) is True

    def test_the_fleet_key_may(self):
        assert require_admin({"user": "", "user_role": "", "authenticated": True,
                              "api_key_id": FLEET_KEY_ID}) is True

    def test_nobody_may_not(self):
        assert require_admin({"user": "", "user_role": "", "authenticated": False,
                              "api_key_id": ""}) is False

    def test_a_stub_with_no_mapping_reads_as_not_an_admin(self):
        assert require_admin(object()) is False
