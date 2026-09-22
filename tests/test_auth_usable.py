"""The dashboard actually sends the credential (#167), and a person signs in (#261).

Three halves, and the third is the front door. The first is text: no fetch in the
UI may bypass the wrapper, and the shell has to load it first, because one
forgotten call site is one panel that 401s. The second runs
``static/js/auth.js`` under node with a stub fetch and a stub storage and checks
the behaviour that matters: the header is attached when a key is stored and
absent when it is not, a caller's own Authorization is left alone, a 401 is
reported once, and a browser that throws on localStorage still works. auth.js is
written DOM-free precisely so this is possible.

The third runs the same file, plus ``static/js/signin.js``, over the boot
decision table a person actually meets: auth off, auth on with a session, auth on
with a stored key, auth on with neither, and auth on with no accounts at all.
It also pins the CSRF header on every request (the server ignores the session
cookie without it) and that a wrong password is answered on the sign-in screen
rather than by firing the "your session ended" handler at it.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "ainode" / "web" / "static"
TEMPLATES = Path(__file__).parent.parent / "ainode" / "web" / "templates"
AUTH_JS = STATIC / "js" / "auth.js"
SIGNIN_JS = STATIC / "js" / "signin.js"
APP_JS = STATIC / "js" / "app.js"

# Every file that talks to the API from a browser.
UI_SOURCES = [
    APP_JS,
    STATIC / "js" / "bench.js",
    STATIC / "js" / "join.js",
    STATIC / "js" / "metrics.js",
    SIGNIN_JS,
]


# =============================================================================
# Nothing bypasses the wrapper
# =============================================================================

def test_every_ui_fetch_goes_through_the_wrapper():
    """A bare fetch( is a request with no key on it."""
    import re
    # fetch( not preceded by a dot, a word character or a $ -- i.e. not
    # AINodeAuth.fetch(, not fetchJSON(, not this.fetch(.
    bare = re.compile(r"(?<![\w.$])fetch\(")
    for path in UI_SOURCES:
        text = path.read_text()
        hits = [n for n, line in enumerate(text.splitlines(), 1) if bare.search(line)]
        assert not hits, f"{path.name} calls fetch() directly on line(s) {hits}"


def test_the_wrapper_is_the_only_file_that_calls_fetch_itself():
    """auth.js holds the one call to the platform's fetch."""
    text = AUTH_JS.read_text()
    assert "fetchImpl" in text
    assert "global.fetch" in text


def test_the_shell_loads_the_wrapper_first():
    html = (TEMPLATES / "index.html").read_text()
    for later in ("/static/js/app.js", "/static/js/bench.js",
                  "/static/js/topology.js", "/static/js/join.js",
                  "/static/js/metrics.js", "/static/js/metrics-data.js",
                  "/static/js/signin.js"):
        assert html.index("/static/js/auth.js") < html.index(later)


def test_the_header_chip_and_the_panel_are_in_the_shell():
    html = (TEMPLATES / "index.html").read_text()
    assert 'id="api-access-chip"' in html
    assert 'id="api-chip-label"' in html
    assert 'data-section="api"' in html


def test_the_dashboard_says_the_api_is_open_in_the_same_words_as_the_api():
    """The header's wording is the server's wording (api/server.py)."""
    from ainode.api.server import auth_status_fields
    app_js = (STATIC / "js" / "app.js").read_text()
    assert auth_status_fields({})["label"] in app_js
    assert "API open, key set but not required" in app_js


def test_a_401_anywhere_opens_the_panel():
    app_js = (STATIC / "js" / "app.js").read_text()
    # One registration, one handler, and the handler ends on the panel.
    assert "AINodeAuth.onUnauthorized(" in app_js
    assert "onUnauthorized(info)" in app_js
    assert "openApiAccess()" in app_js
    assert "renderConfigApiAccess" in app_js


def test_the_bench_report_and_downloads_do_not_rely_on_a_bare_url():
    """An iframe src and an <a href> cannot carry the header (#167)."""
    bench = (STATIC / "js" / "bench.js").read_text()
    assert 'src="/api/bench/report"' not in bench
    assert 'href="/api/bench/results/' not in bench
    assert "srcdoc" in bench
    assert "data-bench-download" in bench


def test_the_key_is_never_put_in_a_url():
    """A key in a query string lands in logs and in the browser's history."""
    for path in UI_SOURCES + [AUTH_JS]:
        text = path.read_text()
        assert "api_key=" not in text
        assert "?key=" not in text


# =============================================================================
# The front door (#261)
# =============================================================================

def test_the_shell_holds_the_front_door_and_the_two_new_sections():
    html = (TEMPLATES / "index.html").read_text()
    # The screen, and the shell it replaces while it is up.
    assert 'id="signin-screen"' in html
    assert 'id="app-shell"' in html
    # Who is signed in, and the way out.
    assert 'id="user-chip"' in html
    assert 'id="sign-out"' in html
    # Where a person manages themselves, and where an admin manages everybody.
    assert 'data-section="account"' in html
    assert 'data-section="users"' in html
    # The screen is loaded before the file that decides to show it.
    assert html.index("/static/js/signin.js") < html.index("/static/js/app.js")


def test_the_boot_asks_who_is_at_the_keyboard_before_it_fetches_anything():
    """A shell of panels that all 401 is the bug, so nothing runs before this."""
    app_js = APP_JS.read_text()
    init = app_js[app_js.index("  init() {"):app_js.index("  start() {")]
    assert "AINodeAuth.loadMe()" in init
    assert "AINodeSignIn.show(" in init
    for later in ("startPolling", "this.refresh()", "fetchJSON", "initTopology"):
        assert later not in init, f"{later} runs before the front door decides"


def test_the_sections_are_routed_and_the_chip_speaks_for_the_person():
    app_js = APP_JS.read_text()
    assert "case 'account':" in app_js
    assert "case 'users':" in app_js
    # authChipText's four states, written from the user's side.
    for phrase in ("Signed in as ", "Using an API key",
                   "Open, no key required", "Sign-in required"):
        assert phrase in app_js, f"the auth chip never says {phrase!r}"


def test_the_api_access_copy_names_the_two_paths_the_front_door_opens():
    """/api/auth/login and /api/auth/me answer with no credential, so say so.

    tests/test_fleet_auth.py walks the whole of ``SKIP_PATHS`` against this same
    sentence; these two are named here as well so the copy cannot drift back
    while the backend half of #261 is landing.
    """
    app_js = APP_JS.read_text()
    section = [line for line in app_js.splitlines()
               if "config-section-desc" in line and "Who may call this node" in line]
    assert len(section) == 1, "the API access panel's description moved"
    for path in ("/api/auth/login", "/api/auth/me"):
        assert f"<code>{path}</code>" in section[0], f"{path} is keyless and unmentioned"
    # And the sentence says which credential belongs to whom.
    assert "A person signs in on the dashboard" in section[0]


def test_a_401_with_no_credential_goes_to_the_front_door_not_to_the_key_box():
    app_js = APP_JS.read_text()
    handler = app_js[app_js.index("  onUnauthorized(info) {"):app_js.index("  async signOut() {")]
    assert "requireSignIn(" in handler
    # A rejected KEY still lands on the panel that holds the key box.
    assert "openApiAccess()" in handler


def test_pasting_a_key_in_a_browser_is_collapsed_and_still_works():
    app_js = APP_JS.read_text()
    assert "Use a key in this browser instead" in app_js
    assert 'id="auth-key-input"' in app_js
    assert 'id="auth-key-save"' in app_js
    assert "Forget the stored key" in app_js


def test_the_switch_that_opens_the_port_is_shown_only_to_an_authenticated_caller():
    """#262: an unauthenticated browser pressing it got a 401 and a dead panel."""
    app_js = APP_JS.read_text()
    gate = app_js.index("if (st.authenticated || signedIn) {")
    button = app_js.index('id="auth-disable"')
    assert gate < button < gate + 600, "the disable button escaped its gate"


def test_sign_out_does_not_forget_a_program_key_nobody_asked_it_to():
    auth_js = AUTH_JS.read_text()
    out = auth_js[auth_js.index("    signOut() {"):]
    assert "clearKey" not in out.split("},")[0]


# =============================================================================
# The wrapper, run under node
# =============================================================================

NODE = shutil.which("node")

# Exercised as a browser would: a storage that can be made to throw, a fetch
# that records what it was called with, and the real auth.js in between.
HARNESS = r"""
const assert = require('assert');
const path = process.argv[2];

function freshStorage(mode) {
  var data = {};
  return {
    getItem(k) {
      if (mode === 'throw') throw new Error('site data blocked');
      return Object.prototype.hasOwnProperty.call(data, k) ? data[k] : null;
    },
    setItem(k, v) {
      if (mode === 'throw') throw new Error('site data blocked');
      data[k] = String(v);
    },
    removeItem(k) {
      if (mode === 'throw') throw new Error('site data blocked');
      delete data[k];
    },
    _data: data,
  };
}

function load() {
  delete require.cache[require.resolve(path)];
  return require(path);
}

function recorder(status) {
  var calls = [];
  var impl = function (url, options) {
    calls.push({ url: url, options: options });
    return Promise.resolve({ status: status || 200, ok: (status || 200) < 400 });
  };
  return { calls: calls, impl: impl };
}

(async function () {
  // -- no key stored: no Authorization header at all ------------------------
  let auth = load();
  auth.storage = freshStorage();
  let rec = recorder(200);
  auth.fetchImpl = rec.impl;
  await auth.fetch('/api/status');
  assert.strictEqual(auth.hasKey(), false);
  assert.ok(!('Authorization' in rec.calls[0].options.headers),
            'no key stored, yet a header was sent');

  // -- key stored: every request carries it --------------------------------
  auth.setKey('deadbeef');
  assert.strictEqual(auth.getKey(), 'deadbeef');
  await auth.fetch('/api/status');
  await auth.fetch('/api/models/unload', { method: 'POST' });
  assert.strictEqual(rec.calls[1].options.headers.Authorization, 'Bearer deadbeef');
  assert.strictEqual(rec.calls[2].options.headers.Authorization, 'Bearer deadbeef');
  assert.strictEqual(rec.calls[2].options.method, 'POST', 'the caller options survive');

  // -- the caller's own options and headers are not mutated -----------------
  var mine = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  await auth.fetch('/api/datasets', mine);
  assert.deepStrictEqual(mine.headers, { 'Content-Type': 'application/json' },
                         'the caller headers object was mutated');
  assert.strictEqual(rec.calls[3].options.headers['Content-Type'], 'application/json');
  assert.strictEqual(rec.calls[3].options.headers.Authorization, 'Bearer deadbeef');

  // -- a caller that sets its own Authorization keeps it --------------------
  await auth.fetch('/v1/chat/completions',
                   { headers: { Authorization: 'Bearer someone-elses' } });
  assert.strictEqual(rec.calls[4].options.headers.Authorization, 'Bearer someone-elses');

  // -- forgetting the key stops the header ---------------------------------
  auth.clearKey();
  await auth.fetch('/api/status');
  assert.ok(!('Authorization' in rec.calls[5].options.headers));

  // -- masking never shows the whole key -----------------------------------
  assert.strictEqual(auth.maskKey('0123456789abcdef'), '0123...cdef');
  assert.strictEqual(auth.maskKey(''), '');

  // -- a 401 is reported once per burst ------------------------------------
  auth = load();
  auth.storage = freshStorage();
  rec = recorder(401);
  auth.fetchImpl = rec.impl;
  var seen = [];
  auth.onUnauthorized(function (info) { seen.push(info); });
  await Promise.all([auth.fetch('/api/status'), auth.fetch('/api/nodes'),
                     auth.fetch('/api/models')]);
  assert.strictEqual(seen.length, 1, 'a 401 storm opened the panel ' + seen.length + ' times');
  assert.strictEqual(seen[0].hadKey, false);
  assert.strictEqual(auth.lastUnauthorized.url, '/api/models');

  // -- a 401 while a key IS stored says so, so the UI can say "stale key" --
  auth = load();
  auth.storage = freshStorage();
  auth.fetchImpl = recorder(401).impl;
  auth.setKey('stale');
  var told = null;
  auth.onUnauthorized(function (info) { told = info; });
  await auth.fetch('/api/status');
  assert.strictEqual(told.hadKey, true);

  // -- a 200 never reports anything ---------------------------------------
  auth = load();
  auth.storage = freshStorage();
  auth.fetchImpl = recorder(200).impl;
  var fired = false;
  auth.onUnauthorized(function () { fired = true; });
  await auth.fetch('/api/status');
  assert.strictEqual(fired, false);

  // -- a storage that throws on every access still works for this page load -
  auth = load();
  auth.storage = freshStorage('throw');
  rec = recorder(200);
  auth.fetchImpl = rec.impl;
  assert.strictEqual(auth.getKey(), '');
  auth.setKey('in-memory-only');
  assert.strictEqual(auth.getKey(), 'in-memory-only');
  await auth.fetch('/api/status');
  assert.strictEqual(rec.calls[0].options.headers.Authorization, 'Bearer in-memory-only');

  // -- no storage at all (an artifact viewer, a sandbox) -------------------
  auth = load();
  auth.storage = null;
  assert.strictEqual(auth.getKey(), '');
  auth.setKey('still-fine');
  assert.strictEqual(auth.getKey(), 'still-fine');

  // -- the storage key is the documented one ------------------------------
  auth = load();
  var store = freshStorage();
  auth.storage = store;
  auth.setKey('written');
  assert.strictEqual(store._data['ainode.apiKey'], 'written');

  console.log(JSON.stringify({ ok: true, calls: rec.calls.length }));
})().catch(function (e) {
  console.error(e && e.stack || String(e));
  process.exit(1);
});
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_wrapper_behaves_under_node(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS)
    proc = subprocess.run(
        [NODE, str(harness), str(AUTH_JS)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert json.loads(proc.stdout.strip().splitlines()[-1])["ok"] is True


# =============================================================================
# The front door, run under node
# =============================================================================

# The decision table a person meets on load, plus the CSRF header the server
# needs before it will look at the session cookie at all. Both halves live in
# auth.js so they can be exercised with no DOM; signin.js is loaded too, because
# the screen's copy (the CLI command, the autocomplete hints) is a pure function.
FRONT_DOOR_HARNESS = r"""
const assert = require('assert');
const authPath = process.argv[2];
const signinPath = process.argv[3];

function load(p) {
  delete require.cache[require.resolve(p)];
  return require(p);
}

function freshStorage() {
  var data = {};
  return {
    getItem(k) { return Object.prototype.hasOwnProperty.call(data, k) ? data[k] : null; },
    setItem(k, v) { data[k] = String(v); },
    removeItem(k) { delete data[k]; },
  };
}

/** A fetch that answers per URL, and records every call. */
function router(routes) {
  var calls = [];
  var impl = function (url, options) {
    calls.push({ url: url, options: options || {} });
    var route = routes[url] || routes['*'] || { status: 404 };
    var status = route.status || 200;
    return Promise.resolve({
      status: status,
      ok: status < 400,
      headers: { get: function (name) { return (route.headers || {})[name] || null; } },
      json: function () {
        if (route.body === undefined) return Promise.reject(new Error('no json'));
        return Promise.resolve(route.body);
      },
    });
  };
  return { calls: calls, impl: impl };
}

(async function () {
  // -- the CSRF header rides on every request, GET and write ----------------
  var auth = load(authPath);
  auth.storage = freshStorage();
  var rec = router({ '*': { status: 200, body: {} } });
  auth.fetchImpl = rec.impl;
  await auth.fetch('/api/status');
  await auth.fetch('/api/models/unload', { method: 'POST' });
  await auth.fetch('/api/auth/sessions/s1', { method: 'DELETE' });
  rec.calls.forEach(function (c, i) {
    assert.strictEqual(c.options.headers['X-AINode-Client'], 'dashboard',
                       'request ' + i + ' would have its session cookie ignored');
    assert.strictEqual(c.options.credentials, 'same-origin',
                       'request ' + i + ' would not send the session cookie');
  });
  // A caller that stamps its own client name keeps it (the desktop app).
  await auth.fetch('/api/status', { headers: { 'X-AINode-Client': 'desktop' } });
  assert.strictEqual(rec.calls[3].options.headers['X-AINode-Client'], 'desktop');

  // -- the boot decision table ---------------------------------------------
  auth = load(authPath);
  auth.storage = freshStorage();
  var open = { user: null, auth_enabled: false, has_users: true };
  var session = { user: { name: 'jason', role: 'admin', session: { id: 's1' } },
                  auth_enabled: true, has_users: true };
  var locked = { user: null, auth_enabled: true, has_users: true };
  var empty = { user: null, auth_enabled: true, has_users: false };

  assert.strictEqual(auth.bootDecision(open).show, 'app', 'an open node asked for a password');
  assert.strictEqual(auth.bootDecision(session).show, 'app', 'a signed-in person was sent to the door');
  assert.deepStrictEqual(auth.bootDecision(locked), { show: 'signin', state: 'form' });
  assert.deepStrictEqual(auth.bootDecision(empty), { show: 'signin', state: 'no-accounts' });
  // A stored key is a program's credential and still gets the app.
  auth.setKey('program-key');
  assert.deepStrictEqual(auth.bootDecision(locked), { show: 'app', state: 'key' });
  auth.clearKey();
  // A release with no front door at all: behave exactly as before.
  assert.strictEqual(auth.bootDecision(null).show, 'app');
  assert.strictEqual(auth.needsSignIn(null), false);

  // -- /api/auth/me is read once, and a 404 reads as "no front door" --------
  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({ '/api/auth/me': { status: 404 } });
  auth.fetchImpl = rec.impl;
  assert.strictEqual(await auth.loadMe(), null);
  assert.strictEqual(auth.needsSignIn(), false);

  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({ '/api/auth/me': { status: 200, body: locked } });
  auth.fetchImpl = rec.impl;
  var me = await auth.loadMe();
  assert.strictEqual(me.auth_enabled, true);
  assert.strictEqual(auth.needsSignIn(), true);
  assert.strictEqual(auth.user(), null);

  // -- signing in ----------------------------------------------------------
  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({
    '/api/auth/login': { status: 200, body: { user: { name: 'jason', role: 'admin' }, session_id: 's1' } },
    '/api/auth/me': { status: 200, body: session },
  });
  auth.fetchImpl = rec.impl;
  var out = await auth.signIn('  jason  ', 'hunter2');
  assert.strictEqual(out.ok, true);
  assert.strictEqual(auth.user().name, 'jason');
  assert.strictEqual(auth.isAdmin(), true);
  assert.strictEqual(rec.calls[0].url, '/api/auth/login');
  assert.strictEqual(rec.calls[0].options.method, 'POST');
  assert.strictEqual(rec.calls[0].options.headers['X-AINode-Client'], 'dashboard');
  assert.deepStrictEqual(JSON.parse(rec.calls[0].options.body),
                         { name: 'jason', password: 'hunter2' });
  // /api/auth/me is the authority on who is signed in, so it is re-read.
  assert.strictEqual(rec.calls[1].url, '/api/auth/me');

  // -- a wrong password is answered on the screen, not by the 401 handler ---
  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({ '/api/auth/login': { status: 401, body: { error: { message: 'Wrong name or password' } } } });
  auth.fetchImpl = rec.impl;
  var fired = 0;
  auth.onUnauthorized(function () { fired++; });
  out = await auth.signIn('jason', 'wrong');
  assert.strictEqual(out.ok, false);
  assert.strictEqual(out.message, 'Wrong name or password');
  assert.strictEqual(fired, 0, 'a wrong password fired the session-ended handler');

  // -- 409, 429 and an empty form all say what to do -----------------------
  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({ '/api/auth/login': { status: 409, body: { error: { message: 'No accounts on this node. Run ainode auth user add jason --admin' } } } });
  auth.fetchImpl = rec.impl;
  out = await auth.signIn('jason', 'x');
  assert.strictEqual(out.noAccounts, true);
  assert.ok(out.message.indexOf('ainode auth user add') >= 0);

  assert.strictEqual(auth.signInError(429, null, '45'),
                     'Too many tries. Wait 45 seconds and sign in again.');
  assert.strictEqual(auth.signInError(429, null, '1'),
                     'Too many tries. Wait 1 second and sign in again.');
  assert.ok(auth.signInError(429, null, null).indexOf('Wait 30 seconds') >= 0);
  assert.ok(auth.signInError(409, null).indexOf('ainode auth user add <name> --admin') >= 0);
  assert.strictEqual(auth.signInError(401, null), 'Wrong name or password.');
  out = await auth.signIn('', '');
  assert.strictEqual(out.message, 'Enter your name and password.');
  assert.strictEqual(auth.userAddCommand('jason'), 'ainode auth user add jason --admin');

  // -- a node that cannot be reached says so -------------------------------
  auth = load(authPath);
  auth.storage = freshStorage();
  auth.fetchImpl = function () { return Promise.reject(new Error('offline')); };
  out = await auth.signIn('jason', 'hunter2');
  assert.strictEqual(out.ok, false);
  assert.ok(out.message.indexOf('Could not reach this node') >= 0);

  // -- signing out ends the session and leaves a program's key alone -------
  auth = load(authPath);
  auth.storage = freshStorage();
  auth.setKey('program-key');
  rec = router({ '/api/auth/logout': { status: 200, body: {} } });
  auth.fetchImpl = rec.impl;
  auth.me = session;
  assert.strictEqual(await auth.signOut(), true);
  assert.strictEqual(auth.user(), null);
  assert.strictEqual(auth.getKey(), 'program-key',
                     'sign out forgot a key nobody asked it to forget');
  assert.strictEqual(rec.calls[0].options.method, 'POST');

  // -- a 401 mid-session says WHICH credential failed ----------------------
  auth = load(authPath);
  auth.storage = freshStorage();
  rec = router({ '*': { status: 401 } });
  auth.fetchImpl = rec.impl;
  auth.me = session;
  var seen = null;
  auth.onUnauthorized(function (info) { seen = info; });
  await auth.fetch('/api/status');
  assert.strictEqual(seen.hadSession, true);
  assert.strictEqual(seen.hadKey, false);

  // -- the screen's own copy ----------------------------------------------
  global.AINodeAuth = auth;
  var screen = load(signinPath);
  var form = screen.screenHTML({ state: 'form', reason: '' });
  assert.ok(form.indexOf('autocomplete="username"') >= 0);
  assert.ok(form.indexOf('autocomplete="current-password"') >= 0);
  assert.ok(form.indexOf('Made in Texas') >= 0);
  assert.ok(form.indexOf('Config &gt; API access') >= 0, 'the screen never mentions keys');
  assert.ok(form.indexOf('id="signin-error" role="alert" hidden') >= 0,
            'the error line is visible before anything went wrong');
  var withReason = screen.screenHTML({ state: 'form', reason: 'This node ended your session. Sign in again.' });
  assert.ok(withReason.indexOf('This node ended your session.') >= 0);
  var none = screen.screenHTML({ state: 'no-accounts' });
  assert.ok(none.indexOf('ainode auth user add &lt;name&gt; --admin') >= 0,
            'the no-accounts screen does not show the command');
  assert.ok(none.indexOf('type="password"') < 0, 'nothing to type, yet a password box');

  console.log(JSON.stringify({ ok: true }));
})().catch(function (e) {
  console.error(e && e.stack || String(e));
  process.exit(1);
});
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_front_door_decides_under_node(tmp_path):
    harness = tmp_path / "front-door.js"
    harness.write_text(FRONT_DOOR_HARNESS)
    proc = subprocess.run(
        [NODE, str(harness), str(AUTH_JS), str(SIGNIN_JS)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert json.loads(proc.stdout.strip().splitlines()[-1])["ok"] is True


def test_the_wrapper_is_loadable_outside_a_browser():
    """No document, no window: the guards a node run depends on."""
    text = AUTH_JS.read_text()
    assert "typeof window !== 'undefined' ? window : globalThis" in text
    assert "module.exports" in text
    # DOM-free in the code, whatever the comments say about the DOM.
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(("*", "//", "/*")))
    for dom in ("document.", "navigator.", "alert(", "window.location"):
        assert dom not in code, f"auth.js must stay DOM-free: found {dom}"
