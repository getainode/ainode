"""The dashboard actually sends the key (#167).

Two halves. The first is text: no fetch in the UI may bypass the wrapper, and the
shell has to load it first, because one forgotten call site is one panel that
401s. The second runs ``static/js/auth.js`` under node with a stub fetch and a
stub storage and checks the behaviour that matters: the header is attached when a
key is stored and absent when it is not, a caller's own Authorization is left
alone, a 401 is reported once, and a browser that throws on localStorage still
works. auth.js is written DOM-free precisely so this is possible.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "ainode" / "web" / "static"
TEMPLATES = Path(__file__).parent.parent / "ainode" / "web" / "templates"
AUTH_JS = STATIC / "js" / "auth.js"

# Every file that talks to the API from a browser.
UI_SOURCES = [
    STATIC / "js" / "app.js",
    STATIC / "js" / "bench.js",
    STATIC / "js" / "join.js",
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
                  "/static/js/topology.js", "/static/js/join.js"):
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
