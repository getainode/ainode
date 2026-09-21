"""The dashboard draws the retained metrics, and draws them honestly.

Three halves, one question each.

**The shell.** The Metrics view is mounted: a nav pill, a view container, and the
two scripts in the order the page needs them (the shaping first, then the charts,
both after the auth wrapper). And the dead seam is gone: #234 left a 400 point
ring buffer in app.js that nothing drew, which is the shape #210 complained
about, and keeping it beside the store would be two sources for one line.

**The shaping, under node.** ``static/js/metrics-data.js`` is DOM-free on purpose,
so every rule that decides what a chart shows is exercised here with plain values
and no browser: which window and step to ask the store for, where the gaps are, a
counter turned into a rate across a restart, and an axis over a series that
measured nothing. The rule behind all of them is the one the store and the
collector already hold (root AGENTS.md, #215, #234): a figure nobody measured is
null, and null is a hole. Not a zero, not the previous sample carried forward,
not an interpolation across the gap. A zero on these charts reads as an idle GPU
on a node that is serving.

**The route.** ``/api/metrics/history?node=<id>`` answers for any node of the
cluster by asking THAT node, so the fleet's charts are each node's own
measurements. A node that cannot be reached is an error naming it, never an empty
grid: empty draws as a node that measured nothing, which is a different fact.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth.fleet import fleet_key_headers
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.metrics.api_routes import register_metrics_routes
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.store import (
    GPU_KEYS,
    LATENCY_KEYS,
    MAX_HISTORY_POINTS,
    REQUEST_KEYS,
    MetricsStore,
)
from ainode.web.serve import get_static_path

STATIC = get_static_path()
TEMPLATES = STATIC.parent / "templates"
APP_JS = STATIC / "js" / "app.js"
METRICS_JS = STATIC / "js" / "metrics.js"
DATA_JS = STATIC / "js" / "metrics-data.js"
STYLE_CSS = STATIC / "css" / "style.css"
INDEX = TEMPLATES / "index.html"

NODE = shutil.which("node")


# =============================================================================
# The shell: the view is mounted, and the seam that drew nothing is gone
# =============================================================================

def test_the_shell_mounts_the_metrics_view():
    html = INDEX.read_text()
    assert 'data-view="metrics"' in html
    assert 'id="view-metrics"' in html
    assert 'id="metrics-content"' in html


def test_the_scripts_load_in_the_order_the_page_needs():
    """The shaping before the charts, both after the auth wrapper."""
    html = INDEX.read_text()
    auth = html.index("/static/js/auth.js")
    data = html.index("/static/js/metrics-data.js")
    view = html.index("/static/js/metrics.js")
    assert auth < data < view


def test_app_js_owns_one_arm_and_nothing_else_of_the_view():
    app_js = APP_JS.read_text()
    assert "case 'metrics':" in app_js
    assert "window.AINodeMetrics.render(this)" in app_js
    # Every other line of the view lives in metrics.js, the same split bench.js
    # has. A panel, a canvas or a range in app.js would be the start of a second
    # home for this view.
    for owned in ("metrics-canvas", "metrics-panel", "metrics-range-pill"):
        assert owned not in app_js, f"{owned} belongs in metrics.js"


def test_the_ring_buffer_that_drew_nothing_is_gone():
    """#234's seam, removed now that the charts read the route it seeded from."""
    app_js = APP_JS.read_text()
    for dead in ("metricsHistory", "seedMetricsHistory", "pushMetricsPoint",
                 "METRICS_BUFFER_POINTS", "METRICS_SERIES"):
        assert dead not in app_js, f"{dead} is still in app.js with no reader"


def test_the_gpu_snapshot_poll_still_has_its_one_reader():
    """/api/metrics is still polled: the topology graphic draws the GPU block.

    The charts took over the request block and the history; they did not take
    over the live local-node ring, which reads the snapshot every three seconds.
    """
    app_js = APP_JS.read_text()
    assert "async pollMetrics()" in app_js
    assert "this.state.metrics = data" in app_js
    assert "this.state.metrics && this.state.metrics.gpu" in app_js


def test_the_view_fetches_through_the_auth_wrapper():
    """A bare fetch( is a request with no key on it (#167)."""
    bare = re.compile(r"(?<![\w.$])fetch\(")
    for path in (METRICS_JS, DATA_JS):
        hits = [n for n, line in enumerate(path.read_text().splitlines(), 1)
                if bare.search(line)]
        assert not hits, f"{path.name} calls fetch() directly on line(s) {hits}"
    assert "AINodeAuth.fetch(" in METRICS_JS.read_text()


def test_the_shaping_module_is_loadable_outside_a_browser():
    """No document, no window: the guards the node run below depends on."""
    text = DATA_JS.read_text()
    assert "typeof window !== 'undefined' ? window : globalThis" in text
    assert "module.exports" in text
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(("*", "//", "/*")))
    for dom in ("document.", "navigator.", "getComputedStyle", "canvas"):
        assert dom not in code, f"metrics-data.js must stay DOM-free: found {dom}"


def test_the_charts_read_the_design_tokens_rather_than_copying_them():
    """A second copy of the palette is a palette that goes stale."""
    view = METRICS_JS.read_text()
    assert "getComputedStyle" in view
    for token in ("--nvidia-green", "--cyan", "--amber", "--red",
                  "--text-secondary", "--text-muted", "--border"):
        assert token in view
    assert ".metrics-panel {" in STYLE_CSS.read_text()


def test_a_series_that_measured_nothing_is_said_in_words():
    """The utilization panel is the case that matters: a GB10 exposes none.

    Drawn as a flat zero it reports an idle GPU forever, which is the bug the
    whole telemetry-truth line of work exists to stop (#176).
    """
    view = METRICS_JS.read_text()
    assert "no GPU utilization figure" in view
    assert "not zero, not idle, not measured" in view
    assert "n/a" in view          # the legend, for the same series


def test_there_is_no_tokens_per_second_panel_while_the_counter_is_dead():
    """Nothing in the product ever increments the token counter.

    ``MetricsCollector.record_request`` takes ``tokens_generated`` and not one of
    its eight call sites passes it, so ``requests.tokens_generated`` and
    ``requests.tokens_per_second`` are 0 on every node forever. A panel over that
    draws a flat line at zero saying "this node generated no tokens", when the
    truth is "no code path counts tokens", and a chart may not say the first when
    it means the second. When the proxy passes the usage block through, the panel
    is one entry in SERIES and one in PANELS, and this test goes away.
    """
    import inspect

    from ainode.metrics.collector import MetricsCollector

    source = inspect.getsource(MetricsCollector)
    assert "tokens_generated: int = 0" in source, "the signature changed; recheck"

    view = METRICS_JS.read_text()
    assert "key: 'tokens'" not in view
    # And the reason is written down where the next person will look.
    assert "nothing counts tokens" in view
    # The series is not even asked for: the store answers it, and a payload full
    # of zeros invites exactly the panel this test exists to prevent.
    asked = _js_ranges()["series"]
    assert "requests.tokens_generated" not in asked
    assert "requests.tokens_per_second" not in asked


def test_nothing_new_carries_an_em_dash():
    """Brand rule, and it applies to the files this change adds.

    The character is built from its codepoint rather than typed, so the release
    gate (`git diff | grep '^+' | grep -c '<em dash>'`) does not count this file
    as a violation of the rule it enforces.
    """
    dash = chr(0x2014)
    for path in (METRICS_JS, DATA_JS):
        assert dash not in path.read_text(), f"{path.name} has an em dash"


# =============================================================================
# The series names are the store's names
# =============================================================================

def _js_ranges() -> dict:
    """RANGES and SERIES out of metrics-data.js, via node."""
    if NODE is None:
        pytest.skip("node is not on PATH")
    script = (
        "const D = require(process.env.AINODE_DATA_JS);"
        "console.log(JSON.stringify({ranges: D.RANGES, series: D.SERIES, "
        "points: D.RANGES.map(r => D.expectedPoints(r))}));"
    )
    proc = subprocess.run([NODE, "-e", script], capture_output=True, text=True,
                          timeout=60,
                          env={**os.environ, "AINODE_DATA_JS": str(DATA_JS)})
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_series_the_chart_asks_for_is_a_series_the_store_keeps():
    """A typo here is a panel that is silently empty forever."""
    known = set()
    for key in GPU_KEYS:
        known.add(f"gpu.{key}")
    for key in REQUEST_KEYS:
        known.add(f"requests.{key}")
    for key in LATENCY_KEYS:
        known.add(f"requests.latency_ms.{key}")
    known.add("uptime_seconds")

    asked = set(_js_ranges()["series"])
    unknown = asked - known
    assert not unknown, f"the chart asks for series the store never writes: {unknown}"


def test_every_range_fits_the_budget_the_store_splits_across_series(tmp_path):
    """The picker's step is the step the store will actually answer with.

    The store splits ONE point budget across every series asked for and coarsens
    the step when a request is over it, so a range whose points times series
    count exceeded the budget would be answered coarser than the picker says,
    silently. This asserts the arithmetic against the store's own constant, and
    then against the store itself.
    """
    dump = _js_ranges()
    per_series = MAX_HISTORY_POINTS // len(dump["series"])
    store = MetricsStore(tmp_path / "metrics.db")
    try:
        now = time.time()
        for spec, points in zip(dump["ranges"], dump["points"]):
            assert points <= per_series, (
                f"{spec['key']} asks for {points} points per series, "
                f"over the {per_series} the store allows for "
                f"{len(dump['series'])} series"
            )
            since, until, step, resolution = store.resolve_window(
                since=now - spec["seconds"], until=now,
                step=spec["step"], resolution=None, now=now,
                max_points=per_series,
            )
            assert step == spec["step"], (
                f"{spec['key']} asked for a {spec['step']}s step and the store "
                f"resolved {step}s"
            )
            assert resolution == spec["resolution"], (
                f"{spec['key']} declares {spec['resolution']} and the store "
                f"chooses {resolution}"
            )
    finally:
        store.close()


# =============================================================================
# The shaping, run under node
# =============================================================================

HARNESS = r"""
const assert = require('assert');
const D = require(process.argv[2]);

// A payload shaped exactly like /api/metrics/history's: an even grid, with null
// in every slot nothing was measured.
function payload(step, values) {
  const t0 = 1789000000;
  const series = {};
  Object.keys(values).forEach(function (name) {
    series[name] = values[name].map(function (v, i) {
      return { ts: t0 + i * step, value: v };
    });
  });
  return {
    since: t0, until: t0 + (values[Object.keys(values)[0]].length - 1) * step,
    step: step, resolution: 'raw',
    points: values[Object.keys(values)[0]].length, series: series,
  };
}

// -- the query: relative window, explicit step and resolution ---------------
var qs = D.historyQuery('6h', ['gpu.temperature_c'], null);
assert.ok(qs.indexOf('since=-21600s') !== -1, qs);
assert.ok(qs.indexOf('step=60s') !== -1, qs);
assert.ok(qs.indexOf('resolution=1m') !== -1, qs);
assert.ok(qs.indexOf('node=') === -1, 'no node asked for, none sent');
// A node id rides along url-encoded.
assert.ok(D.historyQuery('1h', ['a'], 'spark 3').indexOf('node=spark%203') !== -1);
// An unknown range key falls back to the first rather than asking for nothing.
assert.strictEqual(D.rangeFor('nope').key, D.RANGES[0].key);

// -- nulls survive the trip into points -------------------------------------
var p = D.toPoints(payload(15, { 'g': [1, null, 3] }), 'g');
assert.strictEqual(p.length, 3);
assert.strictEqual(p[1].v, null, 'a null slot became a number');
assert.strictEqual(p[0].t, 1789000000 * 1000, 'timestamps are milliseconds');
assert.deepStrictEqual(D.toPoints(payload(15, { 'g': [1] }), 'missing'), []);
// A series of nothing but nulls keeps its length: how much of the window the
// node was up for is itself information.
assert.strictEqual(D.toPoints(payload(15, { 'g': [null, null] }), 'g').length, 2);

// -- scaling keeps a null null ----------------------------------------------
var scaled = D.scalePoints(p, 1 / 1024);
assert.strictEqual(scaled[1].v, null);
assert.ok(Math.abs(scaled[0].v - 1 / 1024) < 1e-12);

// -- gaps are gaps ----------------------------------------------------------
var runs = D.segments(D.toPoints(payload(15, { 'g': [1, 2, null, 4, 5] }), 'g'), 40000);
assert.strictEqual(runs.length, 2, 'a null did not break the line');
assert.deepStrictEqual(runs.map(function (r) { return r.length; }), [2, 2]);
// One measured sample between two holes is still a measurement.
var lone = D.segments(D.toPoints(payload(15, { 'g': [null, 7, null] }), 'g'), 40000);
assert.strictEqual(lone.length, 1);
assert.strictEqual(lone[0].length, 1);
// A hole in the TIMESTAMPS breaks the line too, even with no null in it: the
// grid of a peer, or of a window spanning a restart, can skip.
var sparse = { since: 0, until: 300, step: 15, resolution: 'raw', points: 3,
               series: { g: [{ ts: 0, value: 1 }, { ts: 15, value: 2 },
                             { ts: 300, value: 3 }] } };
assert.strictEqual(D.segments(D.toPoints(sparse, 'g'), 40000).length, 2);
// No limit given: only nulls break it.
assert.strictEqual(D.segments(D.toPoints(sparse, 'g'), 0).length, 1);

// -- what is in a series ----------------------------------------------------
var s = D.summary(D.toPoints(payload(15, { 'g': [null, 2, 8, null, 4] }), 'g'));
assert.strictEqual(s.count, 5);
assert.strictEqual(s.measured, 3);
assert.strictEqual(s.missing, 2);
assert.strictEqual(s.min, 2);
assert.strictEqual(s.max, 8);
assert.strictEqual(s.first, 2);
assert.strictEqual(s.last, 4);
var none = D.summary(D.toPoints(payload(15, { 'g': [null, null, null] }), 'g'));
assert.strictEqual(none.measured, 0, 'a series of nulls claimed a measurement');
assert.strictEqual(none.last, null);
assert.strictEqual(none.min, null, 'a null series must not have a minimum of 0');

// -- not recording yet is not the same fact as a gap ------------------------
// A node that came up in the middle of the window has empty slots in front of
// its first sample, and nothing is wrong. Reporting that as "not measured" is
// true and useless; a hole AFTER the first sample is a real gap.
var window8 = payload(15, { 'uptime_seconds': [null, null, 1, 2, null, 4, 5, 6] });
var pts8 = D.toPoints(window8, 'uptime_seconds');
var started = pts8[2].t;
var cov = D.coverage(pts8, started);
assert.strictEqual(cov.slots, 8);
assert.strictEqual(cov.measured, 5);
assert.strictEqual(cov.missing, 3);
assert.strictEqual(cov.beforeStart, 2, 'the slots before the first sample are not gaps');
assert.strictEqual(cov.gaps, 1, 'the hole after it is');
// No start given: every hole counts as a gap, which is the conservative read.
assert.strictEqual(D.coverage(pts8, null).gaps, 3);
assert.strictEqual(D.coverage([], started).slots, 0);

// -- a counter becomes a rate, and a restart is not a negative one ----------
// 10 requests over 15 seconds is 40 a minute.
var counter = D.toPoints(payload(15, { 'r': [0, 10, 20, null, 40, 5, 15] }), 'r');
var perMinute = D.rate(counter, 60, 40000);
assert.strictEqual(perMinute[0].v, null, 'a rate needs two samples');
assert.strictEqual(perMinute[1].v, 40);
assert.strictEqual(perMinute[2].v, 40);
assert.strictEqual(perMinute[3].v, null, 'a null slot has no rate');
assert.strictEqual(perMinute[4].v, null, 'the interval after a hole has no rate');
assert.strictEqual(perMinute[5].v, null, 'a counter reset is not a negative rate');
assert.strictEqual(perMinute[6].v, 40, 'the rate resumes after the reset');
// Per second, for tokens.
assert.strictEqual(D.rate(counter, 1, 40000)[1].v, 10 / 15);
// A gap wider than the limit: the average would be spread over time nobody
// measured, so there is no rate for it.
var wide = { since: 0, until: 600, step: 15, resolution: 'raw', points: 2,
             series: { r: [{ ts: 0, value: 0 }, { ts: 600, value: 600 }] } };
assert.strictEqual(D.rate(D.toPoints(wide, 'r'), 1, 40000)[1].v, null);
assert.strictEqual(D.rate(D.toPoints(wide, 'r'), 1, 0)[1].v, 1);

// -- the axis --------------------------------------------------------------
assert.strictEqual(D.extent([D.toPoints(payload(15, { g: [null, null] }), 'g')]), null,
                   'an axis was invented for a series with no values');
var ext = D.extent([D.toPoints(payload(15, { g: [10, 30] }), 'g')], { zeroFloor: true, pad: 0 });
assert.strictEqual(ext.min, 0);
assert.strictEqual(ext.max, 30);
// A percentage axis is pinned to 0..100 so two nodes are comparable by eye.
var pct = D.extent([D.toPoints(payload(15, { g: [3, 4] }), 'g')],
                   { zeroFloor: true, min: 0, max: 100, pad: 0 });
assert.strictEqual(pct.min, 0);
assert.strictEqual(pct.max, 100);
// A line that never moves still gets a readable axis.
var flat = D.extent([D.toPoints(payload(15, { g: [41, 41] }), 'g')], { minSpan: 10, pad: 0 });
assert.ok(flat.max - flat.min >= 10, JSON.stringify(flat));
assert.ok(flat.min < 41 && flat.max > 41);

// -- ticks and labels ------------------------------------------------------
var t = D.ticks(0, 100, 4);
assert.ok(t.length >= 3 && t[0] >= 0 && t[t.length - 1] <= 100, JSON.stringify(t));
D.ticks(0, 1, 4).forEach(function (v) {
  // Repeated addition of 0.25 drifts without the re-round.
  assert.strictEqual(String(v).length <= 4, true, 'tick ' + v + ' drifted');
});
assert.deepStrictEqual(D.ticks(5, 5, 4), []);
assert.strictEqual(D.timeTicks(0, 1000, 5).length, 5);
assert.strictEqual(D.fmt(null), null, 'null formatted as a number');
assert.strictEqual(D.fmt(41.0, 0), '41');
assert.strictEqual(D.fmt(0.5, 1), '0.5');

// -- the window is the payload's, not the points' --------------------------
// A node up for two minutes of an hour draws two minutes against an hour of
// axis, rather than stretching to fill the panel.
var w = D.windowOf({ since: 100, until: 3700 }, '1h');
assert.strictEqual(w.from, 100000);
assert.strictEqual(w.to, 3700000);
assert.strictEqual(D.gapLimitMs({ step: 60 }, '1h'), 150000);
assert.strictEqual(D.gapLimitMs(null, '1h'), 37500);

console.log(JSON.stringify({ ok: true }));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_shaping_behaves_under_node(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS)
    proc = subprocess.run([NODE, str(harness), str(DATA_JS)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert json.loads(proc.stdout.strip().splitlines()[-1])["ok"] is True


# =============================================================================
# The route: one node's history, from any node
# =============================================================================

class _Resp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records the url and headers of every GET; answers from a map."""

    def __init__(self, answers=None, raises=None):
        self.answers = answers or {}
        self.raises = raises
        self.calls: list = []

    def get(self, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "headers": dict(headers or {})})
        if self.raises is not None:
            raise self.raises
        path = url.split("?")[0]
        status, payload = self.answers.get(path, (404, {"error": "not found"}))
        return _Resp(status, payload)


def _peer(node_id="spark3", name="Spark-3-DGX", ip="10.100.0.15", web_port=3000):
    return ClusterNode(
        node_id=node_id, node_name=name, gpu_name="NVIDIA GB10",
        gpu_memory_gb=128, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=web_port, last_seen=time.time(), fabric_ip=ip,
    )


def _cluster(nodes):
    state = ClusterState()
    for node in nodes:
        state.add_node(node)
    return state


@pytest.fixture
def store(tmp_path):
    s = MetricsStore(tmp_path / "metrics.db")
    yield s
    s.close()


def _app(store, session=None, cluster=None, cluster_secret=""):
    app = web.Application()
    app["config"] = SimpleNamespace(node_id="spark1", node_name="Spark-1-DGX",
                                    web_port=3000, cluster_secret=cluster_secret)
    if cluster is not None:
        app["cluster_state"] = cluster
    app["client_session"] = session
    register_metrics_routes(app, MetricsCollector(), store)
    return app


@pytest_asyncio.fixture
async def client(store):
    app = _app(store, session=FakeSession(), cluster=_cluster([_peer()]))
    async with TestClient(TestServer(app)) as c:
        yield c


PEER_URL = "http://10.100.0.15:3000/api/metrics/history"


@pytest.mark.asyncio
class TestHistoryPerNode:
    async def test_the_local_answer_says_whose_measurements_these_are(self, client):
        body = await (await client.get("/api/metrics/history?since=-5m&step=15")).json()
        assert body["node"] == {"node_id": "spark1", "node_name": "Spark-1-DGX",
                               "local": True}

    async def test_this_node_by_id_or_by_name_is_answered_locally(self, client, store):
        store.write({"gpu.temperature_c": 41.0})
        for name in ("spark1", "Spark-1-DGX", "SPARK-1-dgx"):
            resp = await client.get(f"/api/metrics/history?node={name}&since=-5m&step=15")
            body = await resp.json()
            assert resp.status == 200
            assert body["node"]["local"] is True
            assert "gpu.temperature_c" in body["series"]
        # Not one of those went out over the network.
        assert client.app["client_session"].calls == []

    async def test_a_peer_is_asked_for_its_own_history(self, store):
        peer_payload = {
            "since": 1.0, "until": 61.0, "step": 15.0, "resolution": "raw",
            "points": 5, "series": {"gpu.temperature_c": [{"ts": 1.0, "value": 44.0}]},
            "store": {"enabled": True, "degraded": False},
            "node": {"node_id": "spark3", "node_name": "Spark-3-DGX", "local": True},
        }
        session = FakeSession({PEER_URL: (200, peer_payload)})
        app = _app(store, session=session, cluster=_cluster([_peer()]),
                   cluster_secret="s3cret")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/metrics/history?node=spark3&series=gpu.temperature_c"
                "&since=-1h&step=15s&resolution=raw",
                headers={"Authorization": "Bearer the-operators-own-key"},
            )
            body = await resp.json()
        assert resp.status == 200
        # The peer's numbers, unchanged.
        assert body["series"]["gpu.temperature_c"][0]["value"] == 44.0
        # Whose they are, from the reader's point of view.
        assert body["node"] == {"node_id": "spark3", "node_name": "Spark-3-DGX",
                               "local": False}
        call = session.calls[0]
        assert call["url"].startswith(PEER_URL + "?")
        # The same window, and `node` is not passed on: it named the peer, and the
        # peer answering it would be asked about itself.
        assert "series=gpu.temperature_c" in call["url"]
        assert "since=-1h" in call["url"] and "step=15s" in call["url"]
        assert "node=" not in call["url"]
        # The FLEET key, not the browser's. A peer accepts the key derived from
        # the shared cluster_secret whatever its own operator key is (#244), so
        # this works on a fleet with auth on everywhere; forwarding whatever the
        # dashboard happened to send would only work where every node holds the
        # same operator key.
        assert call["headers"] == fleet_key_headers("s3cret")
        assert call["headers"]["Authorization"] != "Bearer the-operators-own-key"

    async def test_a_peer_is_asked_on_its_own_web_port(self, store):
        session = FakeSession({"http://10.100.0.15:3111/api/metrics/history":
                               (200, {"series": {}, "points": 0})})
        app = _app(store, session=session,
                   cluster=_cluster([_peer(web_port=3111)]))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
        assert resp.status == 200
        assert ":3111/" in session.calls[0]["url"]

    async def test_a_node_with_no_cluster_secret_sends_no_empty_header(self, store):
        """A fleet that never had a secret makes the request it always made."""
        session = FakeSession({PEER_URL: (200, {"series": {}, "points": 0})})
        app = _app(store, session=session, cluster=_cluster([_peer()]))
        async with TestClient(TestServer(app)) as client:
            await client.get("/api/metrics/history?node=spark3")
        assert "Authorization" not in session.calls[0]["headers"]

    async def test_a_node_this_cluster_has_never_heard_of_is_a_404(self, client):
        resp = await client.get("/api/metrics/history?node=spark9")
        body = await resp.json()
        assert resp.status == 404
        assert "spark9" in body["error"]
        assert body["series"] == {}
        assert body["node"]["local"] is False

    async def test_a_peer_that_does_not_answer_is_an_error_naming_it(self, store):
        """Never an empty grid: that would draw as a node that measured nothing."""
        session = FakeSession(raises=aiohttp.ClientError("connection refused"))
        app = _app(store, session=session, cluster=_cluster([_peer()]))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
            body = await resp.json()
        assert resp.status == 502
        assert "Spark-3-DGX" in body["error"]
        assert body["series"] == {}
        assert body["node"]["node_id"] == "spark3"

    async def test_a_peer_that_answers_an_error_is_reported_as_one(self, store):
        session = FakeSession({PEER_URL: (401, {"error": "this node requires an API key"})})
        app = _app(store, session=session, cluster=_cluster([_peer()]))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
            body = await resp.json()
        assert resp.status == 502
        assert "API key" in body["error"]

    async def test_a_node_with_no_reachable_address_says_so(self, store):
        session = FakeSession()
        app = _app(store, session=session,
                   cluster=_cluster([_peer(ip="", name="")]))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
            body = await resp.json()
        assert resp.status == 502
        assert "address" in body["error"]
        assert session.calls == []

    async def test_no_cluster_state_at_all_is_a_404_and_not_a_crash(self, store):
        app = _app(store, session=FakeSession())      # no cluster_state key
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
        assert resp.status == 404

    async def test_no_http_session_is_reported_rather_than_raised(self, store):
        app = _app(store, session=None, cluster=_cluster([_peer()]))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/metrics/history?node=spark3")
            body = await resp.json()
        assert resp.status == 502
        assert "session" in body["error"]


@pytest.mark.asyncio
async def test_retention_off_still_says_whose_node_it_is():
    app = web.Application()
    app["config"] = SimpleNamespace(node_id="spark1", node_name="Spark-1-DGX")
    register_metrics_routes(app, MetricsCollector())          # no store
    async with TestClient(TestServer(app)) as client:
        body = await (await client.get("/api/metrics/history")).json()
    assert body["store"]["enabled"] is False
    assert body["node"]["node_id"] == "spark1"
