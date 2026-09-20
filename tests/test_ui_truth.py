"""Every launch and unload button does what it says.

One file for the round of UI-truth fixes, because they share one question: does
the thing the button claims to do actually happen, on the instance the user was
looking at?

  * #197 a forwarded request pins the picked instance (proxy + bench resolver)
  * #207 UNLOAD stops the copy on the named node and port, and only that one
  * #209 /api/models/downloaded is registered before /api/models/{model_id}
  * #205 the ModelManager reads the SAME directory the engine mounts
  * #204 cors_origins and the training directories are read, not just saved

The two JS behaviours (the chat request carrying the pin, and a full
localStorage not taking chat persistence down with it) are exercised by running
app.js under node against a stub DOM at the bottom of this file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import textwrap
from pathlib import Path

import pytest

import ainode.api.server as server
import ainode.models.api_routes as mr
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager
from ainode.web.serve import get_static_path

MODEL = "cerebras/Ornith-1.5-35B-A3B-NVFP4"


# ------------------------------------------------------------------ harness --

def _node(nid, name=None, model=MODEL, fabric="", api_port=8000, web_port=3000,
          instances=None):
    return ClusterNode(node_id=nid, node_name=name or nid, gpu_name="NVIDIA GB10",
                       gpu_memory_gb=121.7, unified_memory=True, model=model,
                       status=NodeStatus.ONLINE, api_port=api_port, web_port=web_port,
                       last_seen=0.0, fabric_ip=fabric, instances=instances or [])


def _cluster(nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return c


class _Collector:
    def __init__(self):
        self.calls: list = []

    def record_request(self, model, ms, error=False):
        self.calls.append((model, error))


class _Up:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload if payload is not None else {"ok": 1}

    async def read(self):
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Records every forwarded request, answers 200 to all of them."""

    def __init__(self):
        self.tried: list = []
        self.bodies: list = []

    def request(self, method, url, **kwargs):
        self.tried.append(url)
        raw = kwargs.get("data")
        self.bodies.append(json.loads(raw) if raw else None)
        return _Up()


def _proxy_app(cluster, session=None):
    return {
        "config": NodeConfig(node_id="spark1", node_name="Spark-1-DGX",
                             api_port=8000, web_port=3000, model=MODEL),
        "cluster_state": cluster,
        "client_session": session or _Session(),
        "metrics_collector": _Collector(),
    }


def _proxy(app, body, headers=None, path="/v1/chat/completions"):
    class _R:
        method = "POST"

        def __init__(self):
            self.app = app
            self.path = path
            self.path_qs = path
            self.headers = headers or {}

        async def read(self):
            return json.dumps(body).encode()

    resp = asyncio.run(server.proxy_to_vllm(_R()))
    return resp


def _two_nodes():
    return _cluster([
        _node("spark1", "Spark-1-DGX"),                         # local, :8000
        _node("spark3", "Spark-3-DGX", fabric="10.100.0.15"),   # remote, :8000
    ])


# ------------------------------------------- #197 the proxy honors a pin --

def test_a_pin_is_the_only_candidate_tried():
    """Two nodes serve the id; the pin picks the remote and nothing else is tried."""
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    resp = _proxy(app, {"model": MODEL, "messages": []},
                  headers={"X-AINode-Node": "spark3"})
    assert resp.status == 200
    assert session.tried == ["http://10.100.0.15:8000/v1/chat/completions"]


def test_without_a_pin_the_local_hop_still_goes_first():
    """The unpinned order is unchanged: local first, peers as failover."""
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    _proxy(app, {"model": MODEL, "messages": []})
    assert session.tried[0] == "http://localhost:8000/v1/chat/completions"


def test_a_pin_by_body_field_is_honored_and_stripped_before_forwarding():
    """An API caller that cannot set headers pins in the body; vLLM never sees it."""
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    _proxy(app, {"model": MODEL, "messages": [],
                 "ainode_target": {"node_id": "spark3", "port": 8001}})
    assert session.tried == ["http://10.100.0.15:8001/v1/chat/completions"]
    assert "ainode_target" not in session.bodies[0]
    assert session.bodies[0]["model"] == MODEL


def test_a_pin_by_body_string_takes_node_and_port():
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    _proxy(app, {"model": MODEL, "messages": [], "ainode_target": "spark3:8002"})
    assert session.tried == ["http://10.100.0.15:8002/v1/chat/completions"]


def test_a_port_only_pin_reaches_a_stacked_local_instance():
    """No node id: the pin is a port on this node, which is how a co-resident
    second model (:8001) is addressed at all."""
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    _proxy(app, {"model": MODEL, "messages": []}, headers={"X-AINode-Port": "8001"})
    assert session.tried == ["http://localhost:8001/v1/chat/completions"]


def test_a_pin_on_an_unknown_node_is_a_404_naming_it():
    """Falling back to the local hop is exactly the mis-attribution the pin
    prevents, so an unresolvable pin is an error the caller can see."""
    session = _Session()
    app = _proxy_app(_two_nodes(), session)
    resp = _proxy(app, {"model": MODEL, "messages": []},
                  headers={"X-AINode-Node": "ghost"})
    assert resp.status == 404
    payload = json.loads(resp.body.decode())
    assert payload["error"]["code"] == "unknown_node"
    assert "ghost" in payload["error"]["message"]
    assert session.tried == []


def test_the_response_names_the_instance_that_answered():
    app = _proxy_app(_two_nodes())
    resp = _proxy(app, {"model": MODEL, "messages": []},
                  headers={"X-AINode-Node": "spark3"})
    assert resp.headers["X-AINode-Served-By"] == "10.100.0.15:8000"


def test_pinned_target_reads_either_spelling():
    assert server.pinned_target({"X-AINode-Node": "n1", "X-AINode-Port": "8001"}, {}) \
        == ("n1", 8001)
    assert server.pinned_target({}, {"ainode_target": {"node_id": "n2"}}) == ("n2", None)
    assert server.pinned_target({}, {}) == (None, None)
    # A header wins over the body, and an unparseable port is no port at all.
    assert server.pinned_target({"X-AINode-Node": "n1"},
                                {"ainode_target": {"node_id": "n2"}}) == ("n1", None)
    assert server.pinned_target({"X-AINode-Port": "eight"}, {}) == (None, None)


# ---------------------------------------------- #197 the bench resolver --

def test_the_bench_resolves_the_picked_instance_not_the_first_candidate():
    from ainode.bench.fleet import resolve_target

    app = _proxy_app(_two_nodes())
    default = resolve_target(app, MODEL)
    assert (default.host, default.port) == ("localhost", 8000)

    picked = resolve_target(app, MODEL, node_id="spark3", port=8001)
    assert (picked.host, picked.port) == ("10.100.0.15", 8001)
    assert picked.node_id == "spark3"
    assert picked.node_name == "Spark-3-DGX"

    assert resolve_target(app, MODEL, node_id="ghost") is None


# -------------------------------------------------- #207 a scoped unload --

class _Backend:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def is_running(self):
        return not self.stopped


class _UnloadReq:
    def __init__(self, app, body, query=None):
        self.app = app
        self._b = body
        self.query = query or {}

    async def json(self):
        return self._b


def _stacked_app(model_a=MODEL, model_b=MODEL):
    """A node with two instances: :8000 and :8001. Same model id by default,
    which is the case the port exists to disambiguate."""
    cfg = NodeConfig(node_id="spark1", api_port=8000, model=model_a)
    cfg.save = lambda: None
    mgr = InstanceManager(base_port=8000)
    backends = {}
    for port, model in ((8000, model_a), (8001, model_b)):
        backend = _Backend()
        backends[port] = backend
        mgr.add(InstanceRecord(instance_id=f"spark1:{port}", model=model,
                               head_node_id="spark1", api_port=port), backend)
    app = {"config": cfg, "instances": mgr, "engine": backends[8000],
           "cluster_state": _cluster([_node("spark1")]), "client_session": None}
    return app, mgr, backends


def test_unload_stops_the_instance_on_the_named_port_only(monkeypatch):
    monkeypatch.setattr(mr, "save_instance_manifest", lambda app: None)
    app, mgr, backends = _stacked_app()
    resp = asyncio.run(mr.handle_model_unload(
        _UnloadReq(app, {"model": MODEL, "api_port": 8001})))
    body = json.loads(resp.body)
    assert body["stopped"] is True
    assert body["instance_id"] == "spark1:8001"
    assert backends[8001].stopped is True
    assert backends[8000].stopped is False          # the copy nobody asked about
    assert mgr.by_port(8000) is not None
    assert mgr.by_port(8001) is None


def test_unload_without_a_port_still_stops_one_instance(monkeypatch):
    """Back-compat: a bare {model} is a node-scoped unload of that model."""
    monkeypatch.setattr(mr, "save_instance_manifest", lambda app: None)
    app, mgr, backends = _stacked_app(model_a=MODEL, model_b="other/model")
    resp = asyncio.run(mr.handle_model_unload(_UnloadReq(app, {"model": "other/model"})))
    assert json.loads(resp.body)["stopped"] is True
    assert backends[8001].stopped is True
    assert backends[8000].stopped is False


def test_a_port_that_matches_nothing_does_not_stop_a_neighbour(monkeypatch):
    monkeypatch.setattr(mr, "save_instance_manifest", lambda app: None)
    app, mgr, backends = _stacked_app()
    resp = asyncio.run(mr.handle_model_unload(
        _UnloadReq(app, {"model": MODEL, "api_port": 8099})))
    assert json.loads(resp.body)["stopped"] is False
    assert backends[8000].stopped is False
    assert backends[8001].stopped is False


class _FanoutSession:
    def __init__(self):
        self.posts: list = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))

        class _R:
            status = 200

            async def json(self_inner):
                return {"stopped": True, "errors": []}

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False
        return _R()


def _peer_app():
    cfg = NodeConfig(node_id="spark1", api_port=8000, model="")
    cfg.save = lambda: None
    session = _FanoutSession()
    app = {"config": cfg, "engine": None, "instances": InstanceManager(base_port=8000),
           "cluster_state": _cluster([_node("spark1"),
                                      _node("spark3", fabric="10.100.0.15")]),
           "client_session": session}
    return app, session


def test_unload_does_not_fan_out_to_peers_by_default():
    """The bug: a bare {model} on a node not serving it used to unload every copy
    in the fleet, so UNLOAD on one instance card took down all of them (#207)."""
    app, session = _peer_app()
    resp = asyncio.run(mr.handle_model_unload(_UnloadReq(app, {"model": MODEL})))
    body = json.loads(resp.body)
    assert body["scope"] == "local-miss"
    assert body["stopped"] is False
    assert session.posts == []


def test_unload_fans_out_only_when_the_caller_asks_for_all():
    app, session = _peer_app()
    resp = asyncio.run(mr.handle_model_unload(_UnloadReq(app, {"model": MODEL, "all": True})))
    body = json.loads(resp.body)
    assert body["scope"] == "remote-fanout"
    assert [u for u, _ in session.posts] == [
        "http://10.100.0.15:3000/api/models/unload?fanout=0"]


def test_a_fanout_child_never_recurses_even_with_all():
    app, session = _peer_app()
    asyncio.run(mr.handle_model_unload(
        _UnloadReq(app, {"model": MODEL, "all": True}, query={"fanout": "0"})))
    assert session.posts == []


def test_the_ui_unload_route_carries_the_node_and_port_to_that_node(monkeypatch):
    """The whole path: /api/cluster/unload {node_id, model, api_port} forwards the
    port to the node that owns the card and strips only node_id."""
    posted = {}

    class _Up2:
        status = 200
        headers = {"Content-Type": "application/json"}

        async def read(self):
            return b'{"stopped": true}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Sess:
        def post(self, url, json=None, timeout=None):
            posted["url"] = url
            posted["json"] = json
            return _Up2()

    app = {"config": NodeConfig(node_id="spark1", api_port=8000),
           "cluster_state": _cluster([_node("spark1"),
                                      _node("spark3", fabric="10.100.0.15")]),
           "client_session": _Sess()}
    resp = asyncio.run(server.handle_cluster_unload(
        _UnloadReq(app, {"node_id": "spark3", "model": MODEL, "api_port": 8001})))
    assert resp.status == 200
    assert posted["url"] == "http://10.100.0.15:3000/api/models/unload"
    assert posted["json"] == {"model": MODEL, "api_port": 8001}


# ------------------------------------------------------- #209 route order --

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _real_app(tmp_path, **overrides):
    """A real app, hermetic: its own models / datasets / training directories and
    no startup replay.

    Both halves matter on a developer machine. The replay sweeps engine containers
    and relaunches the node's persisted models, which is the docker boundary a unit
    test must never reach (tests/conftest.py fails the session on a leaked
    ``ainode-vllm*`` container), and the TrainingManager rehydrates every job dir
    under the training root, which on a machine with a real ~/.ainode is thousands
    of them.
    """
    fields = {"node_id": "n1", "model": "", "api_port": _free_port(),
              "models_dir": str(tmp_path / "models"),
              "datasets_dir": str(tmp_path / "datasets"),
              "training_dir": str(tmp_path / "runs")}
    fields.update(overrides)
    config = NodeConfig(**fields)
    config._skip_replay = True
    return server.create_app(config=config, engine=None)


@pytest.fixture(autouse=True)
def _restore_training_dirs():
    """``ainode.training.engine`` holds its configured directories in module state,
    so an app built here must not decide where a later test looks."""
    yield
    import ainode.training.engine as te

    te.configure_dirs(NodeConfig())


def test_downloaded_is_registered_before_the_dynamic_model_route(tmp_path):
    """Registration ORDER, not just today's aiohttp resolution: the literal has to
    come first or /api/models/downloaded answers "Model 'downloaded' not found"
    and the Installed list, the launch dropdown and the on-disk cards go empty."""
    app = _real_app(tmp_path)
    paths = [r.resource.canonical for r in app.router.routes()
             if r.method == "GET" and r.resource is not None]
    assert "/api/models/downloaded" in paths
    assert paths.index("/api/models/downloaded") < paths.index("/api/models/{model_id}")


@pytest.mark.asyncio
async def test_downloaded_answers_a_list_through_the_real_app(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    async with TestClient(TestServer(_real_app(tmp_path))) as client:
        resp = await client.get("/api/models/downloaded")
        assert resp.status == 200
        assert isinstance((await resp.json())["models"], list)


# ------------------------------------------------------- #205 models_dir --

def test_the_model_manager_reads_the_configured_models_dir(tmp_path):
    """The manager and the engine mount have to be the same directory: they were
    not, so a download landed where no launch would look for it."""
    store = tmp_path / "weights"
    app = _real_app(tmp_path, models_dir=str(store))
    manager = app["model_manager"]
    assert Path(manager.models_dir) == store
    assert Path(manager.models_dir) == Path(app["config"].models_dir)
    assert store.is_dir()


# --------------------------------------------------- #204 cors_origins --

def test_cors_always_allows_localhost():
    cfg = NodeConfig(cors_origins=None)
    assert server.cors_allowed_origin(cfg, "http://localhost:3000") == "http://localhost:3000"
    assert server.cors_allowed_origin(cfg, "http://127.0.0.1:3000") == "http://127.0.0.1:3000"


def test_cors_refuses_an_origin_nobody_configured():
    assert server.cors_allowed_origin(NodeConfig(), "https://evil.example") == ""


def test_cors_allows_a_configured_origin_and_only_that_one():
    cfg = NodeConfig(cors_origins="https://fleet.example, https://lab.example")
    assert server.cors_allowed_origin(cfg, "https://fleet.example") == "https://fleet.example"
    assert server.cors_allowed_origin(cfg, "https://lab.example") == "https://lab.example"
    assert server.cors_allowed_origin(cfg, "https://other.example") == ""


def test_cors_star_allows_any_origin():
    cfg = NodeConfig(cors_origins="*")
    assert server.cors_allowed_origin(cfg, "https://anything.example") == "https://anything.example"


def test_cors_handles_a_missing_config():
    assert server.cors_allowed_origin(None, "https://x.example") == ""
    assert server.cors_allowed_origin(None, "") == ""


# ------------------------------------------ #204 datasets_dir / training_dir --

def test_the_training_engine_reads_the_configured_directories(tmp_path):
    import ainode.training.engine as te

    try:
        te.configure_dirs(NodeConfig(datasets_dir=str(tmp_path / "ds"),
                                     training_dir=str(tmp_path / "runs")))
        assert te.datasets_dir() == tmp_path / "ds"
        assert te.training_dir() == tmp_path / "runs"
        assert te.jobs_dir() == tmp_path / "runs" / "jobs"
        # The dataset-path validator has to quote the configured root, or an
        # absolute path under it is refused as "not under" the default one.
        errors = te.TrainingConfig(base_model="m",
                                   dataset_path=str(tmp_path / "ds" / "a.jsonl")).validate()
        assert not [e for e in errors if "dataset_path" in e]

        # Unset falls back to the AINODE_HOME subpaths.
        te.configure_dirs(NodeConfig())
        assert te.datasets_dir() == te.AINODE_HOME / "datasets"
        assert te.jobs_dir() == te.JOBS_DIR
    finally:
        te.configure_dirs(NodeConfig())


def test_the_app_hands_the_training_manager_its_directories(tmp_path):
    import ainode.training.engine as te

    _real_app(tmp_path, datasets_dir=str(tmp_path / "ds"),
              training_dir=str(tmp_path / "runs"))
    assert te.datasets_dir() == tmp_path / "ds"
    assert te.jobs_dir() == tmp_path / "runs" / "jobs"


# ------------------------------------------------- app.js under a stub DOM --
#
# The two behaviours that only exist in the browser. app.js is plain ES5-era
# script with one global (`AINode`), so node can run it against a hand-built DOM
# stub: no bundler, no jsdom, no npm install. Skipped where node is absent
# rather than asserted on the file's text, because what matters is what the
# code DOES with a fetch and with a throwing localStorage.
#
# auth.js is loaded first, exactly as the page loads it, and the REAL
# AINodeAuth.fetch is what the chat request goes through: the pin headers have to
# survive the wrapper that attaches the API key, or a keyed node loses the
# picker (#218 landed that wrapper between this branch and main).

APP_JS = get_static_path() / "js" / "app.js"
AUTH_JS = get_static_path() / "js" / "auth.js"

_STUB = r"""
const fs = require('fs');
const vm = require('vm');

// --- a DOM that answers every query with a generic element -----------------
function el(tag) {
  const e = {
    tagName: tag || 'div', style: {}, dataset: {}, classList: {
      add() {}, remove() {}, toggle() {}, contains() { return false; },
    },
    children: [], value: '', textContent: '', innerHTML: '', disabled: false,
    options: [], selectedIndex: 0, checked: false, scrollHeight: 0, scrollTop: 0,
    addEventListener() {}, removeEventListener() {}, appendChild(c) { this.children.push(c); },
    remove() {}, closest() { return el(); }, focus() {}, click() {},
    querySelector() { return el(); }, querySelectorAll() { return []; },
    getAttribute() { return null; }, setAttribute() {}, scrollIntoView() {},
  };
  return e;
}
global.document = {
  body: el('body'),
  createElement: el,
  getElementById() { return el(); },
  querySelector() { return el(); },
  querySelectorAll() { return []; },
  addEventListener() {},
};
global.window = { location: { href: 'http://localhost:3000/' }, addEventListener() {} };
global.requestAnimationFrame = (fn) => fn();
global.navigator = { clipboard: { writeText: async () => {} } };
global.performance = { now: () => Date.now() };

// --- the seams under test --------------------------------------------------
const calls = [];
global.fetch = async (url, init) => {
  calls.push({ url: url, init: init || {} });
  return {
    ok: true,
    status: 200,
    headers: { get: (k) => (k.toLowerCase() === 'x-ainode-served-by' ? '10.100.0.15:8000' : null) },
    body: { getReader: () => ({ read: async () => ({ done: true }) }) },
    json: async () => ({}),
    text: async () => '',
  };
};

const store = {};
let quota = Infinity;
global.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => {
    if (String(v).length > quota) {
      const e = new Error('QuotaExceededError');
      e.name = 'QuotaExceededError';
      throw e;
    }
    store[k] = String(v);
  },
  removeItem: (k) => { delete store[k]; },
};

// auth.js first, the way index.html loads it. It attaches itself to `window`
// when there is one, so hoist it to the global app.js reads, and inject the two
// seams it documents as injectable rather than reaching around them.
vm.runInThisContext(fs.readFileSync(process.argv[4], 'utf8'), { filename: 'auth.js' });
globalThis.AINodeAuth = global.window.AINodeAuth || globalThis.AINodeAuth;
AINodeAuth.fetchImpl = global.fetch;
AINodeAuth.storage = global.localStorage;

const src = fs.readFileSync(process.argv[2], 'utf8') + '\nglobalThis.AINode = AINode;\n';
vm.runInThisContext(src, { filename: 'app.js' });

const toasts = [];
AINode.toast = (msg, kind) => { toasts.push({ msg: msg, kind: kind }); };
AINode.refresh = async () => {};
AINode.navigate = () => {};
AINode.renderChatMessages = () => {};
AINode.updateStreamingMessage = () => {};
AINode.renderConversationList = () => {};
AINode.renderAttachmentPreview = () => {};
AINode.renderModelCard = () => {};
AINode.recordChatTurn = () => {};
AINode.showChatOverlay = () => {};
AINode.setChatBusy = () => {};
AINode.updateStreamMetrics = () => {};
AINode.loadTimeLine = () => '';

module.exports = { AINode, AINodeAuth, calls, toasts, store,
                   setQuota: (n) => { quota = n; },
                   el: el };
"""

_CHAT_CASE = r"""
const h = require(process.argv[3]);
const A = h.AINode;

A.state.chatInstance = { model: 'MODEL', node_id: 'spark3', node_name: 'Spark-3-DGX', port: 8001 };
A.state.chatSettings = { system: '', temperature: 0.7, maxTokens: 64, thinking: false };
A.state.messages = [];
A.state.conversations = [];
A.state.currentConversation = null;
A.state.pendingAttachments = [{ name: 'a.png', type: 'image/png', size: 4,
                                dataUrl: 'data:image/png;base64,AAAA' }];
// A keyed node: the wrapper has to add Authorization without dropping the pin.
h.AINodeAuth.setKey('k-abc123');
// The input the send path reads.
const input = h.el('textarea');
input.value = 'hello there';
document.getElementById = (id) => (id === 'chat-input' ? input : h.el());

A.sendMessage().then(() => {
  const chat = h.calls.filter((c) => c.url === '/v1/chat/completions');
  const body = JSON.parse(chat[0].init.body);
  const headers = chat[0].init.headers;
  const assistant = A.state.messages[A.state.messages.length - 1];
  console.log(JSON.stringify({
    calls: chat.length,
    headers: headers,
    wrapped: typeof A.sendMessage === 'function' && /AINodeAuth\.fetch/.test(String(A.sendMessage)),
    model: body.model,
    served_by: assistant.stats && assistant.stats.served_by,
    images_in_body: JSON.stringify(body.messages).indexOf('image_url') !== -1,
  }));
});
"""

_QUOTA_CASE = r"""
const h = require(process.argv[3]);
const A = h.AINode;

A.state.conversations = [
  { id: 'c1', title: 'newest', created_at: 2, messages: [
      { role: 'user', content: 'look', images: ['data:image/png;base64,' + 'A'.repeat(200)] }] },
  { id: 'c2', title: 'oldest', created_at: 1, messages: [{ role: 'user', content: 'hi' }] },
];
A.state.currentConversation = 'c1';

// 1. A normal save must not carry the image bytes into storage.
const okSave = A.saveConversations();
const stored = JSON.parse(h.store['ainode_conversations']);

// 2. Now make every write throw, the way a full origin quota does.
h.setQuota(0);
const fullSave = A.saveConversations();

console.log(JSON.stringify({
  ok_save: okSave,
  stored_has_image_bytes: h.store['ainode_conversations'].indexOf('base64') !== -1,
  stored_image_count: stored[0].messages[0].image_count,
  live_images_kept: A.state.conversations[0].messages[0].images.length,
  full_save: fullSave,
  warned: h.toasts.map((t) => t.msg).join(' | '),
  remaining: A.state.conversations.length,
}));
"""


def _run_node(case: str, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH; the app.js behaviour tests need it")
    harness = tmp_path / "stub_dom.js"
    harness.write_text(textwrap.dedent(_STUB))
    script = tmp_path / "case.js"
    script.write_text(textwrap.dedent(case))
    proc = subprocess.run(
        [node, str(script), str(APP_JS), str(harness), str(AUTH_JS)],
        capture_output=True, text=True, timeout=60,
        cwd=str(tmp_path), env={**os.environ, "NODE_OPTIONS": ""})
    assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_chat_request_carries_the_picked_instance(tmp_path):
    """#197 in the browser: one POST, pinned to the instance in the picker, and
    the instance that answered is recorded from the proxy's own header."""
    out = _run_node(_CHAT_CASE, tmp_path)
    assert out["calls"] == 1
    assert out["wrapped"] is True          # through AINodeAuth, not a bare fetch
    assert out["headers"]["X-AINode-Node"] == "spark3"
    assert out["headers"]["X-AINode-Port"] == "8001"
    assert out["headers"]["Content-Type"] == "application/json"
    # The key the wrapper attaches rides alongside the pin, not instead of it.
    assert out["headers"]["Authorization"] == "Bearer k-abc123"
    assert out["model"] == "MODEL"
    assert out["images_in_body"] is True         # the image still goes to the engine
    assert out["served_by"] == "10.100.0.15:8000"


def test_a_full_localstorage_is_caught_and_reported(tmp_path):
    """#202: the image bytes never reach localStorage, and a quota error is a
    warning plus an eviction rather than an exception that kills persistence for
    the life of the browser profile."""
    out = _run_node(_QUOTA_CASE, tmp_path)
    assert out["ok_save"] is True
    assert out["stored_has_image_bytes"] is False
    assert out["stored_image_count"] == 1
    assert out["live_images_kept"] == 1          # still re-sendable this session
    assert out["full_save"] is False             # it failed, and said so
    assert "full" in out["warned"]
    assert out["remaining"] == 1                 # the oldest was evicted, not the current
