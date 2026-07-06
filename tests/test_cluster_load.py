"""F2: cluster load/unload forwarder + member→solo reset on solo load."""

from __future__ import annotations

import asyncio
import json

import ainode.api.server as server
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState


def _node(nid, fabric="", web_port=3000):
    return ClusterNode(node_id=nid, node_name=nid, gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
                       unified_memory=True, model="", status=NodeStatus.ONLINE, api_port=8000,
                       web_port=web_port, last_seen=0.0, fabric_ip=fabric)


class _Req:
    def __init__(self, app, body):
        self.app = app
        self._b = body

    async def json(self):
        return self._b


def _app(node_id, nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return {"config": NodeConfig(node_id=node_id, api_port=8000), "cluster_state": c}


def test_cluster_load_local_calls_local_handler(monkeypatch):
    called = {}

    async def fake_local(req):
        called["body"] = await req.json()
        from aiohttp import web
        return web.json_response({"status": "launching"})

    monkeypatch.setattr("ainode.models.api_routes.handle_model_load", fake_local)
    app = _app("spark1", [_node("spark1", "10.100.0.11")])
    # node_id == local → local handler, node_id stripped-through is fine
    resp = asyncio.run(server.handle_cluster_load(_Req(app, {"node_id": "spark1", "model": "A"})))
    assert json.loads(resp.body)["status"] == "launching"
    assert called["body"]["model"] == "A"


def test_cluster_load_remote_forwards_over_fabric(monkeypatch):
    posted = {}

    class _Up:
        status = 200
        headers = {"Content-Type": "application/json; charset=utf-8"}  # charset must be stripped
        async def read(self):
            return b'{"status":"launching"}'
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class _Sess:
        def post(self, url, json=None, timeout=None):
            posted["url"] = url
            posted["json"] = json
            return _Up()

    app = _app("spark1", [_node("spark1", "10.100.0.11"), _node("spark3", "10.100.0.15")])
    app["client_session"] = _Sess()
    resp = asyncio.run(server.handle_cluster_load(_Req(app, {"node_id": "spark3", "model": "B"})))
    assert resp.status == 200
    assert posted["url"] == "http://10.100.0.15:3000/api/models/load"
    assert posted["json"] == {"model": "B"}            # node_id stripped before forwarding


def test_cluster_load_unknown_node_404():
    app = _app("spark1", [_node("spark1", "10.100.0.11")])
    resp = asyncio.run(server.handle_cluster_load(_Req(app, {"node_id": "ghost", "model": "B"})))
    assert resp.status == 404


def test_cluster_unload_remote_targets_unload_path(monkeypatch):
    posted = {}

    class _Up:
        status = 200
        headers = {"Content-Type": "application/json; charset=utf-8"}  # charset must be stripped
        async def read(self):
            return b"{}"
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class _Sess:
        def post(self, url, json=None, timeout=None):
            posted["url"] = url
            return _Up()

    app = _app("spark1", [_node("spark1", "10.100.0.11"), _node("spark3", "10.100.0.15")])
    app["client_session"] = _Sess()
    asyncio.run(server.handle_cluster_unload(_Req(app, {"node_id": "spark3", "model": "B"})))
    assert posted["url"] == "http://10.100.0.15:3000/api/models/unload"


class _FakeBackend:
    """A backend stand-in: records its config + start, no container."""
    def __init__(self, cfg, instance_id=""):
        self.config = cfg
        self.instance_id = instance_id
        self.started = False
        self.stopped = False
    def is_running(self):
        return self.started and not self.stopped
    def start(self):
        self.started = True
        return True
    def stop(self):
        self.stopped = True


def _patch_backend(monkeypatch, sink=None):
    """Patch get_backend everywhere handle_model_load imports it; capture builds."""
    import ainode.engine.backends as backends_mod
    made = sink if sink is not None else {}
    made.setdefault("backends", [])
    def fake_get_backend(cfg, instance_id="", on_ready=None):
        b = _FakeBackend(cfg, instance_id)
        made["backends"].append(b)
        return b
    monkeypatch.setattr(backends_mod, "get_backend", fake_get_backend)
    return made


def test_lazy_engine_uses_get_backend(monkeypatch):
    """A solo load builds the serving backend via get_backend (not the legacy
    host-venv VLLMEngine, which would never launch a container) and starts it."""
    import ainode.models.api_routes as mr

    made = _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark3", api_port=8000)
    cfg.save = lambda: None
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "Qwen/Q"})))
    assert made["backends"] and made["backends"][0].started
    # The primary is wired into app["engine"] for the back-compat proxy/status path.
    assert app["engine"] is made["backends"][0]


def test_solo_load_resets_member_mode(monkeypatch):
    """A solo load on a node stuck in 'member' mode resets it to solo so it serves."""
    import ainode.models.api_routes as mr

    _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark3", api_port=8000)
    cfg.distributed_mode = "member"
    cfg.peer_ips = ["10.100.0.11"]
    cfg.save = lambda: None
    # no cluster workers → sharding_config stays None → solo path
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "Qwen/Q"})))
    assert cfg.distributed_mode == "solo"
    assert cfg.peer_ips == []


def test_solo_load_appends_not_replaces(monkeypatch):
    """A 2nd solo load on a busy node ADDS an instance (own port + container token),
    leaving the first serving. Reloading the SAME model replaces only that one."""
    import ainode.models.api_routes as mr
    from ainode.discovery.instance import InstanceRecord  # noqa: F401 (sanity import)

    made = _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark1", api_port=8000)
    cfg.save = lambda: None
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}

    asyncio.run(mr.handle_model_load(_Req(app, {"model": "model-A"})))
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "model-B"})))

    mgr = app["instances"]
    recs = {r.model: r for r in mgr.records()}
    assert set(recs) == {"model-A", "model-B"}
    # distinct ports: primary keeps 8000, the stacked one gets 8001
    assert recs["model-A"].api_port == 8000
    assert recs["model-B"].api_port == 8001
    # both backends are alive (append did NOT stop the first)
    assert all(b.started and not b.stopped for b in made["backends"])
    # primary stays wired to app["engine"]; the 2nd is manager-only
    assert app["engine"].config.model == "model-A"

    # Reloading model-A replaces ONLY that instance (stops the old A backend).
    n_before = len(made["backends"])
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "model-A"})))
    assert len(made["backends"]) == n_before + 1  # a fresh A backend
    assert {r.model for r in mgr.records()} == {"model-A", "model-B"}
    # B's port is now free's lowest → A reclaims 8000? No: B holds 8001, A re-gets 8000.
    recs = {r.model: r for r in mgr.records()}
    assert recs["model-A"].api_port == 8000
    assert recs["model-B"].api_port == 8001


def test_load_appends_when_boot_engine_already_seeded(monkeypatch):
    """When the manager is pre-seeded with the boot engine (as run_server does),
    a NEW-model load stacks on :8001 and never touches the boot's :8000 — the
    boot stays primary (app["engine"]) and on its own port/container."""
    import ainode.models.api_routes as mr
    from ainode.discovery.instance import InstanceRecord
    from ainode.engine.instance_manager import InstanceManager

    _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark1", api_port=8000)
    cfg.model = "boot-model"
    cfg.save = lambda: None
    boot = _FakeBackend(cfg, "")
    boot.started = True  # boot engine already serving on :8000
    mgr = InstanceManager(base_port=8000)
    mgr.add(InstanceRecord(instance_id="spark1:boot-model", model="boot-model",
                           head_node_id="spark1", peer_ips=[], api_port=8000,
                           tensor_parallel_size=1, status="serving"), boot)
    app = {"engine": boot, "instances": mgr, "config": cfg,
           "cluster_state": ClusterState(), "ray_autostart_state": None}

    asyncio.run(mr.handle_model_load(_Req(app, {"model": "new-model"})))

    recs = {r.model: r.api_port for r in mgr.records()}
    assert recs == {"boot-model": 8000, "new-model": 8001}  # appended on 8001
    assert app["engine"] is boot          # boot stays primary, untouched
    assert boot.stopped is False          # boot container never killed
    assert cfg.model == "boot-model"      # shared config still the primary's


def test_instance_manifest_persist_and_replay(monkeypatch, tmp_path):
    """Always-on: the loaded solo set is persisted, and on a simulated restart the
    replay re-loads the STACKED extras (skipping the boot primary) — no duplicates."""
    import ainode.models.api_routes as mr

    _patch_backend(monkeypatch)
    monkeypatch.setattr(mr, "_manifest_path", lambda: tmp_path / "instances.json")

    cfg = NodeConfig(node_id="spark1", api_port=8000)
    cfg.save = lambda: None
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}

    # Load two solo models — both captured in the manifest with their gmu.
    mr.append_solo_instance(app, "model-A", 0.3)
    mr.append_solo_instance(app, "model-B", 0.25)
    saved = mr.load_instance_manifest()
    assert {e["model"] for e in saved} == {"model-A", "model-B"}
    assert any(e["gpu_memory_utilization"] == 0.25 for e in saved)

    # Simulate a restart: fresh app where the boot engine has claimed model-A as
    # primary. Replay must bring back model-B only (A already loaded), no dupes.
    cfg2 = NodeConfig(node_id="spark1", api_port=8000)
    cfg2.save = lambda: None
    app2 = {"engine": None, "config": cfg2, "cluster_state": ClusterState(),
            "ray_autostart_state": None}
    # Boot seeding loads the primary WITHOUT rewriting the manifest (persist=False),
    # mirroring server startup — so the saved [A,B] set survives for replay.
    mr.append_solo_instance(app2, "model-A", 0.3, persist=False)  # boot primary

    async def _noop_sleep(*a, **k):
        return None
    monkeypatch.setattr(mr.asyncio, "sleep", _noop_sleep)
    asyncio.run(mr.replay_instances_on_startup(app2))

    models = {i.record.model for i in app2["instances"].instances()}
    assert models == {"model-A", "model-B"}


def test_unload_one_stacked_instance_leaves_the_other(monkeypatch):
    """Unloading one stacked model stops ONLY that instance; the co-resident one
    keeps serving and stays in the manager."""
    import ainode.models.api_routes as mr

    _patch_backend(monkeypatch)
    cfg = NodeConfig(node_id="spark1", api_port=8000)
    cfg.save = lambda: None
    app = {"engine": None, "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "model-A"})))
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "model-B"})))
    mgr = app["instances"]
    b_backend = mgr.by_model("model-B").backend

    resp = asyncio.run(mr.handle_model_unload(_Req(app, {"model": "model-B"})))
    body = json.loads(resp.body)
    assert body["stopped"] is True
    assert body["remaining"] == 1
    assert b_backend.stopped is True
    # A survives, untouched
    a = mgr.by_model("model-A")
    assert a is not None and a.backend.stopped is False
    assert mgr.by_model("model-B") is None
