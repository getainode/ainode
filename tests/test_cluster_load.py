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


def test_solo_load_resets_member_mode(monkeypatch):
    """A solo load on a node stuck in 'member' mode resets it to solo so it serves."""
    import ainode.models.api_routes as mr

    class _Eng:
        def is_running(self):
            return False
        def stop(self):
            pass
        def start(self):
            return True

    cfg = NodeConfig(node_id="spark3", api_port=8000)
    cfg.distributed_mode = "member"
    cfg.peer_ips = ["10.100.0.11"]
    cfg.save = lambda: None
    # no cluster workers → sharding_config stays None → solo path
    app = {"engine": _Eng(), "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "Qwen/Q"})))
    assert cfg.distributed_mode == "solo"
    assert cfg.peer_ips == []
