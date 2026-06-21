"""'Federation usable from browser': mem-util knob, routing failover, clear-on-fail."""

from __future__ import annotations

import asyncio
import json

import ainode.api.server as server
import ainode.engine.backends as backends_mod
import ainode.models.api_routes as mr
from ainode.api.server import _routing_candidates
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.engine.backends.nvidia import NvidiaBackend


def _node(nid, model="", fabric="", status=NodeStatus.ONLINE, api_port=8000):
    return ClusterNode(node_id=nid, node_name=nid, gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
                       unified_memory=True, model=model, status=status, api_port=api_port,
                       web_port=3000, last_seen=0.0, fabric_ip=fabric)


def _cluster(nodes):
    c = ClusterState()
    for n in nodes:
        c.add_node(n)
    return c


class _Req:
    def __init__(self, app, body=None):
        self.app = app
        self._b = body or {}
    async def json(self):
        return self._b


# --- 1. mem-util knob -------------------------------------------------------

def test_default_gpu_mem_util_lowered():
    assert NodeConfig(node_id="n").gpu_memory_utilization == 0.5  # was 0.9


def test_gpu_mem_util_reaches_serve_args():
    b = NvidiaBackend(__import__("dataclasses").replace(
        NodeConfig(node_id="n"), gpu_memory_utilization=0.3))
    args = b._build_vllm_serve_args(tp_size=1)
    i = args.index("--gpu-memory-utilization")
    assert args[i + 1] == "0.3"


def test_load_body_sets_gpu_mem_util(monkeypatch):
    class _Eng:
        def is_running(self): return False
        def stop(self): pass
        def start(self): return True
    cfg = NodeConfig(node_id="n"); cfg.save = lambda: None
    app = {"engine": _Eng(), "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    asyncio.run(mr.handle_model_load(_Req(app, {"model": "m/x", "gpu_memory_utilization": 0.25})))
    assert cfg.gpu_memory_utilization == 0.25


# --- 3. routing-truth: candidates + failover + clear-on-fail ----------------

def test_candidates_local_first_then_all_serving():
    c = _cluster([
        _node("head", model="M", fabric="10.0.0.9"),   # local node also serves M
        _node("b", model="M", fabric="10.0.0.2"),      # remote also serves M
    ])
    cands = _routing_candidates(c, "M", local_node_id="head", local_port=8000)
    assert cands[0] == ("localhost", 8000)             # local hop first
    assert ("10.0.0.2", 8000) in cands                 # remote is a failover target


def test_candidates_excludes_non_serving_model():
    c = _cluster([_node("a", model="OTHER", fabric="10.0.0.1")])
    assert _routing_candidates(c, "M", "head", 8000) == []


def test_proxy_fails_over_past_ghost(monkeypatch):
    # ghost (10.0.0.1) listed first but unreachable; live (10.0.0.2) second.
    c = _cluster([
        _node("ghost", model="M", fabric="10.0.0.1"),
        _node("live", model="M", fabric="10.0.0.2"),
    ])
    tried = []

    class _Up:
        status = 200
        headers = {"Content-Type": "application/json"}
        async def read(self): return b'{"ok":1}'
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Sess:
        def request(self, method, url, **kw):
            tried.append(url)
            if "10.0.0.1" in url:
                raise server.aiohttp.ClientError("ghost down")
            return _Up()

    class _Collector:
        def record_request(self, *a, **k): pass

    app = {"config": NodeConfig(node_id="head", api_port=8000), "cluster_state": c,
           "client_session": _Sess(), "metrics_collector": _Collector()}

    class _R:
        method = "POST"; path = "/v1/completions"; headers = {}
        def __init__(self): self.app = app
        async def read(self): return b'{"model":"M","prompt":"hi"}'
    resp = asyncio.run(server.proxy_to_vllm(_R()))
    assert resp.status == 200                       # failed over, not 502
    assert any("10.0.0.1" in u for u in tried) and any("10.0.0.2" in u for u in tried)


def test_failed_load_clears_model_claim(monkeypatch):
    class _Eng:
        def is_running(self): return False
        def stop(self): pass
        def start(self): return False          # launch fails
    cfg = NodeConfig(node_id="n", model="m/x"); cfg.save = lambda: None
    app = {"engine": _Eng(), "config": cfg, "cluster_state": ClusterState(),
           "ray_autostart_state": None}
    resp = asyncio.run(mr.handle_model_load(_Req(app, {"model": "m/x"})))
    assert resp.status == 500
    assert cfg.model is None                    # ghost claim cleared
