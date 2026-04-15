"""Phase 1 distributed inference tests — Ray autostart + VRAM aggregation + sharded launch.

Ray is mocked so these tests never touch a real Ray runtime.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.engine.ray_autostart import (
    RayAutostartState,
    autostart_loop,
    join_worker_if_possible,
    start_head_if_needed,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ann(node_id: str, role: str = "auto") -> NodeAnnouncement:
    return NodeAnnouncement(
        node_id=node_id,
        node_name=f"host-{node_id}",
        gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0,
        unified_memory=True,
        model="",
        status="starting",
        api_port=8000,
        web_port=3000,
        cluster_id="default",
        role=role,
        is_master=False,
    )


def _node(node_id: str, mem: float = 128.0, role: str = "auto") -> ClusterNode:
    return ClusterNode(
        node_id=node_id,
        node_name=f"host-{node_id}",
        gpu_name="NVIDIA GB10",
        gpu_memory_gb=mem,
        unified_memory=True,
        model="",
        status=NodeStatus.ONLINE,
        api_port=8000,
        web_port=3000,
        last_seen=0.0,
        cluster_id="default",
        role=role,
    )


# ---------------------------------------------------------------------------
# Ray autostart primitives
# ---------------------------------------------------------------------------

class TestRayAutostartPrimitives:
    def test_start_head_no_ray_installed(self):
        state = RayAutostartState()
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=False):
            out = start_head_if_needed(state)
        assert out.is_head is False
        assert out.enabled is False
        assert out.error == "ray not installed"

    def test_start_head_success(self):
        state = RayAutostartState()
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=True), \
             patch("ainode.engine.ray_autostart.start_ray_head", return_value="10.0.0.2:6379") as m, \
             patch("ainode.engine.ray_autostart._gpu_count", return_value=1):
            out = start_head_if_needed(state)
        assert out.is_head is True
        assert out.head_address == "10.0.0.2:6379"
        m.assert_called_once()

    def test_start_head_idempotent(self):
        state = RayAutostartState(is_head=True, head_address="x:6379")
        with patch("ainode.engine.ray_autostart.start_ray_head") as m:
            start_head_if_needed(state)
        m.assert_not_called()

    def test_join_worker_success(self):
        state = RayAutostartState()
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=True), \
             patch("ainode.engine.ray_autostart.join_ray_cluster", return_value=True) as m, \
             patch("ainode.engine.ray_autostart._gpu_count", return_value=1):
            out = join_worker_if_possible(state, "10.0.0.1:6379")
        assert out.joined_as_worker is True
        assert out.head_address == "10.0.0.1:6379"
        m.assert_called_once()

    def test_join_worker_no_ray(self):
        state = RayAutostartState()
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=False):
            out = join_worker_if_possible(state, "10.0.0.1:6379")
        assert out.joined_as_worker is False
        assert out.error == "ray not installed"


# ---------------------------------------------------------------------------
# Autostart loop — master vs worker behaviour
# ---------------------------------------------------------------------------

class TestAutostartLoop:
    def _run_one_iteration(self, cluster, get_master_addr):
        """Run the autostart loop briefly and cancel."""
        state = RayAutostartState()

        async def runner():
            task = asyncio.create_task(
                autostart_loop(cluster, get_master_addr, state, poll_seconds=0.01)
            )
            # Allow a couple of iterations to fire
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return state

        return asyncio.run(runner())

    def test_master_starts_head(self):
        ann = _ann("aaa", role="master")
        cluster = ClusterState(local_announcement=ann)
        # The local announcement is automatically treated as a member
        assert cluster.is_master_of_cluster() is True

        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=True), \
             patch("ainode.engine.ray_autostart.start_ray_head", return_value="1.2.3.4:6379") as m, \
             patch("ainode.engine.ray_autostart._gpu_count", return_value=1):
            state = self._run_one_iteration(cluster, lambda: None)

        assert state.is_head is True
        assert state.head_address == "1.2.3.4:6379"
        assert m.called

    def test_worker_joins_when_master_discovered(self):
        # Local node is a worker, remote is master
        ann = _ann("zzz", role="worker")
        cluster = ClusterState(local_announcement=ann)
        cluster.add_node(_node("aaa", role="master"))
        assert cluster.is_master_of_cluster() is False

        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=True), \
             patch("ainode.engine.ray_autostart.join_ray_cluster", return_value=True) as m, \
             patch("ainode.engine.ray_autostart._gpu_count", return_value=1):
            state = self._run_one_iteration(cluster, lambda: "1.2.3.4:6379")

        assert state.joined_as_worker is True
        assert m.called

    def test_worker_waits_for_master(self):
        ann = _ann("zzz", role="worker")
        cluster = ClusterState(local_announcement=ann)
        # Only ourselves, no master discovered
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=True), \
             patch("ainode.engine.ray_autostart.join_ray_cluster") as m_join, \
             patch("ainode.engine.ray_autostart.start_ray_head") as m_head:
            state = self._run_one_iteration(cluster, lambda: None)

        assert state.joined_as_worker is False
        assert state.is_head is False
        m_join.assert_not_called()
        m_head.assert_not_called()

    def test_graceful_when_ray_missing(self):
        ann = _ann("aaa", role="master")
        cluster = ClusterState(local_announcement=ann)
        with patch("ainode.engine.ray_autostart.is_ray_available", return_value=False):
            state = self._run_one_iteration(cluster, lambda: None)
        assert state.enabled is False
        assert state.error == "ray not installed"


# ---------------------------------------------------------------------------
# Cluster VRAM aggregation endpoint
# ---------------------------------------------------------------------------

class TestClusterResourcesEndpoint:
    def test_vram_math_single_node(self):
        cluster = ClusterState()
        cluster.add_node(_node("a", mem=128.0))
        ready = cluster.members()
        total_vram = sum(n.gpu_memory_gb for n in ready)
        assert total_vram == 128.0
        assert len(ready) == 1

    def test_vram_math_three_nodes(self):
        cluster = ClusterState()
        cluster.add_node(_node("a", mem=128.0))
        cluster.add_node(_node("b", mem=128.0))
        cluster.add_node(_node("c", mem=128.0))
        ready = cluster.members()
        total_vram = sum(n.gpu_memory_gb for n in ready)
        assert total_vram == 384.0
        assert len(ready) == 3

    def test_cluster_resources_handler_payload(self):
        """Call the handler directly (avoids pytest-aiohttp fixture dep)."""
        from ainode.api.server import handle_cluster_resources

        ann = _ann("aaa", role="master")
        cluster = ClusterState(local_announcement=ann)
        cluster.add_node(_node("bbb", mem=128.0))

        app = {
            "cluster_state": cluster,
            "ray_autostart_state": RayAutostartState(is_head=True, head_address="x:6379"),
        }

        class _Req:
            pass
        request = _Req()
        request.app = app

        resp = asyncio.run(handle_cluster_resources(request))
        import json as _json
        data = _json.loads(resp.body.decode())
        assert data["total_nodes"] == 2
        assert data["total_vram_gb"] == 256.0
        assert data["total_gpus"] == 2
        assert len(data["nodes"]) == 2
        assert data["ray"]["is_head"] is True


# ---------------------------------------------------------------------------
# Sharded launch hand-off
# ---------------------------------------------------------------------------

class TestShardedLaunch:
    def test_launch_uses_sharding_plan_when_workers_present(self):
        """Verify planner is invoked and distributed launch is attempted when >1 node."""
        from ainode.engine.sharding import ShardingPlanner, ShardingStrategy

        cluster = ClusterState()
        cluster.add_node(_node("a", mem=128.0))
        cluster.add_node(_node("b", mem=128.0))

        planner = ShardingPlanner()
        config = planner.plan_sharding(
            "meta-llama/Llama-3.1-70B-Instruct", cluster, ShardingStrategy.AUTO
        )
        assert config.is_distributed is True
        assert config.world_size == 2
        assert config.pipeline_parallel_size == 2

    def test_launch_single_node_path(self):
        """With no workers, sharding plan stays on one node."""
        from ainode.engine.sharding import ShardingPlanner

        cluster = ClusterState()
        cluster.add_node(_node("a", mem=256.0))

        planner = ShardingPlanner()
        config = planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)
        assert config.is_distributed is False
        assert config.tensor_parallel_size == 1

    def test_launch_distributed_sets_ray_env(self):
        """VLLMEngine.launch_distributed sets RAY_ADDRESS when distributed."""
        from ainode.core.config import NodeConfig
        from ainode.engine.sharding import ShardingConfig, ShardingStrategy, ShardInfo
        from ainode.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine(NodeConfig())
        cfg = ShardingConfig(
            model="meta-llama/Llama-3.1-70B-Instruct",
            strategy=ShardingStrategy.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            nodes=["a:8000", "b:8000"],
            shard_map={
                "a": ShardInfo(node_id="a", host="a", api_port=8000, role="head", layers="0-19", estimated_memory_gb=80),
                "b": ShardInfo(node_id="b", host="b", api_port=8000, role="worker", layers="20-39", estimated_memory_gb=80),
            },
            ray_head_address="10.0.0.1:6379",
        )

        popen_mock = MagicMock()
        popen_mock.poll.return_value = None
        popen_mock.stdout = None

        with patch("ainode.engine.vllm_engine.subprocess.Popen", return_value=popen_mock) as m, \
             patch("ainode.engine.vllm_engine.detect_gpu", return_value=None), \
             patch("ainode.engine.vllm_engine.threading.Thread"):
            ok = engine.launch_distributed(cfg)
        assert ok is True
        # Inspect env passed to Popen
        _, kwargs = m.call_args
        assert kwargs["env"]["RAY_ADDRESS"] == "10.0.0.1:6379"
        # And that tensor/pipeline args were appended to the cmd
        cmd = m.call_args[0][0]
        assert "--pipeline-parallel-size" in cmd
        assert "2" in cmd
