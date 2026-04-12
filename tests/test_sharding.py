"""Tests for ainode.engine.sharding — ShardingPlanner, memory estimation, shard assignment."""

import pytest

from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterState, ClusterNode
from ainode.engine.sharding import (
    ShardingPlanner,
    ShardingStrategy,
    estimate_model_size,
    MEMORY_OVERHEAD_FACTOR,
)


def _make_node(node_id, gpu_memory_gb, model=""):
    """Helper to create a ClusterNode for testing."""
    return ClusterNode(
        node_id=node_id,
        node_name="node-" + node_id,
        gpu_name="NVIDIA GB10",
        gpu_memory_gb=gpu_memory_gb,
        unified_memory=True,
        model=model,
        status=NodeStatus.ONLINE,
        api_port=8000,
        web_port=3000,
        last_seen=0.0,
    )


def _make_cluster(*nodes):
    """Helper to create a ClusterState from a list of nodes."""
    cluster = ClusterState()
    for node in nodes:
        cluster.add_node(node)
    return cluster


class TestEstimateModelSize:
    def test_known_model(self):
        assert estimate_model_size("meta-llama/Llama-3.1-70B-Instruct") == 140.0

    def test_known_model_awq(self):
        assert estimate_model_size("meta-llama/Llama-3.1-70B-Instruct-AWQ") == 35.0

    def test_unknown_model_with_param_count(self):
        size = estimate_model_size("some-org/CustomModel-13B-Chat")
        assert size == 26.0

    def test_unknown_model_with_decimal_param_count(self):
        size = estimate_model_size("org/Model-1.5B")
        assert size == 3.0

    def test_completely_unknown_model(self):
        size = estimate_model_size("org/mystery-model")
        assert size == 14.0


class TestShardingPlannerMemoryEstimation:
    def setup_method(self):
        self.planner = ShardingPlanner()

    def test_single_node_memory(self):
        mem = self.planner.estimate_memory_per_node(100.0, 1, ShardingStrategy.PIPELINE_PARALLEL)
        assert mem == 100.0 * MEMORY_OVERHEAD_FACTOR

    def test_two_node_pipeline(self):
        mem = self.planner.estimate_memory_per_node(100.0, 2, ShardingStrategy.PIPELINE_PARALLEL)
        assert mem == 50.0 * MEMORY_OVERHEAD_FACTOR

    def test_four_node_pipeline(self):
        mem = self.planner.estimate_memory_per_node(200.0, 4, ShardingStrategy.PIPELINE_PARALLEL)
        assert mem == 50.0 * MEMORY_OVERHEAD_FACTOR

    def test_tensor_parallel_has_higher_overhead(self):
        mem_tp = self.planner.estimate_memory_per_node(100.0, 2, ShardingStrategy.TENSOR_PARALLEL)
        mem_pp = self.planner.estimate_memory_per_node(100.0, 2, ShardingStrategy.PIPELINE_PARALLEL)
        assert mem_tp > mem_pp

    def test_zero_nodes_returns_full_overhead(self):
        mem = self.planner.estimate_memory_per_node(100.0, 0, ShardingStrategy.AUTO)
        assert mem == 100.0 * MEMORY_OVERHEAD_FACTOR


class TestCanFitModel:
    def setup_method(self):
        self.planner = ShardingPlanner()

    def test_single_large_node_fits(self):
        cluster = _make_cluster(_make_node("a", 256.0))
        assert self.planner.can_fit_model("meta-llama/Llama-3.1-70B-Instruct", cluster) is True

    def test_single_small_node_does_not_fit(self):
        cluster = _make_cluster(_make_node("a", 24.0))
        assert self.planner.can_fit_model("meta-llama/Llama-3.1-70B-Instruct", cluster) is False

    def test_two_nodes_combined_fit(self):
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        assert self.planner.can_fit_model("meta-llama/Llama-3.1-70B-Instruct", cluster) is True

    def test_empty_cluster(self):
        cluster = ClusterState()
        assert self.planner.can_fit_model("meta-llama/Llama-3.2-3B-Instruct", cluster) is False

    def test_small_model_fits_small_node(self):
        cluster = _make_cluster(_make_node("a", 16.0))
        assert self.planner.can_fit_model("meta-llama/Llama-3.2-3B-Instruct", cluster) is True


class TestPlanSharding:
    def setup_method(self):
        self.planner = ShardingPlanner()

    def test_single_node_no_sharding(self):
        cluster = _make_cluster(_make_node("a", 256.0))
        config = self.planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)

        assert config.is_distributed is False
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert len(config.shard_map) == 1
        assert "a" in config.shard_map
        assert config.shard_map["a"].role == "head"

    def test_two_node_sharding(self):
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        config = self.planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)

        assert config.is_distributed is True
        assert config.pipeline_parallel_size == 2
        assert config.tensor_parallel_size == 1
        assert config.strategy == ShardingStrategy.PIPELINE_PARALLEL
        assert len(config.shard_map) == 2
        roles = set()
        for s in config.shard_map.values():
            roles.add(s.role)
        assert roles == {"head", "worker"}

    def test_four_node_large_model(self):
        """405B model needs multiple large nodes (e.g. 4x DGX Spark with 300GB each)."""
        cluster = _make_cluster(
            _make_node("a", 300.0),
            _make_node("b", 300.0),
            _make_node("c", 300.0),
            _make_node("d", 300.0),
        )
        config = self.planner.plan_sharding("meta-llama/Llama-3.1-405B-Instruct", cluster)

        assert config.is_distributed is True
        assert config.pipeline_parallel_size >= 2
        assert len(config.shard_map) >= 2

    def test_auto_strategy_selects_pipeline(self):
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        config = self.planner.plan_sharding(
            "meta-llama/Llama-3.1-70B-Instruct", cluster, ShardingStrategy.AUTO
        )
        assert config.strategy == ShardingStrategy.PIPELINE_PARALLEL

    def test_explicit_tensor_parallel(self):
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        config = self.planner.plan_sharding(
            "meta-llama/Llama-3.1-70B-Instruct", cluster, ShardingStrategy.TENSOR_PARALLEL
        )
        assert config.strategy == ShardingStrategy.TENSOR_PARALLEL
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 1

    def test_insufficient_memory_raises(self):
        cluster = _make_cluster(_make_node("a", 32.0))
        with pytest.raises(ValueError, match="lacks memory"):
            self.planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)

    def test_no_nodes_raises(self):
        cluster = ClusterState()
        with pytest.raises(ValueError, match="No online nodes"):
            self.planner.plan_sharding("meta-llama/Llama-3.2-3B-Instruct", cluster)

    def test_selects_fewest_nodes_needed(self):
        cluster = _make_cluster(
            _make_node("big", 256.0),
            _make_node("medium", 128.0),
            _make_node("small", 64.0),
        )
        config = self.planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)
        assert len(config.shard_map) == 1
        assert "big" in config.shard_map


class TestShardAssignment:
    def test_shard_layers_are_assigned(self):
        planner = ShardingPlanner()
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        config = planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)
        assignment = planner.get_shard_assignment(config)

        for node_id, shard in assignment.items():
            assert shard.layers is not None
            assert shard.estimated_memory_gb > 0


class TestShardingConfigSerialization:
    def test_to_dict(self):
        planner = ShardingPlanner()
        cluster = _make_cluster(_make_node("a", 128.0), _make_node("b", 128.0))
        config = planner.plan_sharding("meta-llama/Llama-3.1-70B-Instruct", cluster)
        d = config.to_dict()

        assert d["model"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert d["strategy"] in ("tensor_parallel", "pipeline_parallel")
        assert d["is_distributed"] is True
        assert d["world_size"] >= 2
        assert isinstance(d["shard_map"], dict)
        for node_id, shard in d["shard_map"].items():
            assert "role" in shard
            assert "layers" in shard
            assert "estimated_memory_gb" in shard
