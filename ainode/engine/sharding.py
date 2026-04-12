"""Model sharding coordinator — plans and manages distributed inference across cluster nodes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ainode.discovery.cluster import ClusterNode, ClusterState

logger = logging.getLogger(__name__)

# Approximate model sizes in GB (weights only, at fp16).
# This is a rough heuristic — real memory usage includes KV cache, activations, etc.
KNOWN_MODEL_SIZES: Dict[str, float] = {
    "meta-llama/Llama-3.2-3B-Instruct": 6.0,
    "meta-llama/Llama-3.1-8B-Instruct": 16.0,
    "meta-llama/Llama-3.1-70B-Instruct": 140.0,
    "meta-llama/Llama-3.1-70B-Instruct-AWQ": 35.0,
    "meta-llama/Llama-3.3-70B-Instruct": 140.0,
    "meta-llama/Llama-3.1-405B-Instruct": 810.0,
    "Qwen/Qwen2.5-72B-Instruct": 145.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 14.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 14.0,
    "microsoft/Phi-3-mini-4k-instruct": 7.5,
    "meta-llama/CodeLlama-34b-Instruct-hf": 63.0,
}

# Overhead factor: vLLM needs extra memory for KV cache, CUDA context, etc.
MEMORY_OVERHEAD_FACTOR = 1.20


class ShardingStrategy(str, Enum):
    """How to distribute model layers across devices."""
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    AUTO = "auto"


@dataclass
class ShardInfo:
    """Describes what a single node is responsible for in a sharded deployment."""
    node_id: str
    host: str
    api_port: int
    role: str  # "head" or "worker"
    layers: Optional[str] = None  # e.g. "0-39" for pipeline parallel
    estimated_memory_gb: float = 0.0


@dataclass
class ShardingConfig:
    """Complete sharding plan for a model across the cluster."""
    model: str
    strategy: ShardingStrategy
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    nodes: List[str] = field(default_factory=list)  # node endpoint addresses
    shard_map: Dict[str, ShardInfo] = field(default_factory=dict)
    total_memory_required_gb: float = 0.0
    ray_head_address: Optional[str] = None

    @property
    def is_distributed(self) -> bool:
        """True if model spans multiple nodes (requires Ray)."""
        return len(self.nodes) > 1

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "strategy": self.strategy.value,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "nodes": self.nodes,
            "shard_map": {k: _shard_to_dict(v) for k, v in self.shard_map.items()},
            "total_memory_required_gb": round(self.total_memory_required_gb, 1),
            "is_distributed": self.is_distributed,
            "world_size": self.world_size,
            "ray_head_address": self.ray_head_address,
        }


def _shard_to_dict(s: ShardInfo) -> dict:
    return {
        "node_id": s.node_id,
        "host": s.host,
        "api_port": s.api_port,
        "role": s.role,
        "layers": s.layers,
        "estimated_memory_gb": round(s.estimated_memory_gb, 1),
    }


def estimate_model_size(model_id: str) -> float:
    """Return estimated model memory in GB.

    Uses the known sizes table, or falls back to a rough heuristic based on
    common naming patterns (e.g. '70B' in the name).
    """
    if model_id in KNOWN_MODEL_SIZES:
        return KNOWN_MODEL_SIZES[model_id]

    # Heuristic: extract parameter count from model name
    name_lower = model_id.lower()
    for token in name_lower.replace("-", " ").replace("_", " ").split():
        if token.endswith("b") and token[:-1].replace(".", "").isdigit():
            params_b = float(token[:-1])
            # ~2 bytes per param at fp16
            return params_b * 2.0

    # Default fallback: assume 14 GB (7B model at fp16)
    return 14.0


class ShardingPlanner:
    """Plans optimal sharding strategy for a model given the cluster state."""

    def __init__(self, memory_utilization: float = 0.85):
        self.memory_utilization = memory_utilization

    def estimate_memory_per_node(
        self, model_size_gb: float, num_nodes: int, strategy: ShardingStrategy
    ) -> float:
        """Estimate how much memory each node needs for the given sharding plan."""
        if num_nodes <= 0:
            return model_size_gb * MEMORY_OVERHEAD_FACTOR

        if strategy == ShardingStrategy.TENSOR_PARALLEL:
            # Tensor parallel splits weights evenly but each node also stores
            # some shared buffers — overhead is higher.
            per_node = (model_size_gb / num_nodes) * 1.10
        elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
            # Pipeline parallel splits layers across nodes.
            per_node = model_size_gb / num_nodes
        else:
            # Auto: assume pipeline for multi-node (lower overhead)
            per_node = model_size_gb / num_nodes

        return per_node * MEMORY_OVERHEAD_FACTOR

    def can_fit_model(self, model_id: str, cluster_state: ClusterState) -> bool:
        """Check if the cluster has enough total memory to run the model."""
        model_size = estimate_model_size(model_id)
        required = model_size * MEMORY_OVERHEAD_FACTOR
        nodes = cluster_state.get_nodes(include_offline=False)
        if not nodes:
            return False
        total_available = sum(
            n.gpu_memory_gb * self.memory_utilization for n in nodes
        )
        return total_available >= required

    def get_shard_assignment(self, config: ShardingConfig) -> Dict[str, ShardInfo]:
        """Return the shard_map — already populated during plan_sharding."""
        return dict(config.shard_map)

    def plan_sharding(
        self, model_id: str, cluster_state: ClusterState, strategy: ShardingStrategy = ShardingStrategy.AUTO
    ) -> ShardingConfig:
        """Given a model and available cluster nodes, determine optimal sharding.

        Strategy:
        - If model fits on a single node, don't shard.
        - If model needs multiple nodes, use pipeline parallel (simpler over network).
        - tensor_parallel is used only within a single node with multiple GPUs.
        """
        model_size = estimate_model_size(model_id)
        required = model_size * MEMORY_OVERHEAD_FACTOR
        nodes = cluster_state.get_nodes(include_offline=False)

        if not nodes:
            raise ValueError("No online nodes in cluster")

        # Sort nodes by available memory (largest first)
        nodes_sorted = sorted(nodes, key=lambda n: n.gpu_memory_gb, reverse=True)

        # Check if a single node can handle it
        best_node = nodes_sorted[0]
        best_usable = best_node.gpu_memory_gb * self.memory_utilization

        if best_usable >= required:
            # Single node — no sharding needed
            shard = ShardInfo(
                node_id=best_node.node_id,
                host="localhost",
                api_port=best_node.api_port,
                role="head",
                layers="all",
                estimated_memory_gb=required,
            )
            return ShardingConfig(
                model=model_id,
                strategy=ShardingStrategy.TENSOR_PARALLEL,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                nodes=[f"localhost:{best_node.api_port}"],
                shard_map={best_node.node_id: shard},
                total_memory_required_gb=required,
            )

        # Need multiple nodes — find minimum set that has enough memory
        selected_nodes: List[ClusterNode] = []
        accumulated = 0.0
        for node in nodes_sorted:
            selected_nodes.append(node)
            accumulated += node.gpu_memory_gb * self.memory_utilization
            if accumulated >= required:
                break

        if accumulated < required:
            raise ValueError(
                f"Cluster lacks memory for {model_id}: need {required:.1f} GB, "
                f"have {accumulated:.1f} GB usable across {len(nodes_sorted)} nodes"
            )

        num_nodes = len(selected_nodes)

        # Determine actual strategy
        if strategy == ShardingStrategy.AUTO:
            effective_strategy = ShardingStrategy.PIPELINE_PARALLEL
        else:
            effective_strategy = strategy

        # Calculate per-node memory
        per_node_mem = self.estimate_memory_per_node(model_size, num_nodes, effective_strategy)

        # Build shard assignments
        shard_map: Dict[str, ShardInfo] = {}
        node_endpoints: List[str] = []

        # Estimate total layers based on model size heuristic
        estimated_layers = max(int(model_size / 2.0), 1)
        layers_per_node = estimated_layers / num_nodes

        for i, node in enumerate(selected_nodes):
            role = "head" if i == 0 else "worker"
            start_layer = int(i * layers_per_node)
            end_layer = int((i + 1) * layers_per_node) - 1 if i < num_nodes - 1 else estimated_layers - 1
            endpoint = f"{node.node_id}:{node.api_port}"
            node_endpoints.append(endpoint)

            shard_map[node.node_id] = ShardInfo(
                node_id=node.node_id,
                host=node.node_id,
                api_port=node.api_port,
                role=role,
                layers=f"{start_layer}-{end_layer}",
                estimated_memory_gb=per_node_mem,
            )

        tp_size = 1
        pp_size = num_nodes

        if effective_strategy == ShardingStrategy.TENSOR_PARALLEL:
            tp_size = num_nodes
            pp_size = 1

        config = ShardingConfig(
            model=model_id,
            strategy=effective_strategy,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            nodes=node_endpoints,
            shard_map=shard_map,
            total_memory_required_gb=required,
        )

        logger.info(
            "Sharding plan: %s across %d nodes (%s), TP=%d PP=%d",
            model_id, num_nodes, effective_strategy.value, tp_size, pp_size,
        )
        return config
