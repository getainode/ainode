"""AutoData — agentic Δ-filtered synthetic training-data generation (Meta Autodata, MVP).

Pure HTTP orchestration over AInode-served OpenAI-compatible endpoints. No torch/GPU here
— the compute is the served models, so this runs in the slim orchestrator. The loop keeps
only "zone of proximal development" examples: tasks the STRONG solver gets right and the
WEAK solver gets wrong (Δ = I_strong - I_weak == 1).
"""
from .core import run, AutoDataConfig  # noqa: F401
from .valset import valset_lift, evaluate  # noqa: F401  # v2.2 Evalchemy-style val-set objective
