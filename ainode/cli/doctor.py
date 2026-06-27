"""AINode doctor — cluster/node health report (stub).

Full implementation arrives in v0.5.0. See
``ops/slices/nvidia-vllm-engine/runbooks/04-install-ux-spec.md`` § 3.1 for
the target UX — per-section green/yellow/red summary of hardware, network,
RoCE/RDMA, storage, credentials, peers, and the AINode service.

The real work (HCA discovery, MTU probes, NCCL bandwidth check) depends on
``ainode.cluster.hca_discovery`` and related modules that live in a sibling
slice (``nvidia-vllm-engine`` Phase 4 Agent A). This stub is intentionally
dependency-free so ``ainode --help`` works even before those modules land.
"""

from __future__ import annotations


_TARGET_SPEC_DOC = (
    "ops/slices/nvidia-vllm-engine/runbooks/04-install-ux-spec.md § 3.1"
)


def cmd_doctor(args) -> None:
    """Stub for ``ainode doctor``. Prints a pointer + exits 0.

    Intentionally does NOT import from ``ainode.cluster.*`` at module load
    time — those modules are created by Agent A in a parallel workstream.
    When the real implementation lands, replace the body of this function
    with calls into ``ainode.cluster.hca_discovery.discover_hcas()`` etc.
    """

    try:
        # Rich is an existing AINode dependency — safe to import.
        from rich.console import Console
        console = Console()
        console.print(
            "[bold yellow]ainode doctor[/bold yellow] — stub (coming in v0.5.0)\n"
        )
        console.print(
            f"  See {_TARGET_SPEC_DOC} for the target behavior:\n"
            "    hardware / network / RoCE / storage / credentials /\n"
            "    peers / service health in a single 30-second report.\n"
        )
        console.print(
            "  Real implementation depends on [cyan]ainode.cluster.hca_discovery[/cyan]"
            " (being built in parallel).\n"
        )
    except Exception:  # pragma: no cover — rich should always be present
        # If rich isn't available for any reason, fall back to plain print
        # so this command never crashes the CLI.
        print(
            "ainode doctor — stub (coming in v0.5.0). "
            f"See {_TARGET_SPEC_DOC} for target behavior."
        )

    raise SystemExit(0)
