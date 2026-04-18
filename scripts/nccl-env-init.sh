#!/bin/bash
# AINode per-node NCCL environment init.
#
# Detects the LOCAL node's InfiniBand HCAs (MOFED-named mlx5_* or stock
# Ubuntu rdma-core rocep*/roceP*), filters to those that are Up and on the
# cluster subnet (AINODE_CLUSTER_SUBNET), and exports NCCL_IB_HCA so
# children of this process inherit the correct per-node value.
#
# Needed because eugr's launch-cluster.sh passes a single cluster-wide
# .env file, so a heterogeneous cluster (mlx5_X on some nodes, rocep* on
# others — or mismatched HCA indices across nodes) cannot be handled by
# one broadcast value.
#
# Two modes:
#   1. Entrypoint mode (no args or non-flag args): detect, export, exec $@.
#      Used via ``docker run --entrypoint`` on each vllm_node container.
#   2. Export-only mode (``--export-only``): emit ``export NCCL_IB_HCA=...``
#      lines on stdout for the caller to ``eval``. Used belt-and-suspenders
#      inside the head's exec-script.sh because ``docker exec`` on the head
#      does not inherit PID-1 runtime env.
#
# Cluster subnet filter: on nodes with both a fabric-switch RoCE port and
# a direct-connect port, this keeps only the switch-facing one so the
# vestigial direct-connect never enters the NCCL ring.
#
# TODO(v0.4.10): Move this script to /usr/local/bin/nccl-env-init.sh
# in Dockerfile.ainode so the shim is available without requiring
# /mnt/shared-models to be mounted. Once done, this NFS-publish path
# becomes optional instead of required.

set -euo pipefail

EXPORT_ONLY=false
if [[ "${1:-}" == "--export-only" ]]; then
    EXPORT_ONLY=true
    shift
fi

log() {
    # In export-only mode, logs go to stderr so the caller's eval isn't
    # polluted. In entrypoint mode, stderr is the normal place.
    echo "[ainode-nccl-init] $*" >&2
}

SUBNET="${AINODE_CLUSTER_SUBNET:-}"

ip_in_subnet() {
    local ip="$1"
    [[ -z "$SUBNET" ]] && return 0
    python3 - "$SUBNET" "$ip" <<'PY' 2>/dev/null
import ipaddress, sys
try:
    net = ipaddress.ip_network(sys.argv[1], strict=False)
    ip = ipaddress.ip_interface(sys.argv[2]).ip
    sys.exit(0 if ip in net else 1)
except Exception:
    sys.exit(1)
PY
}

declare -a KEEP=()
if command -v ibdev2netdev >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ "$line" == *"(Up)" ]] || continue
        hca=$(echo "$line" | grep -oE "(mlx5_[0-9]+|rocep[A-Za-z0-9_]+|roceP[A-Za-z0-9_]+)" | head -1)
        netdev=$(echo "$line" | sed -E 's|.*==>[[:space:]]+([^[:space:]]+).*|\1|')
        [[ -z "$hca" || -z "$netdev" ]] && continue

        if [[ -n "$SUBNET" ]]; then
            cidr=$(ip -o -4 addr show dev "$netdev" 2>/dev/null | awk '{print $4}' | head -1)
            [[ -z "$cidr" ]] && continue
            ip_only="${cidr%/*}"
            ip_in_subnet "$ip_only" || continue
        fi

        KEEP+=("$hca")
    done < <(ibdev2netdev 2>/dev/null)
else
    log "ibdev2netdev not present; NCCL will auto-detect"
fi

if [[ ${#KEEP[@]} -gt 0 ]]; then
    joined=$(IFS=,; echo "${KEEP[*]}")
    if [[ "$EXPORT_ONLY" == "true" ]]; then
        echo "export NCCL_IB_HCA=$joined"
    else
        export NCCL_IB_HCA="$joined"
        log "NCCL_IB_HCA=$joined (node-local, subnet=$SUBNET)"
    fi
else
    log "no matching local HCAs; leaving NCCL_IB_HCA unset (NCCL auto-detect)"
fi

# Entrypoint mode: preserve whatever CMD was passed (e.g. ``sleep infinity``).
# Export-only mode: caller handles the rest.
if [[ "$EXPORT_ONLY" == "false" ]]; then
    exec "$@"
fi
