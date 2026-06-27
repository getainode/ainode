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
#
# TODO(v0.4.10): eugr's autodiscover.sh has the same ibdev2netdev
# dependency this shim originally did. If AINode's sysfs detection
# ever returns empty, the launcher's fallback to autodiscover fails
# the same way (ibdev2netdev not in container). Upstream a /sys-based
# autodiscover to eugr, or wrap it.

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

# Get the primary IPv4 address of a netdev by name. Uses python3 with
# SIOCGIFADDR (ioctl 0x8915; see include/uapi/linux/sockios.h) rather
# than the ``ip`` command from iproute2 — ``ip`` is NOT present in the
# eugr vllm-node base image. A prior version of this shim called
# ``ip -o -4 addr show dev $netdev | awk | head``, which died with
# rc=127 under ``set -o pipefail; set -e`` whenever SUBNET was set,
# because the ``ip`` binary wasn't found and the pipe inherited 127.
# python3 is always present in the base image; socket / fcntl / struct
# are stdlib; SIOCGIFADDR is a kernel ABI.
netdev_ipv4() {
    local ifname="$1"
    python3 - "$ifname" <<'PY' 2>/dev/null
import socket, fcntl, struct, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packed = fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR (include/uapi/linux/sockios.h)
        struct.pack('256s', sys.argv[1].encode()[:15]),
    )
    print(socket.inet_ntoa(packed[20:24]))
except OSError:
    pass  # interface has no IPv4 address — emit nothing
PY
}

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

# Detect HCAs via /sys/class/infiniband — kernel-provided, always present
# inside the eugr base image. Prior implementation used ``ibdev2netdev``,
# but that tool ships in MOFED / infiniband-diags on the host, not in the
# container where this shim actually runs.
declare -a KEEP=()
if [[ -d /sys/class/infiniband ]]; then
    for hca_dir in /sys/class/infiniband/*/; do
        [[ -d "$hca_dir" ]] || continue
        hca=$(basename "$hca_dir")
        # Accept only the two naming schemes we expect — the kernel may
        # expose unrelated virtual devices under this path on some hosts.
        [[ "$hca" =~ ^(mlx5_[0-9]+|rocep[A-Za-z0-9_]+|roceP[A-Za-z0-9_]+)$ ]] || continue

        # Port state — parse leading integer from the file and match
        # against the ib_port_state enum (include/rdma/ib_verbs.h):
        #   0 = NOP, 1 = DOWN, 2 = INIT, 3 = ARMED,
        #   4 = ACTIVE (normal up),
        #   5 = ACTIVE_DEFER (also functional for traffic).
        # Accept {4, 5}. Leading-integer parse is stable across kernels
        # where the textual label (e.g. "4: ACTIVE") may vary.
        state_file="$hca_dir/ports/1/state"
        [[ -r "$state_file" ]] || continue
        state=$(cat "$state_file" 2>/dev/null | tr -d '[:space:]')
        state_num="${state%%[!0-9]*}"
        case "$state_num" in
            4|5) ;;
            *)   continue ;;
        esac

        # Netdev via /sys/class/infiniband/<hca>/device/net/<netdev>.
        # Empty when no Ethernet overlay (pure-IB ports); skip those.
        netdev_dir="$hca_dir/device/net"
        [[ -d "$netdev_dir" ]] || continue
        # shellcheck disable=SC2012
        netdev=$(ls "$netdev_dir" 2>/dev/null | head -1)
        [[ -z "$netdev" ]] && continue

        if [[ -n "$SUBNET" ]]; then
            netdev_ip=$(netdev_ipv4 "$netdev")
            [[ -z "$netdev_ip" ]] && continue
            ip_in_subnet "$netdev_ip" || continue
        fi

        KEEP+=("$hca")
    done
else
    log "/sys/class/infiniband not present; NCCL will auto-detect"
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
    # Defensive ``exec --``: terminates bash builtin option parsing so
    # args that start with ``-`` are forwarded as positional args to the
    # exec'd command rather than interpreted as flags to ``exec``. Not
    # the cause of any observed failure (docker inspect confirms Cmd
    # starts with the CMD binary) — but the right long-term pattern for
    # an entrypoint shim that forwards arbitrary arguments.
    exec -- "$@"
fi
