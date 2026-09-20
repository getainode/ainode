#!/usr/bin/env bash
# AINode installer, container-native. The release is resolved at run time.
#
# Usage:
#   curl -fsSL https://ainode.dev/install | bash
#   curl -fsSL https://ainode.dev/install | bash -s -- --job master
#   curl -fsSL https://ainode.dev/install | bash -s -- --job worker
#   AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash -s -- --job master
#
# --job master  Head node: runs the inference engine, serves the web UI,
#               manages the cluster. Set a model via the web UI after install.
# --job worker  Worker node: no model, no engine on startup. Announces itself
#               to the cluster and waits for the head to assign work.
# (default)     Solo node: standalone, pick a model via the web UI.
# --dry-run     Render config.json, the systemd unit and the host wrapper into
#               $AINODE_HOME and stop. Pulls nothing, starts nothing, needs no
#               docker, no GPU and no sudo. Use it to see exactly what an
#               install would write; the test suite runs the same path.

set -euo pipefail

# -- Defaults ---------------------------------------------------------------
# Version + image are resolved from the highest numeric GHCR tag below (unless
# the caller pins them explicitly). Leaving these empty is the signal to resolve.
AINODE_VERSION="${AINODE_VERSION:-}"
AINODE_IMAGE="${AINODE_IMAGE:-}"
# The vLLM ENGINE image, pre-pulled so the first model launch is not the thing
# that waits on a multi-GB download. Empty (the default) means "ask the AINode
# image we just pulled", which keeps ONE home for the value: NVIDIA_VLLM_IMAGE in
# ainode/engine/backends/nvidia.py, the same constant the backend launches from.
# Installs up to 0.5.25 pre-pulled nvcr.io/nvidia/vllm:26.02-py3 here instead:
# ~15 GB plus an NGC login for an image no code path has ever run (issue #164).
# Set a tag to pre-pull a different one, or "skip" to pull no engine image.
AINODE_NVIDIA_IMAGE="${AINODE_NVIDIA_IMAGE:-}"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
AINODE_PEERS="${AINODE_PEERS:-}"           # comma-separated IPs
AINODE_SSH_USER="${AINODE_SSH_USER:-$USER}"
AINODE_JOB="${AINODE_JOB:-solo}"          # solo | master | worker
SETUP_SSH="false"
USER_MODE="false"
DRY_RUN="false"

# HF token location (read-only hint; we never write it).
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

# -- Arg parsing ------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --job)
            shift
            case "${1:-}" in
                master) AINODE_JOB="master" ;;
                worker) AINODE_JOB="worker" ;;
                solo)   AINODE_JOB="solo" ;;
                *) echo "Unknown --job value: ${1:-} (use master, worker, or solo)" >&2; exit 2 ;;
            esac
            ;;
        --setup-ssh) SETUP_SSH="true" ;;
        --user)      USER_MODE="true" ;;
        --dry-run)   DRY_RUN="true" ;;
        -h|--help)
            sed -n '1,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$*"; }
die() { printf "\033[1;31mXX\033[0m %s\n" "$*" >&2; exit 1; }

# Resolve the highest numeric GHCR tag anonymously (public image). Echoes the
# tag on success; returns non-zero if resolution fails (caller falls back).
resolve_latest_tag() {
    local token tags
    token=$(curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:getainode/ainode:pull" 2>/dev/null \
        | sed -E 's/.*"token":"([^"]+)".*/\1/') || return 1
    [ -n "$token" ] || return 1
    tags=$(curl -fsSL -H "Authorization: Bearer $token" \
        "https://ghcr.io/v2/getainode/ainode/tags/list" 2>/dev/null) || return 1
    echo "$tags" | tr ',' '\n' \
        | grep -oE '"[0-9]+\.[0-9]+\.[0-9]+"' | tr -d '"' \
        | sort -t. -k1,1n -k2,2n -k3,3n | tail -1
}

# -- Cluster-interface detection --------------------------------------------
# The NIC that NCCL / Ray / Gloo bind to is hardware specific: a DGX Spark
# names its direct-connect port enP2p1s0f1np1, an ASUS GX10 names the same
# class of port enp1s0f0np0. The installer used to hardcode the Spark name,
# so on anything else the engine came up bound to nothing and the user had
# to find the real name with `ip -br addr` and hand-edit config.json
# (github.com/getainode/ainode issues #34, #61). Detect it instead, using
# the same ranking as ainode/cluster/netdev.py.

# Overridable so the detection can be exercised against a fake tree.
SYS_CLASS_NET="${SYS_CLASS_NET:-/sys/class/net}"

# Devices that are never the cluster fabric (bridges, veth, VPN/overlay).
is_virtual_netdev() {
    case "$1" in
        lo|docker*|br-*|veth*|virbr*|tailscale*|wg*|tun*|tap*|cni*|flannel*|cali*|kube*|lxc*|zt*)
            return 0 ;;
    esac
    return 1
}

# Echo the IPv4 address bound to $1, or nothing.
netdev_ipv4() {
    ip -o -4 addr show dev "$1" 2>/dev/null \
        | awk '{for (i=1; i<=NF; i++) if ($i == "inet") {split($(i+1), a, "/"); print a[1]; exit}}' \
        || true
}

# Echo the cluster fabric NIC: an RDMA-capable device carrying an IPv4
# first, then the default-route device, else nothing (empty string means
# "autodetect at startup", which AINode now does).
detect_cluster_interface() {
    local path name ip4
    for path in "$SYS_CLASS_NET"/*; do
        [ -e "$path" ] || continue
        name="${path##*/}"
        if is_virtual_netdev "$name"; then continue; fi
        [ -e "$path/device/infiniband" ] || continue
        ip4="$(netdev_ipv4 "$name")"
        if [ -n "$ip4" ]; then
            printf '%s\n' "$name"
            return 0
        fi
    done
    name="$(ip route show default 2>/dev/null \
        | awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}' || true)"
    if [ -n "$name" ] && ! is_virtual_netdev "$name"; then
        printf '%s\n' "$name"
        return 0
    fi
    printf '\n'
    return 0
}

# -- 1. Preflight -----------------------------------------------------------
# Every check here is about the machine being installed ON, so a dry run (which
# only renders files into $AINODE_HOME) skips the lot: no Linux, no docker, no
# GPU and no sudo needed to see what an install would write.
preflight() {
    log "Checking prerequisites"
    [ "$(uname -s)" = "Linux" ] || die "AINode requires Linux (detected $(uname -s))"
    command -v docker >/dev/null 2>&1 || die "docker not found. Install: https://docs.docker.com/engine/install/"
    docker info >/dev/null 2>&1 || die "docker daemon not reachable. Run 'sudo systemctl start docker' or add \$USER to the 'docker' group"

    # /mnt/shared-models is the shared model store: it is where downloaded weights
    # are staged, and a node whose models_dir points at it keeps the engine
    # container's HF cache there too, so the path has to exist and has to have room
    # for the weights. The systemd unit this installer renders bind-mounts it into
    # the AINode container unconditionally (see EXEC_START below), with `--mount
    # type=bind` rather than `-v` so a missing source fails at container start
    # instead of being invented as an empty root-owned directory. Check it here so
    # the requirement is stated up front rather than arriving as a cryptic docker
    # error on first start. For clusters: an NFS mount from the master's model
    # storage, so every node reads one copy. For a single node: a directory is
    # enough.
    # Note: the reason given here through v0.5.x was the per-node NCCL init shim,
    # which was the retired eugr launcher's use of the path and not the default
    # backend's. The mount, and therefore this check, is still unconditional; the
    # open follow-up is making it conditional on shared storage being configured,
    # not dropping it while the unit still names it.
    #
    # Real newlines, not "\n": die() prints its message through printf's %s, which
    # does not expand escapes, so a "\n" here reaches the operator's terminal as
    # two literal characters.
    if [ ! -d /mnt/shared-models ]; then
        die "AINode requires /mnt/shared-models to exist: it is the model store the service bind-mounts, where downloaded weights live.
  Create it before re-running this installer:
    sudo mkdir -p /mnt/shared-models
  For clusters, mount shared model storage there (NFS from master recommended)."
    fi

    # GPU check (nvidia-container-toolkit). AINode targets NVIDIA GB10; skip if
    # missing, let the container fail fast with a clear error.
    if ! docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        warn "Could not run a GPU container. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        warn "Continuing anyway. AINode will error at start time if the GPU isn't accessible."
    fi
}

if [ "$DRY_RUN" = "true" ]; then
    log "Dry run: skipping prerequisite checks (nothing is pulled or started)"
else
    preflight
fi

# -- 2. Create config dir, pull image, write initial config ----------------
log "Preparing $AINODE_HOME"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

# Resolve the image to a PINNED numeric tag (never a floating :latest, which
# has drifted to ancient builds in the past). Explicit AINODE_IMAGE wins.
if [ -z "$AINODE_IMAGE" ]; then
    if RESOLVED_TAG="$(resolve_latest_tag)" && [ -n "$RESOLVED_TAG" ]; then
        AINODE_VERSION="$RESOLVED_TAG"
        AINODE_IMAGE="ghcr.io/getainode/ainode:${RESOLVED_TAG}"
        log "Resolved latest GHCR tag: $RESOLVED_TAG"
    else
        AINODE_IMAGE="ghcr.io/getainode/ainode:latest"
        warn "Could not resolve latest GHCR tag — falling back to :latest"
    fi
fi
# Derive the banner version from the pinned tag when not otherwise set.
[ -n "$AINODE_VERSION" ] || AINODE_VERSION="${AINODE_IMAGE##*:}"

if [ "$DRY_RUN" = "true" ]; then
    log "Dry run: would pull $AINODE_IMAGE (AINode orchestrator, slim, ~400 MB)"
else
    log "Pulling $AINODE_IMAGE (AINode orchestrator, slim, ~400 MB)"
    docker pull "$AINODE_IMAGE"

    # Pin the image for the systemd unit's EnvironmentFile so `ainode update`
    # can swap it later without re-rendering the unit. Written atomically
    # (temp+rename) so the unit never reads a half-written line.
    printf 'AINODE_IMAGE=%s\n' "$AINODE_IMAGE" > "$AINODE_HOME/image.env.tmp"
    mv -f "$AINODE_HOME/image.env.tmp" "$AINODE_HOME/image.env"
fi

# -- 2a. Pre-pull the ENGINE image ------------------------------------------
# AINode is two images, not one: this slim orchestrator, plus a vLLM ENGINE
# container per loaded model. Pre-pulling the engine here is the difference
# between "click Launch, wait a couple of minutes" and "click Launch, wait
# twenty". Which image that is comes out of the AINode image we just pulled, so
# the installer cannot drift from the code that launches it.
resolve_engine_image() {
    docker run --rm --entrypoint python3 "$AINODE_IMAGE" -c \
        'from ainode.engine.backends.nvidia import NVIDIA_VLLM_IMAGE as i; print(i)' \
        2>/dev/null | tr -d '[:space:]'
}

if [ "$AINODE_NVIDIA_IMAGE" = "skip" ]; then
    warn "Skipping the engine image pre-pull (AINODE_NVIDIA_IMAGE=skip)."
    warn "  The first model launch pulls it instead."
elif [ "$DRY_RUN" = "true" ]; then
    log "Dry run: skipping the engine image pre-pull"
else
    if [ -z "$AINODE_NVIDIA_IMAGE" ]; then
        AINODE_NVIDIA_IMAGE="$(resolve_engine_image || true)"
    fi
    if [ -z "$AINODE_NVIDIA_IMAGE" ]; then
        warn "Could not read the default engine image out of $AINODE_IMAGE."
        warn "  Skipping the pre-pull; the first model launch pulls it."
    elif docker image inspect "$AINODE_NVIDIA_IMAGE" >/dev/null 2>&1; then
        log "Engine image already present: $AINODE_NVIDIA_IMAGE"
    else
        log "Pulling the vLLM engine image: $AINODE_NVIDIA_IMAGE"
        log "  (~8.5 GB to download, ~22 GB on disk. Minutes on a fast link,"
        log "   longer on a slow one. One-time: the first model launch pays for"
        log "   it otherwise.)"
        if docker pull "$AINODE_NVIDIA_IMAGE"; then
            log "  Engine image pulled: $AINODE_NVIDIA_IMAGE"
        else
            warn "  docker pull $AINODE_NVIDIA_IMAGE failed."
            warn "  AINode still installs; the first model launch retries the pull."
        fi
    fi
fi

# -- 2b. HuggingFace token hint --------------------------------------------
# We never capture or write the token — just nudge the user so gated-model
# downloads don't fail with a cryptic 403 at first launch.
if [ -n "${HF_TOKEN:-}" ] || [ -r "$HF_TOKEN_FILE" ]; then
    log "HuggingFace token: OK (env var or $HF_TOKEN_FILE)"
else
    warn "HuggingFace token NOT configured."
    warn "  Gated models (Llama, Nemotron, Gemma, ...) will fail to download without this."
    warn "  Fix:  hf auth login          (stores at $HF_TOKEN_FILE)"
    warn "    or: export HF_TOKEN=hf_... (in your shell profile)"
fi

# Write initial config.json if not already present.
# No model is set — the user picks one via the web UI after install.
# Job role determines whether this node runs an engine (master/solo)
# or just announces itself and waits for work (worker).
if [ ! -f "$AINODE_HOME/config.json" ]; then
    log "Writing initial config (job: $AINODE_JOB)"
    case "$AINODE_JOB" in
        master)
            DIST_MODE="head"
            PEER_IPS=$([ -n "$AINODE_PEERS" ] && echo "\"$(echo "$AINODE_PEERS" | sed 's/,/","/g')\"" || echo "")
            ;;
        worker)
            DIST_MODE="member"
            PEER_IPS=""
            ;;
        *)
            DIST_MODE="solo"
            PEER_IPS=""
            ;;
    esac

    CLUSTER_IFACE="$(detect_cluster_interface)"
    if [ -n "$CLUSTER_IFACE" ]; then
        log "Cluster interface: $CLUSTER_IFACE (auto-detected; edit cluster_interface in ~/.ainode/config.json to change)"
    else
        log "Cluster interface: none detected (AINode re-detects at startup; edit cluster_interface in ~/.ainode/config.json to pin one)"
    fi

    # Three of these are load-bearing enough to be written out rather than left
    # to the code defaults, so the file says what this node is doing:
    #
    #   engine_backend "nvidia" runs each model as its own vLLM container. It is
    #   the code default now too, but a fresh install is where getting it wrong
    #   costs the most: up to 0.5.25 the default was "eugr", which execs a `vllm`
    #   binary the shipped image does not contain, so every Launch click on a
    #   new node answered HTTP 500 until somebody hand-edited this file (#164).
    #
    #   model null means NO model on this node yet, which is what the comment above
    #   has always claimed and the file never said: leaving the key out inherits
    #   NodeConfig's default, so every fresh node booted straight into a launch of
    #   meta-llama/Llama-3.2-3B-Instruct that nobody asked for. That repo is
    #   gated, so on a node without HF access it fails with a 401 the user cannot
    #   place, and on a node WITH access it downloads 6 GB and reserves 0.6 of the
    #   GPU before the user has picked anything.
    #
    #   gpu_memory_utilization 0.6 leaves room for a SECOND model. vLLM reserves
    #   this fraction of the node's memory for its KV cache however small the
    #   model is, and the stacked-load guard (models/api_routes.py) refuses a
    #   load whose total would pass 0.90. At the 0.9 this installer used to
    #   write, every stacked load on a brand-new node was a 409; 0.6 still gives
    #   a single model ~73 GB of cache on a 122 GB GB10 and leaves 0.30 for one
    #   stacked neighbour.
    cat > "$AINODE_HOME/config.json" << CONFIG
{
  "node_name": "$(hostname)",
  "onboarded": true,
  "model": null,
  "engine_backend": "nvidia",
  "distributed_mode": "${DIST_MODE}",
  "peer_ips": [${PEER_IPS}],
  "cluster_id": "ainode-cluster",
  "cluster_interface": "${CLUSTER_IFACE}",
  "ssh_user": "${AINODE_SSH_USER}",
  "api_port": 8000,
  "web_port": 3000,
  "discovery_port": 5679,
  "gpu_memory_utilization": 0.6
}
CONFIG
    log "Node configured as: $AINODE_JOB (distributed_mode=$DIST_MODE)"
fi

# -- 3. Optional passwordless SSH bootstrap ---------------------------------
# Needed only for distributed (multi-node) mode: the head SSHes into each peer
# and `docker run`s an engine container there (engine/backends/nvidia.py).
if [ "$DRY_RUN" = "true" ] && { [ "$SETUP_SSH" = "true" ] || [ -n "$AINODE_PEERS" ]; }; then
    log "Dry run: skipping the passwordless SSH bootstrap"
elif [ "$SETUP_SSH" = "true" ] || [ -n "$AINODE_PEERS" ]; then
    log "Setting up passwordless SSH for distributed mode"
    if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" >/dev/null
        log "Generated $HOME/.ssh/id_ed25519"
    fi
    IFS=',' read -ra PEER_LIST <<< "$AINODE_PEERS"
    for peer in "${PEER_LIST[@]}"; do
        [ -z "$peer" ] && continue
        log "ssh-copy-id ${AINODE_SSH_USER}@${peer} (you may be prompted once)"
        ssh-copy-id -o StrictHostKeyChecking=accept-new \
            -i "$HOME/.ssh/id_ed25519.pub" \
            "${AINODE_SSH_USER}@${peer}" || warn "ssh-copy-id to $peer failed"
        if ssh -o BatchMode=yes -o ConnectTimeout=5 \
                "${AINODE_SSH_USER}@${peer}" true 2>/dev/null; then
            log "  verified passwordless SSH to $peer"
        else
            warn "  passwordless SSH to $peer NOT working — distributed launch will fail"
        fi
    done
fi

# -- 4. Install systemd unit ------------------------------------------------
log "Installing systemd service"
SERVICE_ARGS=()
if [ "$USER_MODE" = "true" ]; then
    SERVICE_ARGS+=("--user")
fi

# Write the systemd unit file directly from install.sh.
# We do NOT use `docker run ... ainode service install` here because the
# container has no systemd bus. The unit content is simple enough to
# generate inline — it's just a docker run command as ExecStart.
log "Writing systemd unit file"

WANTED_BY="multi-user.target"
UNIT_DIR="/etc/systemd/system"
if [ "$USER_MODE" = "true" ]; then
    WANTED_BY="default.target"
    UNIT_DIR="$HOME/.config/systemd/user"
fi

[ "$DRY_RUN" = "true" ] || mkdir -p "$UNIT_DIR"

# The unit is rendered to a staging path first, then moved into place (a dry run
# leaves it in $AINODE_HOME and moves nothing).
UNIT_STAGING="/tmp/ainode.service"
[ "$DRY_RUN" = "true" ] && UNIT_STAGING="$AINODE_HOME/ainode.service"

# ExecStart references ${AINODE_IMAGE}, escaped so systemd expands it rather
# than this shell. The pinned Environment default is overridden by image.env, so
# `ainode update` swaps the image without re-rendering this unit.
EXEC_START="/usr/bin/docker run --rm --name ainode \
 --network=host --gpus all --ipc=host --shm-size=64g --pid=host \
 -v ${AINODE_HOME}:/root/.ainode \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v ${HOME}/.ssh:/host-ssh:ro \
 -v ${HOME}/.docker:/root/.docker:ro \
 --mount type=bind,source=/mnt/shared-models,target=/mnt/shared-models,bind-propagation=rshared \
 -e AINODE_HOME=/root/.ainode \
 -e AINODE_HOST_HOME=${AINODE_HOME} \
 -e AINODE_UNIT_SWAPPABLE=1 \
 -e HF_HUB_ENABLE_HF_TRANSFER=1 \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
 \${AINODE_IMAGE}"

cat > "$UNIT_STAGING" << UNIT
[Unit]
Description=AINode — Local AI inference platform
Documentation=https://ainode.dev
After=network.target docker.service nvidia-persistenced.service
Wants=docker.service nvidia-persistenced.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ainode
Environment=AINODE_IMAGE=${AINODE_IMAGE}
EnvironmentFile=-${AINODE_HOME}/image.env
ExecStart=${EXEC_START}
ExecStop=/usr/bin/docker stop -t 30 ainode
Restart=always
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=45
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=AINODE_HOME=${AINODE_HOME}

[Install]
WantedBy=${WANTED_BY}
UNIT

if [ "$DRY_RUN" = "true" ]; then
    log "Dry run: unit rendered at $UNIT_STAGING (not installed, systemd untouched)"
elif [ "$USER_MODE" = "true" ]; then
    mv "$UNIT_STAGING" "$UNIT_DIR/ainode.service"
    systemctl --user daemon-reload
    systemctl --user enable --now ainode.service
    log "  (consider: sudo loginctl enable-linger $USER, so the service survives logout)"
else
    sudo mv "$UNIT_STAGING" "$UNIT_DIR/ainode.service"
    sudo systemctl daemon-reload
    sudo systemctl enable --now ainode.service
fi

# -- 5. Install host-side `ainode` wrapper ----------------------------------
# The ainode CLI lives inside the container. This thin wrapper on the host
# dispatches `ainode update` to docker-pull + systemctl-restart (the only
# operations that must happen outside the container), and forwards every
# other command to `docker exec ainode ainode ...` so users never need to
# type the docker command themselves.
WRAPPER_PATH="/usr/local/bin/ainode"
WRAPPER_SUDO="sudo"
if [ "$DRY_RUN" = "true" ]; then
    WRAPPER_PATH="$AINODE_HOME/ainode-wrapper"
    WRAPPER_SUDO=""
    log "Dry run: rendering the host wrapper at $WRAPPER_PATH"
else
    [ -w "$(dirname "$WRAPPER_PATH")" ] && WRAPPER_SUDO=""
    log "Installing $WRAPPER_PATH host wrapper"
fi

$WRAPPER_SUDO tee "$WRAPPER_PATH" >/dev/null <<WRAPPER
#!/usr/bin/env bash
# AINode host wrapper, installed by install.sh. Not user-editable.
set -euo pipefail
AINODE_IMAGE="\${AINODE_IMAGE:-ghcr.io/getainode/ainode:latest}"
# NOT resolved to \$HOME/.ainode here: see resolve_ainode_home below. Under sudo
# \$HOME is root's, and pinning the image into /root/.ainode is a silent no-op.
AINODE_HOME_ENV="\${AINODE_HOME:-}"
AINODE_SERVICE="ainode.service"

# Resolve the highest numeric GHCR tag anonymously (public image).
resolve_latest_tag() {
    local token tags
    token=\$(curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:getainode/ainode:pull" 2>/dev/null \\
        | sed -E 's/.*"token":"([^"]+)".*/\\1/') || return 1
    [ -n "\$token" ] || return 1
    tags=\$(curl -fsSL -H "Authorization: Bearer \$token" \\
        "https://ghcr.io/v2/getainode/ainode/tags/list" 2>/dev/null) || return 1
    echo "\$tags" | tr ',' '\\n' \\
        | grep -oE '"[0-9]+\\.[0-9]+\\.[0-9]+"' | tr -d '"' \\
        | sort -t. -k1,1n -k2,2n -k3,3n | tail -1
}

# Which .ainode does the running service actually read? \`sudo ainode update\`
# used to answer "/root/.ainode", because the wrapper resolved \$HOME after sudo
# had already changed it: the pull succeeded, image.env was written where nothing
# reads it, and systemd restarted the OLD image with nothing said about it. The
# unit bakes its AINODE_HOME in at install time, so ask the unit (issue #164).
unit_ainode_home() {
    local f
    # Overridable so the test suite can point this at a temp unit; a host that
    # HAS a real /etc/systemd/system/ainode.service would otherwise answer for it
    # (same reason install.sh itself takes \$SYS_CLASS_NET).
    local units="\${AINODE_UNIT_FILES:-/etc/systemd/system/ainode.service \$HOME/.config/systemd/user/ainode.service}"
    for f in \$units; do
        [ -r "\$f" ] || continue
        sed -n 's/^Environment=AINODE_HOME=//p' "\$f" | tail -1
        return 0
    done
    return 0
}

# Home directory of \$1, without trusting \$HOME.
user_home() {
    local u="\$1" h=""
    if command -v getent >/dev/null 2>&1; then
        h="\$(getent passwd "\$u" 2>/dev/null | cut -d: -f6 || true)"
    fi
    if [ -z "\$h" ]; then
        h="\$(eval printf '%s' "~\$u" 2>/dev/null || true)"
        [ "\$h" = "~\$u" ] && h=""
    fi
    [ -n "\$h" ] || return 1
    printf '%s\n' "\$h"
}

# The .ainode whose image.env systemd reads. Non-zero exit means "cannot tell",
# and the caller must refuse rather than write a file nothing will read.
resolve_ainode_home() {
    local h
    if [ -n "\$AINODE_HOME_ENV" ]; then
        printf '%s\n' "\$AINODE_HOME_ENV"; return 0
    fi
    h="\$(unit_ainode_home)"
    if [ -n "\$h" ]; then printf '%s\n' "\$h"; return 0; fi
    if [ "\$(id -u)" = "0" ] && [ -n "\${SUDO_USER:-}" ] && [ "\$SUDO_USER" != "root" ]; then
        if h="\$(user_home "\$SUDO_USER")"; then
            printf '%s\n' "\$h/.ainode"; return 0
        fi
        return 1
    fi
    printf '%s\n' "\$HOME/.ainode"
}

is_user_mode() {
    systemctl --user is-enabled "\$AINODE_SERVICE" >/dev/null 2>&1
}

# The web port this node serves /api/status on, from the config the service
# reads. 3000 is what the installer writes and the fallback when the file is
# missing or says nothing.
node_web_port() {
    local home="\$1" port=""
    if [ -r "\$home/config.json" ]; then
        port=\$(sed -n 's/.*"web_port"[[:space:]]*:[[:space:]]*\\([0-9]\\{1,\\}\\).*/\\1/p' \\
            "\$home/config.json" 2>/dev/null | head -1) || true
    fi
    printf '%s\\n' "\${port:-3000}"
}

# The version /api/status reports. Empty output plus non-zero when the node does
# not answer at all. "version" is the only key with that exact name in the
# payload (driver_version and friends do not match the leading quote).
api_version() {
    local url="\$1" body=""
    body=\$(curl -fsS --max-time 5 "\$url" 2>/dev/null) || return 1
    printf '%s\\n' "\$body" \\
        | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p' | head -1
}

# Wait for the node to come back reporting \$2. Prints the last version it saw.
# Non-zero when the target never appears, which is the point: "Update complete"
# used to print for an update that never applied, because nothing ever asked the
# running node what version it was (the pull landed, image.env was written where
# the unit does not read it, and the OLD container came back).
wait_for_version() {
    local url="\$1" target="\$2" timeout="\${3:-180}" waited=0 seen=""
    while [ "\$waited" -lt "\$timeout" ]; do
        seen=\$(api_version "\$url") || seen=""
        if [ -n "\$seen" ] && [ "\$seen" = "\$target" ]; then
            printf '%s\\n' "\$seen"
            return 0
        fi
        sleep 3
        waited=\$(( waited + 3 ))
    done
    printf '%s\\n' "\$seen"
    return 1
}

print_update_help() {
    cat <<UPDATEHELP
Usage: ainode update [version] [--keep-images N]

Pull a release, pin it for the systemd unit, restart, and verify.

  version            release to install (default: the highest numeric GHCR tag)
  --keep-images N    rollback generations of the AINode image to keep below the
                     new one (default 1, so the release you updated FROM stays
                     on disk and 'ainode update <older>' can go back).
                     0 removes every older AINode image. Engine images and
                     ainode-base are never touched.
  -h, --help         this text

After the restart the wrapper waits for this node's /api/status to report the
version it just installed, and exits non-zero if it does not: an update that did
not apply must not report success. Only then are the images it replaced removed,
so a failed update still has something to fall back to.

Environment:
  AINODE_HOME                     .ainode the unit reads (normally detected)
  AINODE_KEEP_IMAGES              default for --keep-images
  AINODE_UPDATE_VERIFY_TIMEOUT    seconds to wait for the new version (180)
UPDATEHELP
}

restart_service() {
    if is_user_mode; then
        systemctl --user restart "\$AINODE_SERVICE"
    else
        sudo systemctl restart "\$AINODE_SERVICE"
    fi
}

case "\${1:-}" in
    update)
        # Optional explicit version: 'ainode update 0.5.0'. Otherwise resolve
        # the highest numeric GHCR tag. Never pull a floating :latest here.
        shift
        TARGET_VERSION=""
        KEEP_IMAGES="\${AINODE_KEEP_IMAGES:-1}"
        VERIFY_TIMEOUT="\${AINODE_UPDATE_VERIFY_TIMEOUT:-180}"
        while [ \$# -gt 0 ]; do
            case "\$1" in
                -h|--help) print_update_help; exit 0 ;;
                --keep-images=*) KEEP_IMAGES="\${1#*=}"; shift ;;
                --keep-images)
                    if [ \$# -lt 2 ]; then
                        echo "XX --keep-images needs a number" >&2
                        exit 2
                    fi
                    KEEP_IMAGES="\$2"; shift 2 ;;
                --*)
                    echo "XX Unknown option: \$1" >&2
                    print_update_help >&2
                    exit 2 ;;
                *) TARGET_VERSION="\$1"; shift ;;
            esac
        done
        case "\$KEEP_IMAGES" in
            ''|*[!0-9]*)
                echo "XX --keep-images needs a non-negative integer, got '\$KEEP_IMAGES'" >&2
                exit 2 ;;
        esac
        if [ -z "\$TARGET_VERSION" ]; then
            TARGET_VERSION="\$(resolve_latest_tag || true)"
        fi
        if [ -n "\$TARGET_VERSION" ]; then
            PULL_IMAGE="ghcr.io/getainode/ainode:\$TARGET_VERSION"
        else
            PULL_IMAGE="\$AINODE_IMAGE"
            echo "!! Could not resolve a version — pulling \$PULL_IMAGE"
        fi
        # Resolve WHERE to pin it before pulling anything: a sudo-ainode-update
        # that cannot tell must say so, not write /root/.ainode and
        # report success while systemd relaunches the old image.
        if ! AINODE_HOME="\$(resolve_ainode_home)"; then
            echo "XX Cannot tell which .ainode this service reads." >&2
            echo "   Running as root via sudo (SUDO_USER=\${SUDO_USER:-unset}) with no" >&2
            echo "   resolvable home and no systemd unit naming an AINODE_HOME, so" >&2
            echo "   pinning the image would write a file nothing reads." >&2
            echo "   Re-run as the install user, or name it:" >&2
            echo "     sudo AINODE_HOME=/home/<user>/.ainode ainode update" >&2
            exit 1
        fi
        echo "==> Pulling \$PULL_IMAGE"
        docker pull "\$PULL_IMAGE"
        # Pin it for the systemd unit's EnvironmentFile, then restart. Written
        # atomically (temp+rename) so the unit never reads a half-written line.
        mkdir -p "\$AINODE_HOME"
        printf 'AINODE_IMAGE=%s\n' "\$PULL_IMAGE" > "\$AINODE_HOME/image.env.tmp"
        mv -f "\$AINODE_HOME/image.env.tmp" "\$AINODE_HOME/image.env"
        echo "==> Pinned \$PULL_IMAGE in \$AINODE_HOME/image.env"
        echo "==> Restarting \$AINODE_SERVICE"
        if is_user_mode || systemctl is-active --quiet "\$AINODE_SERVICE" 2>/dev/null; then
            restart_service
        else
            echo "==> Pulled and pinned, but the service is not running, so nothing"
            echo "    was restarted and no old image was removed."
            echo "    Start it with: sudo systemctl start ainode"
            exit 0
        fi

        # Ask the node what it is running before claiming anything happened.
        STATUS_URL="http://127.0.0.1:\$(node_web_port "\$AINODE_HOME")/api/status"
        if [ -z "\$TARGET_VERSION" ]; then
            echo "!! No version was resolved, so there is nothing to verify against."
            echo "   /api/status reports: \$(api_version "\$STATUS_URL" || echo 'no answer')"
            echo "   Images were left alone."
            exit 0
        fi
        echo "==> Waiting up to \${VERIFY_TIMEOUT}s for \$STATUS_URL to report \$TARGET_VERSION"
        RUNNING_VERSION="\$(wait_for_version "\$STATUS_URL" "\$TARGET_VERSION" "\$VERIFY_TIMEOUT")" || {
            echo "XX Update did NOT apply. \$TARGET_VERSION was pulled and pinned in" >&2
            echo "   \$AINODE_HOME/image.env, but after the restart /api/status reports" >&2
            echo "   '\${RUNNING_VERSION:-nothing}'." >&2
            echo "   Nothing was removed. Check: systemctl status ainode; docker ps;" >&2
            echo "   grep AINODE_HOME /etc/systemd/system/ainode.service" >&2
            exit 1
        }
        echo "==> Node is serving \$RUNNING_VERSION"

        # Only now, with the new version proven serving, reclaim what it replaced
        # (#184). Keeps \$KEEP_IMAGES rollback generation(s); the decision itself
        # lives in ainode.core.image_prune, inside the container that is now up.
        echo "==> Removing AINode images older than \$RUNNING_VERSION (keeping \$KEEP_IMAGES)"
        docker exec ainode ainode prune-images \\
            --keep-images "\$KEEP_IMAGES" --current "\$PULL_IMAGE" || \\
            echo "!! Could not prune; images were left alone. Retry: ainode prune-images"

        echo "==> Update complete. Version: \$RUNNING_VERSION"
        ;;
    "" | -h | --help)
        cat <<HELP
AINode host CLI. Commands that change the running container (update,
restart) run on the host; everything else is forwarded to the container.

Usage: ainode <command> [args...]

Host-side:
  update [version] [--keep-images N]
                          pull the release, pin it for the systemd unit, restart,
                          verify /api/status reports it, then remove the images it
                          replaced (keeping N rollback generations, default 1).
                          Exits non-zero if the node does not come back on the new
                          version. Under sudo the image is pinned in the .ainode
                          the UNIT reads, not root's; override with
                          AINODE_HOME=/home/<user>/.ainode if it guesses wrong.
                          Run 'ainode update --help' for the rest.
  --version               print the wrapper's pinned image tag

Container-side (forwarded via docker exec):
  status, models, config, logs, service, auth, ...

Run \`ainode status\` to see the live container commands.
HELP
        ;;
    --version)
        echo "ainode wrapper (image: \$AINODE_IMAGE)"
        docker exec ainode ainode --version 2>/dev/null || \\
            docker run --rm --entrypoint ainode "\$AINODE_IMAGE" --version 2>/dev/null || true
        ;;
    *)
        # Forward everything else into the running container. If the
        # container isn't up, fall back to a one-shot docker run so
        # \`ainode --help\`, \`ainode service install\`, etc. still work.
        if docker exec ainode true 2>/dev/null; then
            exec docker exec -it ainode ainode "\$@"
        else
            # Same sudo trap as update: mount the .ainode the unit uses.
            CONF_HOME="\$(resolve_ainode_home || true)"
            [ -n "\$CONF_HOME" ] || CONF_HOME="\$HOME/.ainode"
            exec docker run --rm -it \\
                --entrypoint ainode \\
                -v "\$CONF_HOME":/root/.ainode \\
                "\$AINODE_IMAGE" "\$@"
        fi
        ;;
esac
WRAPPER

$WRAPPER_SUDO chmod +x "$WRAPPER_PATH"

# -- Banner -----------------------------------------------------------------
if [ "$DRY_RUN" = "true" ]; then
    printf '\n'
    log "Dry run complete. Rendered under $AINODE_HOME:"
    log "  config.json      the node config an install would write"
    log "  ainode.service   the systemd unit (not installed)"
    log "  ainode-wrapper   the /usr/local/bin/ainode host wrapper"
    printf '\n'
    exit 0
fi

printf '\n'
printf '    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '    \033[1;32m  AINode v%s installed!\033[0m\n' "${AINODE_VERSION}"
printf '    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\n'
printf '    Web:     http://localhost:3000\n'
printf '    API:     http://localhost:8000/v1\n'
printf '    Access:  API open, no key set. Require one in Config > API access.\n'
printf '    Status:  ainode status\n'
printf '    Logs:    ainode logs -f\n'
printf '    Update:  ainode update\n'
printf '\n'
printf '    Made in Texas\n'
printf '\n'
