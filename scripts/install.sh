#!/usr/bin/env bash
# AINode installer, container-native. The release is resolved at run time.
#
# Usage:
#   curl -fsSL https://ainode.dev/install | bash
#   curl -fsSL https://ainode.dev/install | bash -s -- --job master
#   curl -fsSL https://ainode.dev/install | bash -s -- --job worker
#   AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash -s -- --job master
#   AINODE_JOIN="10.0.0.1:3000:<token>" curl -fsSL https://ainode.dev/install | bash
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
#
# AINODE_JOIN   "<master-host>[:<port>]:<token>", a join token minted on the
#               master with `ainode cluster token`. Installs this node and then
#               joins it: the cluster id, the cluster secret, the discovery port
#               and the master address are written for you, so a new node needs
#               no hand-edited config.json. Mutually exclusive with --job master.
# AINODE_CLUSTER_SECRET
#               The shared secret this cluster signs discovery with. Every node in
#               one cluster must carry the SAME value. Left unset, a fresh install
#               generates its own, which is right for the FIRST node and wrong for
#               a second one installed independently: two nodes with different
#               secrets are invisible to each other. So a second node either joins
#               (AINODE_JOIN, which writes the master's value) or is installed with
#               the master's value pasted here. It is also what node-to-node calls
#               authenticate with once auth is on (the fleet key, ainode/auth/fleet.py).
# AINODE_AUTH   "on" (default) or "off", and it applies to a FRESH install only.
#               On: the installer mints one API key, stores its SHA-256 in
#               $AINODE_HOME/auth.json with auth switched on, and prints the key
#               once at the end. Every /api and /v1 route then needs
#               "Authorization: Bearer <key>", the dashboard asks for it on first
#               open, and the node's own cluster authenticates with the fleet key
#               derived from cluster_secret, so clustering still works.
#               Off: the node answers anyone who can reach the port, which is the
#               pre-0.5.30 behaviour and a deliberate choice on a trusted LAN.
#               An install over an existing $AINODE_HOME/config.json changes
#               neither the auth state nor the keys, whatever this is set to.

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
# "<host>[:<port>]:<token>" -- join this node to a cluster after installing it.
AINODE_JOIN="${AINODE_JOIN:-}"
# The cluster's shared discovery secret, identical on every node (see the header).
AINODE_CLUSTER_SECRET="${AINODE_CLUSTER_SECRET:-}"
# "on" (default) or "off": whether a FRESH install requires an API key.
AINODE_AUTH="${AINODE_AUTH:-on}"
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

case "$AINODE_AUTH" in
    on|off) ;;
    *) echo "AINODE_AUTH must be on or off; got \"$AINODE_AUTH\"" >&2; exit 2 ;;
esac

if [ -n "$AINODE_JOIN" ] && [ "$AINODE_JOB" = "master" ]; then
    echo "AINODE_JOIN joins an existing cluster; --job master heads its own." >&2
    echo "Pick one." >&2
    exit 2
fi

# N random bytes as lowercase hex. openssl when it is there, /dev/urandom
# otherwise, because openssl is not on every minimal image. One home for the
# idiom: the cluster secret and the installer's API key both come from here.
random_hex() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex "$1"
    else
        head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}

# SHA-256 of $1 as lowercase hex, on stdout. Non-zero when this host has no
# hasher, which is the one case where the installer leaves auth off rather than
# writing a key it cannot store hashed: auth.json holds hashes, never a key.
sha256_hex() {
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$1" | sha256sum | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        printf '%s' "$1" | openssl dgst -sha256 | awk '{print $NF}'
    else
        return 1
    fi
}

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

# Is this a first install on this box, or an install over one that exists?
# The ONLY signal is config.json: everything below that changes a node's identity
# or its access control is gated on this being a fresh install, because an update
# must never turn auth on under a running fleet or re-key a node whose operator
# already holds a key.
FRESH_INSTALL="false"
[ -f "$AINODE_HOME/config.json" ] || FRESH_INSTALL="true"

# Write initial config.json if not already present.
# No model is set — the user picks one via the web UI after install.
# Job role determines whether this node runs an engine (master/solo)
# or just announces itself and waits for work (worker).
if [ "$FRESH_INSTALL" = "true" ]; then
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

    # A shared secret for this cluster, generated here so a fresh node HAS one
    # rather than running unauthenticated discovery until somebody notices. It is
    # only ever identical across a cluster by being handed over: `ainode cluster
    # token` on this node prints the join command, and `ainode join` writes THIS
    # value on the joining node. A node that is joining somebody else's cluster
    # (AINODE_JOIN below) has its value overwritten by the join a moment later.
    #
    # Generated with openssl when it is there, /dev/urandom otherwise, because
    # openssl is not on every minimal image.
    if [ -n "$AINODE_CLUSTER_SECRET" ]; then
        CLUSTER_SECRET="$AINODE_CLUSTER_SECRET"
        log "Cluster secret: taken from AINODE_CLUSTER_SECRET"
    else
        CLUSTER_SECRET="$(random_hex 32)"
    fi
    if [ -z "$AINODE_CLUSTER_SECRET" ] && [ -z "$AINODE_JOIN" ]; then
        log "Cluster secret: generated for this node"
        log "  A SECOND node has to end up with this same value to see this one."
        log "  Join it (ainode cluster token here, then the ainode join line there),"
        log "  or install it with AINODE_CLUSTER_SECRET set to this node's value."
    fi
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
  "cluster_secret": "${CLUSTER_SECRET}",
  "cluster_interface": "${CLUSTER_IFACE}",
  "ssh_user": "${AINODE_SSH_USER}",
  "api_port": 8000,
  "web_port": 3000,
  "discovery_port": 5679,
  "gpu_memory_utilization": 0.6
}
CONFIG
    # 0600: this file carries cluster_secret, which signs discovery and derives
    # the key every node-to-node call presents. NodeConfig.save() keeps it that
    # way; the file is born here.
    chmod 600 "$AINODE_HOME/config.json"
    log "Node configured as: $AINODE_JOB (distributed_mode=$DIST_MODE)"
fi

# -- 2d. An API key, on a fresh install -------------------------------------
# AINode serves the dashboard, the OpenAI-compatible API and every management
# route (load a model, unload one, PATCH the config) on the same port, so the key
# IS the access control. Shipping open meant a node on a LAN answered anyone who
# could reach port 3000, and the fleet ran open because auth used to break the
# product: the UI could not send a key (#167) and a node with auth on could not
# talk to its own peers. Both are fixed, so the default flips.
#
# Gated on FRESH_INSTALL, and separately on auth.json not existing: an update runs
# this same script, and an update that turned auth on would lock out every client
# the operator has already pointed at this node, with a key they never saw.
AUTH_JSON="$AINODE_HOME/auth.json"
INSTALL_API_KEY=""
INSTALL_API_KEY_ID=""

# Whether the node ALREADY requires a key, for the summary line on an install
# over an existing home. Read with grep rather than a JSON parser because this
# script cannot assume python, and the file is written by AuthConfig.save().
auth_enabled_on_disk() {
    [ -f "$AUTH_JSON" ] || return 1
    grep -q '"enabled"[[:space:]]*:[[:space:]]*true' "$AUTH_JSON"
}

if [ "$FRESH_INSTALL" != "true" ] || [ -f "$AUTH_JSON" ]; then
    log "API access: left exactly as it is (this is not a fresh install)"
    log "  Change it with: ainode auth enable | ainode auth disable"
elif [ "$AINODE_AUTH" = "off" ]; then
    log "API access: OPEN, because you asked for it with AINODE_AUTH=off"
    log "  Anyone who can reach port 3000 on this host can load and unload"
    log "  models, read the config and change it. That is a reasonable choice"
    log "  on a LAN you trust and a bad one on anything routable."
    log "  Require a key later with: ainode auth enable"
elif ! INSTALL_API_KEY_HASH="$(sha256_hex probe)"; then
    warn "API access: OPEN. No sha256sum, shasum or openssl on this host, so the"
    warn "  installer cannot store a key hashed, and it will not write one in the"
    warn "  clear. Install coreutils or openssl, then: ainode auth enable"
else
    INSTALL_API_KEY="$(random_hex 16)"
    INSTALL_API_KEY_ID="$(random_hex 4)"
    INSTALL_API_KEY_HASH="$(sha256_hex "$INSTALL_API_KEY")"
    # Same shape AuthConfig reads and writes (ainode/auth/middleware.py): the
    # HASH is stored and the key itself exists only in the box printed at the end.
    cat > "$AUTH_JSON" << AUTHJSON
{
  "enabled": true,
  "api_keys": [
    {
      "id": "${INSTALL_API_KEY_ID}",
      "key_hash": "${INSTALL_API_KEY_HASH}",
      "name": "installer",
      "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    }
  ]
}
AUTHJSON
    chmod 600 "$AUTH_JSON"
    log "API access: PROTECTED. One key minted, stored hashed in $AUTH_JSON"
    log "  The key is printed once at the end of this install. Copy it then."
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

# The version /api/health reports. Empty output plus non-zero when the node does
# not answer at all. "version" is the only key with that exact name in the
# payload (driver_version and friends do not match the leading quote).
#
# /api/health and NOT /api/status, because health is the one route that answers
# without an API key (ainode/auth/middleware.py::SKIP_PATHS) and a fresh install
# requires one: reading status here made every update on a protected node pull,
# pin, restart, read nothing and then report that the update had not applied.
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

After the restart the wrapper waits for this node's /api/health to report the
version it just installed, and exits non-zero if it does not: an update that did
not apply must not report success. Only then are the images it replaced removed,
so a failed update still has something to fall back to. /api/health is the route
that answers with no API key, so this works on a node that requires one.

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

# Forward a command into the running container, falling back to a one-shot
# docker run when it is not up so \`ainode --help\`, \`ainode service install\`
# and friends still work. Both the catch-all below and the \`tls\` case use it.
#
# AINODE_TAILNET_NAME rides along, set or not: the container has no tailscale
# binary and no path to the tailnet daemon, so this variable is the cheapest way
# for the CLI in there to know which tailnet node it is running on.
forward_to_container() {
    # A TTY is allocated only when there is one to pass on. \`docker exec -t\` with
    # a pipe or an \`ssh host ainode ...\` command line dies with "the input device
    # is not a TTY" before the CLI in the container runs at all, so a hardcoded
    # -it is what would stop
    # \`echo pw | ainode auth user add x --password-stdin\` working and what
    # \`ainode doctor --peer\` already has to route around (cli/doctor.py).
    local exec_tty="-i"
    if [ -t 0 ] && [ -t 1 ]; then exec_tty="-it"; fi
    if docker exec ainode true 2>/dev/null; then
        exec docker exec \$exec_tty -e AINODE_TAILNET_NAME -e AINODE_HOST_SERVICE_STATE ainode ainode "\$@"
    fi
    # Same sudo trap as update: mount the .ainode the unit uses.
    local conf_home
    conf_home="\$(resolve_ainode_home || true)"
    [ -n "\$conf_home" ] || conf_home="\$HOME/.ainode"
    exec docker run --rm \$exec_tty \\
        --entrypoint ainode \\
        -e AINODE_TAILNET_NAME \\
        -e AINODE_HOST_SERVICE_STATE \\
        -v "\$conf_home":/root/.ainode \\
        "\$AINODE_IMAGE" "\$@"
}

# This node's MagicDNS name without the trailing dot, or empty.
#
# --peers=false keeps the answer to this node alone; a tailscale old enough to
# reject the flag exits non-zero and the plain form runs, where Self is the first
# DNSName in the document anyway.
tailnet_name() {
    command -v tailscale >/dev/null 2>&1 || return 0
    { tailscale status --peers=false --json 2>/dev/null \\
        || tailscale status --json 2>/dev/null; } \\
        | sed -n 's/.*"DNSName"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p' \\
        | head -1 | sed 's/\\.\$//'
}

# \`tailscale cert\` as whoever is allowed to run it. The plain call works when
# \`sudo tailscale set --operator=\$USER\` was run once; otherwise sudo -n, which
# never prompts, because this also runs from a timer with no terminal attached.
tailscale_cert_into() {
    local cert="\$1" key="\$2" name="\$3" out=""
    if out=\$(tailscale cert --cert-file "\$cert" --key-file "\$key" "\$name" 2>&1); then
        printf '%s\\n' "\$out"
        return 0
    fi
    if out=\$(sudo -n tailscale cert --cert-file "\$cert" --key-file "\$key" "\$name" 2>&1); then
        printf '%s\\n' "\$out"
        return 0
    fi
    printf '%s\\n' "\$out" >&2
    return 1
}

# The notAfter date in a certificate file, or empty when it cannot be read.
# Tells "tailscale issued a new one" from "tailscale handed back the cached one",
# which is what decides whether a restart is needed at all.
cert_not_after() {
    [ -r "\$1" ] || return 0
    command -v openssl >/dev/null 2>&1 || return 0
    openssl x509 -in "\$1" -noout -enddate 2>/dev/null \\
        | sed -n 's/^notAfter=//p' | head -1
}

# Make the pair readable by the server and by nobody else. It is root-owned
# after sudo, so hand it to whoever owns the .ainode the service reads: a later
# renewal and \`ainode tls status\` are then not root-only operations.
own_tls_pair() {
    local home="\$1" cert="\$2" key="\$3"
    chmod 600 "\$key" 2>/dev/null || true
    chmod 644 "\$cert" 2>/dev/null || true
    if [ "\$(id -u)" = "0" ]; then
        chown --reference="\$home" "\$cert" "\$key" 2>/dev/null || true
    fi
}

print_tls_help() {
    cat <<TLSHELP
Usage: ainode tls {enable|disable|status|renew} [options]

Two of these run on the HOST, because tailscale does and the AINode container
ships no tailscale binary and has no path to the tailnet daemon:

  enable --tailscale   get a real Let's Encrypt certificate for this node's
                       MagicDNS name, write it into <AINODE_HOME>/tls/, then have
                       the container record it in config.json. Restart to serve it.
  renew                replace that certificate when it is inside its last 14
                       days, and restart the node so the new one is served.
                       ainode-tls-renew.timer runs this daily.

Everything else (enable with a pair or self-signed, disable, status) is forwarded
into the container unchanged.

Environment:
  AINODE_HOME                  .ainode the unit reads (normally detected)
  AINODE_TLS_RENEW_RESTART=0   renew the pair but leave the restart to a human
TLSHELP
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
        STATUS_URL="http://127.0.0.1:\$(node_web_port "\$AINODE_HOME")/api/health"
        if [ -z "\$TARGET_VERSION" ]; then
            # Non-zero on purpose: \$AINODE_IMAGE (a floating :latest) was pulled and
            # the service restarted, but with no version resolved nothing can say
            # whether the node came back on anything new. "exit 0" is what a roll
            # script reads as "this node is updated", and this is not that.
            echo "XX Could not resolve a version from ghcr.io, so nothing was verified." >&2
            echo "   \$PULL_IMAGE was pulled and \$AINODE_SERVICE restarted." >&2
            echo "   /api/status reports: \$(api_version "\$STATUS_URL" || echo 'no answer')" >&2
            echo "   Images were left alone. Name the version to be sure:" >&2
            echo "     ainode update <version>" >&2
            exit 1
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
  tls enable --tailscale  get a real certificate for this node's MagicDNS name
                          (tailscale runs on the host, not in the container), then
                          record it in config.json. Restart to serve it.
  tls renew               replace that certificate inside its last 14 days and
                          restart so the new one is served. Run daily by
                          ainode-tls-renew.timer. 'ainode tls --help' for the rest.
  --version               print the wrapper's pinned image tag
  doctor [args...]        the container's own report, plus the host unit state
                          handed in so the systemd check is real and not a WARN
                          about there being no systemd in a container

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
    tls)
        # Two shapes under \`tls\` are host work, for one reason: tailscale runs on
        # the host and is not in the image. Getting a tailnet certificate and
        # renewing one happen here; the container is still what writes the config
        # block, because it is what knows its own AINODE_HOME. Every other \`tls\`
        # subcommand is forwarded untouched.
        TLS_SUB="\${2:-}"
        TLS_TAILSCALE=false
        for tls_arg in "\$@"; do
            [ "\$tls_arg" = "--tailscale" ] && TLS_TAILSCALE=true
            case "\$tls_arg" in -h|--help) print_tls_help; exit 0 ;; esac
        done

        if [ "\$TLS_SUB" = "enable" ] && [ "\$TLS_TAILSCALE" = "true" ]; then
            if ! command -v tailscale >/dev/null 2>&1; then
                echo "XX No tailscale on this host, so there is no tailnet name to" >&2
                echo "   get a certificate for." >&2
                echo "   Join the tailnet, or make a self-signed pair instead:" >&2
                echo "     ainode tls enable" >&2
                exit 1
            fi
            TLS_NAME="\$(tailnet_name)"
            if [ -z "\$TLS_NAME" ]; then
                echo "XX Could not read this node's MagicDNS name from tailscale." >&2
                echo "   Is tailscaled up? Check: tailscale status" >&2
                exit 1
            fi
            # WHERE before WHAT, the same rule \`update\` follows: a pair written
            # into the wrong .ainode is a node with TLS enabled and no certificate.
            if ! AINODE_HOME="\$(resolve_ainode_home)"; then
                echo "XX Cannot tell which .ainode this service reads, so the pair" >&2
                echo "   would land where the server does not look. Name it:" >&2
                echo "     sudo AINODE_HOME=/home/<user>/.ainode ainode tls enable --tailscale" >&2
                exit 1
            fi
            mkdir -p "\$AINODE_HOME/tls"
            chmod 700 "\$AINODE_HOME/tls" 2>/dev/null || true
            # Named after the certificate's one name: it is the only thing that
            # carries WHICH name this pair is for across the container boundary,
            # and it is how the CLI in there finds the pair with no arguments.
            TLS_CERT="\$AINODE_HOME/tls/\$TLS_NAME.crt"
            TLS_KEY="\$AINODE_HOME/tls/\$TLS_NAME.key"
            echo "==> tailscale cert for \$TLS_NAME"
            if ! tailscale_cert_into "\$TLS_CERT" "\$TLS_KEY" "\$TLS_NAME"; then
                echo "XX tailscale could not issue a certificate for \$TLS_NAME." >&2
                echo "   If it said HTTPS is not enabled: Tailscale admin console >" >&2
                echo "   DNS > HTTPS Certificates > Enable." >&2
                echo "   If it said cert access denied, run once on this host:" >&2
                echo "     sudo tailscale set --operator=\$USER" >&2
                echo "   A self-signed pair works meanwhile: ainode tls enable" >&2
                exit 1
            fi
            own_tls_pair "\$AINODE_HOME" "\$TLS_CERT" "\$TLS_KEY"
            echo "==> Wrote \$TLS_CERT"
            echo "==> Recording it in config.json"
            export AINODE_TAILNET_NAME="\$TLS_NAME"
            forward_to_container "\$@"
        fi

        if [ "\$TLS_SUB" = "renew" ]; then
            if ! AINODE_HOME="\$(resolve_ainode_home)"; then
                echo "XX Cannot tell which .ainode this service reads." >&2
                exit 1
            fi
            # The DECISION is the container's: it can read the certificate and it
            # owns the 14 day rule, in ainode/tls/renew.py. The ACTING is the
            # host's. --check prints key=value lines and exits 10 for "nothing to
            # do", so a daily timer says nothing for 75 days.
            TLS_DECISION="\$(docker exec ainode ainode tls renew --check 2>/dev/null)" || true
            TLS_RENEW=\$(printf '%s\n' "\$TLS_DECISION" | sed -n 's/^renew=//p' | head -1)
            TLS_NAME=\$(printf '%s\n' "\$TLS_DECISION" | sed -n 's/^name=//p' | head -1)
            TLS_CERT_NAME=\$(printf '%s\n' "\$TLS_DECISION" | sed -n 's/^cert_name=//p' | head -1)
            TLS_KEY_NAME=\$(printf '%s\n' "\$TLS_DECISION" | sed -n 's/^key_name=//p' | head -1)
            TLS_REASON=\$(printf '%s\n' "\$TLS_DECISION" | sed -n 's/^reason=//p' | head -1)
            if [ -z "\$TLS_RENEW" ]; then
                # A node that is down is not a renewal that failed: exit 0, so a
                # daily timer does not mark itself failed all through an outage.
                echo "!! Could not ask the container about the certificate."
                echo "   Is ainode running? Nothing was changed."
                exit 0
            fi
            if [ "\$TLS_RENEW" != "yes" ]; then
                echo "==> No renewal needed: \$TLS_REASON"
                exit 0
            fi
            if [ -z "\$TLS_NAME" ]; then
                echo "XX The node says a renewal is due but names no tailnet name:" >&2
                echo "   \$TLS_REASON" >&2
                exit 1
            fi
            if ! command -v tailscale >/dev/null 2>&1; then
                echo "XX \$TLS_REASON" >&2
                echo "   There is no tailscale on this host to renew it with." >&2
                exit 1
            fi
            # The directory can be gone: the decision above renews an enabled
            # block whose pair vanished, and that is one of the ways it vanishes.
            mkdir -p "\$AINODE_HOME/tls"
            chmod 700 "\$AINODE_HOME/tls" 2>/dev/null || true
            # The files the CONFIG points at, not name-derived ones: a pair from
            # an earlier manual run can be sitting at cert.pem, and renewing into
            # a new filename would leave the server reading the expiring one.
            [ -n "\$TLS_CERT_NAME" ] || TLS_CERT_NAME="\$TLS_NAME.crt"
            [ -n "\$TLS_KEY_NAME" ] || TLS_KEY_NAME="\$TLS_NAME.key"
            TLS_CERT="\$AINODE_HOME/tls/\$TLS_CERT_NAME"
            TLS_KEY="\$AINODE_HOME/tls/\$TLS_KEY_NAME"
            TLS_BEFORE="\$(cert_not_after "\$TLS_CERT")"
            echo "==> \$TLS_REASON"
            echo "==> tailscale cert for \$TLS_NAME"
            if ! tailscale_cert_into "\$TLS_CERT" "\$TLS_KEY" "\$TLS_NAME"; then
                echo "XX tailscale could not renew the certificate for \$TLS_NAME." >&2
                echo "   The old pair is still in place and still being served." >&2
                echo "   Check: Tailscale admin console > DNS > HTTPS Certificates," >&2
                echo "   and 'sudo tailscale set --operator=\$USER' for cert access." >&2
                exit 1
            fi
            own_tls_pair "\$AINODE_HOME" "\$TLS_CERT" "\$TLS_KEY"
            TLS_AFTER="\$(cert_not_after "\$TLS_CERT")"
            if [ -n "\$TLS_BEFORE" ] && [ "\$TLS_AFTER" = "\$TLS_BEFORE" ]; then
                echo "==> tailscale handed back the same certificate (\$TLS_AFTER)."
                echo "    Nothing to restart; the next run tries again."
                exit 0
            fi
            echo "==> New certificate valid until \$TLS_AFTER"
            # The listener and its SSLContext are built at boot, so this restart
            # is what makes the renewal real. Loaded models run in their own
            # containers and are not touched by it.
            if [ "\${AINODE_TLS_RENEW_RESTART:-1}" = "0" ]; then
                echo "!! Restart skipped (AINODE_TLS_RENEW_RESTART=0). This node goes on"
                echo "   serving the OLD certificate until: sudo systemctl restart ainode"
                exit 0
            fi
            echo "==> Restarting \$AINODE_SERVICE so the new pair is served"
            if is_user_mode; then
                systemctl --user try-restart "\$AINODE_SERVICE"
            elif [ "\$(id -u)" = "0" ]; then
                systemctl try-restart "\$AINODE_SERVICE"
            else
                sudo -n systemctl try-restart "\$AINODE_SERVICE"
            fi
            echo "==> Renewed. Loaded models run in their own containers and were"
            echo "    not touched."
            exit 0
        fi

        export AINODE_TAILNET_NAME="\$(tailnet_name)"
        forward_to_container "\$@"
        ;;
    doctor)
        # The doctor runs in the container like everything else, but three of its
        # checks are about the HOST: the systemd unit, the docker daemon and which
        # image the container runs. It cannot see the unit from in there, which is
        # why every containerized node carried a permanent WARN nobody in there
        # could clear (#225). So read the unit state here and hand it in through
        # the environment, the way \`tls\` hands the tailnet name in.
        AINODE_HOST_SERVICE_STATE="\$(systemctl is-active "\$AINODE_SERVICE" 2>/dev/null || true)"
        HOST_STATE="\$AINODE_HOST_SERVICE_STATE"
        if [ -z "\$HOST_STATE" ] || [ "\$HOST_STATE" = "inactive" ]; then
            USER_STATE="\$(systemctl --user is-active "\$AINODE_SERVICE" 2>/dev/null || true)"
            [ -n "\$USER_STATE" ] && AINODE_HOST_SERVICE_STATE="\$USER_STATE"
        fi
        export AINODE_HOST_SERVICE_STATE
        forward_to_container "\$@"
        ;;
    *)
        # Forward everything else into the running container.
        forward_to_container "\$@"
        ;;
esac
WRAPPER

$WRAPPER_SUDO chmod +x "$WRAPPER_PATH"

# -- 5b. Renew the tailnet certificate without a human ----------------------
# A `tailscale cert` pair is Let's Encrypt, so it lives 90 days, and something
# has to notice day 76 without being asked. It cannot be the server: that runs
# inside the container, which ships no tailscale binary and has no path to the
# tailnet daemon, and the TLS listener is built at boot so serving a new pair is
# a restart. It cannot be `ainode doctor` either, which warns correctly but only
# when a human runs it. So it is a timer on the HOST, and it runs the wrapper:
# the container decides (ainode/tls/renew.py owns the 14 day rule), the host acts.
#
# Installed whatever this node's TLS state is. A node with TLS off costs one
# `docker exec` a day that prints "nothing to renew", and a node that turns TLS
# on later is already covered rather than needing a reinstall.
TLS_RENEW_STAGING="/tmp/ainode-tls-renew"
[ "$DRY_RUN" = "true" ] && TLS_RENEW_STAGING="$AINODE_HOME/ainode-tls-renew"

cat > "$TLS_RENEW_STAGING.service" << TLSRENEWUNIT
[Unit]
Description=Renew AINode's tailnet TLS certificate when it is nearly expired
Documentation=https://ainode.dev
After=docker.service
Wants=docker.service

[Service]
Type=oneshot
Environment=AINODE_HOME=${AINODE_HOME}
ExecStart=${WRAPPER_PATH} tls renew
# One run must never hold the timer open: tailscale cert is a network call and
# the restart afterwards waits on the node coming back.
TimeoutStartSec=600
TLSRENEWUNIT

cat > "$TLS_RENEW_STAGING.timer" << TLSRENEWTIMER
[Unit]
Description=Daily check on AINode's tailnet TLS certificate
Documentation=https://ainode.dev

[Timer]
OnCalendar=daily
# Spread the fleet out: a cluster installed in one sitting would otherwise ask
# Let's Encrypt for every node's certificate in the same second.
RandomizedDelaySec=4h
# A node that was off when the timer was due still runs it once it is back,
# which is the case that matters: a machine asleep past its renewal window.
Persistent=true

[Install]
WantedBy=timers.target
TLSRENEWTIMER

if [ "$DRY_RUN" = "true" ]; then
    log "Dry run: renewal timer rendered at $TLS_RENEW_STAGING.{service,timer}"
elif [ "$USER_MODE" = "true" ]; then
    mv "$TLS_RENEW_STAGING.service" "$UNIT_DIR/ainode-tls-renew.service"
    mv "$TLS_RENEW_STAGING.timer" "$UNIT_DIR/ainode-tls-renew.timer"
    systemctl --user daemon-reload
    systemctl --user enable --now ainode-tls-renew.timer
    log "Installed ainode-tls-renew.timer (user scope, daily)"
else
    sudo mv "$TLS_RENEW_STAGING.service" "$UNIT_DIR/ainode-tls-renew.service"
    sudo mv "$TLS_RENEW_STAGING.timer" "$UNIT_DIR/ainode-tls-renew.timer"
    sudo systemctl daemon-reload
    sudo systemctl enable --now ainode-tls-renew.timer
    log "Installed ainode-tls-renew.timer (daily)"
fi

# -- 6. Join an existing cluster --------------------------------------------
# AINODE_JOIN is "<host>[:<port>]:<token>". The token is the last colon-separated
# field, so a host with a port still parses: 10.0.0.1:3000:abc -> 10.0.0.1:3000
# and abc. `ainode join` inside the container does the work (it writes the six
# config keys and nothing else) and the restart below applies them; the wrapper
# is already installed above, so this is the same command a user would type.
#
# Deliberately AFTER the service is up: the join is an HTTP call to the master
# and a config write, and a node that fails to join is still a working node.
if [ -n "$AINODE_JOIN" ]; then
    JOIN_TOKEN="${AINODE_JOIN##*:}"
    JOIN_HOST="${AINODE_JOIN%:*}"
    if [ -z "$JOIN_TOKEN" ] || [ -z "$JOIN_HOST" ] || [ "$JOIN_HOST" = "$AINODE_JOIN" ]; then
        warn "AINODE_JOIN must be \"<host>[:<port>]:<token>\"; got \"$AINODE_JOIN\""
        warn "  Skipping the join. Mint a token on the master (ainode cluster token)"
        warn "  and run: ainode join <host>:3000 <token>"
    elif [ "$DRY_RUN" = "true" ]; then
        log "Dry run: would join the cluster at $JOIN_HOST (token not used)"
    else
        log "Joining the cluster at $JOIN_HOST"
        if "$WRAPPER_PATH" join "$JOIN_HOST" "$JOIN_TOKEN" --name "$(hostname)"; then
            log "  Joined. Restarting to apply."
            if [ "$USER_MODE" = "true" ]; then
                systemctl --user restart ainode.service || \
                    warn "  restart failed; run: systemctl --user restart ainode"
            else
                sudo systemctl restart ainode.service || \
                    warn "  restart failed; run: sudo systemctl restart ainode"
            fi
        else
            warn "  The join failed. This node is installed and running standalone."
            warn "  Mint a fresh token on the master (ainode cluster token) and run:"
            warn "    ainode join $JOIN_HOST <token>"
        fi
    fi
fi

# The one thing an operator must not scroll past. Printed by both the dry run and
# a real install, and only when THIS run minted a key: the plaintext exists
# nowhere else, and `ainode auth key create` is the only way to get another.
print_api_key_box() {
    [ -n "$INSTALL_API_KEY" ] || return 0
    printf '\n'
    printf '    \033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
    printf '    \033[1;33m  YOUR API KEY. SHOWN ONCE, STORED HASHED. COPY IT NOW.\033[0m\n'
    printf '    \033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
    printf '\n'
    printf '      \033[1;37m%s\033[0m\n' "$INSTALL_API_KEY"
    printf '\n'
    printf '      Key id:     %s  (name: installer)\n' "$INSTALL_API_KEY_ID"
    printf '      Dashboard:  http://localhost:3000 asks for it on first open\n'
    printf '                  (Config > API access), then remembers it.\n'
    printf '      curl:       -H "Authorization: Bearer %s"\n' "$INSTALL_API_KEY"
    printf '      Another:    ainode auth key create --name <client>\n'
    printf '      No key:     ainode auth disable   (the API answers everyone)\n'
    printf '    \033[1;33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
}

# One sentence about who may call this node, decided once and printed by both the
# dry run and the real banner. The vocabulary is the dashboard's and
# /api/status's for the same state (ainode/api/server.py::auth_status_fields).
if [ -n "$INSTALL_API_KEY" ]; then
    ACCESS_LINE="API protected, one key. It is printed below, once."
elif auth_enabled_on_disk; then
    ACCESS_LINE="API key required. Unchanged by this install."
else
    ACCESS_LINE="API open, no key set. Require one in Config > API access."
fi

# The two commands a human needs to be able to open the dashboard with a name and
# a password instead of pasting an API key (#261). Printed only when this node
# requires a credential, because with auth off the dashboard opens without a login
# and these lines would be advice about a problem nobody has. There is no route
# that mints the first admin: the first account is made on the box, by the operator.
print_login_lines() {
    [ -n "$INSTALL_API_KEY" ] || auth_enabled_on_disk || return 0
    printf '    Login:   ainode auth enable\n'
    printf '             ainode auth user add <name> --admin\n'
    printf '             [the account is how a human signs in instead of pasting\n'
    printf '              the key; --password-stdin where there is no terminal]\n'
}

# -- Banner -----------------------------------------------------------------
if [ "$DRY_RUN" = "true" ]; then
    printf '\n'
    log "Dry run complete. Rendered under $AINODE_HOME:"
    log "  config.json      the node config an install would write"
    log "                   (including a generated cluster_secret)"
    if [ -n "$INSTALL_API_KEY" ]; then
        log "  auth.json        auth ON with one key, stored as a SHA-256 hash"
    fi
    log "  ainode.service   the systemd unit (not installed)"
    log "  ainode-wrapper   the /usr/local/bin/ainode host wrapper"
    log "  ainode-tls-renew.{service,timer}"
    log "                   the daily tailnet certificate renewal (not installed)"
    log "  Access:          $ACCESS_LINE"
    print_api_key_box
    print_login_lines
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
printf '    Access:  %s\n' "$ACCESS_LINE"
printf '    Status:  ainode status\n'
printf '    Logs:    ainode logs -f\n'
printf '    Update:  ainode update\n'
print_api_key_box
print_login_lines
printf '\n'
printf '    Made in Texas\n'
printf '\n'
