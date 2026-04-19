#!/usr/bin/env bash
# AINode installer — v0.4.1 container-native.
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

set -euo pipefail

# -- Defaults ---------------------------------------------------------------
AINODE_VERSION="${AINODE_VERSION:-0.4.1}"
AINODE_IMAGE="${AINODE_IMAGE:-ghcr.io/getainode/ainode:latest}"
# NVIDIA official vLLM engine image — pre-pulled so first dashboard launch
# doesn't hit a 5–10 min download. Override with AINODE_NVIDIA_IMAGE=skip to
# suppress, or a custom tag for testing.
AINODE_NVIDIA_IMAGE="${AINODE_NVIDIA_IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
AINODE_PEERS="${AINODE_PEERS:-}"           # comma-separated IPs
AINODE_SSH_USER="${AINODE_SSH_USER:-$USER}"
AINODE_JOB="${AINODE_JOB:-solo}"          # solo | master | worker
SETUP_SSH="false"
USER_MODE="false"

# NGC / HF token locations (read-only hints; we never write these).
NGC_TOKEN_FILE="${NGC_TOKEN_FILE:-/etc/ainode/ngc.token}"
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
        -h|--help)
            sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

log() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$*"; }
die() { printf "\033[1;31mXX\033[0m %s\n" "$*" >&2; exit 1; }

# -- 1. Preflight -----------------------------------------------------------
log "Checking prerequisites"
[ "$(uname -s)" = "Linux" ] || die "AINode requires Linux (detected $(uname -s))"
command -v docker >/dev/null 2>&1 || die "docker not found. Install: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die "docker daemon not reachable — run 'sudo systemctl start docker' or add \$USER to the 'docker' group"

# /mnt/shared-models is required by the v0.4.9 systemd unit (--mount type=bind
# fails loudly if the source doesn't exist). Surface the setup requirement
# here instead of waiting for first-start to fail with a cryptic docker error.
# For clusters: make this an NFS mount from the master's model storage. For
# single-node: a directory is enough. See CHANGELOG v0.4.9 for context.
# TODO(v0.4.10): once the NCCL init shim is baked into ainode-base, this
# path becomes optional and the precheck can be dropped.
if [ ! -d /mnt/shared-models ]; then
    die "AINode v0.4.9+ requires /mnt/shared-models to exist for the per-node NCCL init shim.\n  Create it before re-running this installer:\n    sudo mkdir -p /mnt/shared-models\n  For clusters, mount shared model storage there (NFS from master recommended)."
fi

# GPU check (nvidia-container-toolkit). AINode targets NVIDIA GB10; skip if
# missing, let the container fail fast with a clear error.
if ! docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "Could not run a GPU container. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    warn "Continuing — AINode will error at start time if the GPU isn't accessible."
fi

# -- 2. Create config dir, pull image, write initial config ----------------
log "Preparing $AINODE_HOME"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

log "Pulling $AINODE_IMAGE (AINode orchestrator; slim — ~500 MB)"
docker pull "$AINODE_IMAGE"

# -- 2a. NGC login (for nvcr.io/nvidia/vllm image pull) -------------------
# The NVIDIA engine backend requires a pull from NGC. Non-interactive login
# prefers a token file or env var; otherwise prints a clear next-step.
nvidia_image_pulled() {
    docker image inspect "$AINODE_NVIDIA_IMAGE" >/dev/null 2>&1
}

ngc_login_noninteractive() {
    local key=""
    if [ -n "${NGC_API_KEY:-}" ]; then
        key="$NGC_API_KEY"
    elif [ -r "$NGC_TOKEN_FILE" ]; then
        key="$(tr -d '[:space:]' < "$NGC_TOKEN_FILE")"
    fi
    if [ -z "$key" ]; then
        return 1
    fi
    # NGC uses a literal '$oauthtoken' as the username (single-quoted on
    # purpose — it is NOT a shell variable). See
    # https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/.
    echo "$key" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null 2>&1
}

if [ "$AINODE_NVIDIA_IMAGE" = "skip" ]; then
    warn "Skipping NVIDIA vLLM image pre-pull (AINODE_NVIDIA_IMAGE=skip)"
elif nvidia_image_pulled; then
    log "NVIDIA vLLM image already present: $AINODE_NVIDIA_IMAGE"
else
    log "Pulling NVIDIA vLLM engine image: $AINODE_NVIDIA_IMAGE"
    log "  (~15 GB; takes 5–10 minutes on a 1 Gbps link. One-time operation.)"
    if ngc_login_noninteractive; then
        log "  NGC login: OK (from NGC_API_KEY or $NGC_TOKEN_FILE)"
    else
        warn "NGC credentials not found in \$NGC_API_KEY or $NGC_TOKEN_FILE."
        warn "  Pre-pull skipped. The NVIDIA engine backend will fail until you"
        warn "  run:"
        warn "     docker login nvcr.io"
        warn "  (username: \$oauthtoken   password: your NGC API key from https://ngc.nvidia.com)"
    fi
    if docker pull "$AINODE_NVIDIA_IMAGE"; then
        log "  NVIDIA vLLM image pulled: $AINODE_NVIDIA_IMAGE"
    else
        warn "  docker pull $AINODE_NVIDIA_IMAGE failed (likely 401 / unauthenticated)."
        warn "  Fix:  docker login nvcr.io   (see https://ngc.nvidia.com for an API key)"
        warn "  Continuing — AINode will still install; the NVIDIA engine will error until the pull succeeds."
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

    cat > "$AINODE_HOME/config.json" << CONFIG
{
  "node_name": "$(hostname)",
  "onboarded": true,
  "distributed_mode": "${DIST_MODE}",
  "peer_ips": [${PEER_IPS}],
  "cluster_id": "ainode-cluster",
  "cluster_interface": "enP2p1s0f1np1",
  "ssh_user": "${AINODE_SSH_USER}",
  "api_port": 8000,
  "web_port": 3000,
  "discovery_port": 5679,
  "gpu_memory_utilization": 0.9
}
CONFIG
    log "Node configured as: $AINODE_JOB (distributed_mode=$DIST_MODE)"
fi

# -- 3. Optional passwordless SSH bootstrap ---------------------------------
# Needed only for distributed (multi-node) mode: the head runs eugr's
# launcher which SSHes into each peer and `docker run`s a worker container.
if [ "$SETUP_SSH" = "true" ] || [ -n "$AINODE_PEERS" ]; then
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

mkdir -p "$UNIT_DIR"

EXEC_START="/usr/bin/docker run --rm --name ainode \
 --network=host --gpus all --ipc=host --shm-size=64g \
 -v ${AINODE_HOME}:/root/.ainode \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v ${HOME}/.ssh:/host-ssh:ro \
 --mount type=bind,source=/mnt/shared-models,target=/mnt/shared-models,bind-propagation=rshared \
 -e AINODE_HOME=/root/.ainode \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
 ghcr.io/getainode/ainode:latest"

cat > /tmp/ainode.service << UNIT
[Unit]
Description=AINode — Local AI inference platform
Documentation=https://ainode.dev
After=network.target docker.service nvidia-persistenced.service
Wants=docker.service nvidia-persistenced.service
Requires=docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ainode
ExecStart=${EXEC_START}
ExecStop=/usr/bin/docker stop -t 30 ainode
Restart=on-failure
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=45
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=AINODE_HOME=${AINODE_HOME}

[Install]
WantedBy=${WANTED_BY}
UNIT

if [ "$USER_MODE" = "true" ]; then
    mv /tmp/ainode.service "$UNIT_DIR/ainode.service"
    systemctl --user daemon-reload
    systemctl --user enable --now ainode.service
    log "  (consider: sudo loginctl enable-linger $USER  — so the service survives logout)"
else
    sudo mv /tmp/ainode.service "$UNIT_DIR/ainode.service"
    sudo systemctl daemon-reload
    sudo systemctl enable --now ainode.service
fi

# -- 5. Install host-side `ainode` wrapper ----------------------------------
# The ainode CLI lives inside the container. This thin wrapper on the host
# dispatches `ainode update` to docker-pull + systemctl-restart (the only
# operations that must happen outside the container), and forwards every
# other command to `docker exec ainode ainode ...` so users never need to
# type the docker command themselves.
log "Installing /usr/local/bin/ainode host wrapper"
WRAPPER_PATH="/usr/local/bin/ainode"
WRAPPER_SUDO="sudo"
[ -w "$(dirname "$WRAPPER_PATH")" ] && WRAPPER_SUDO=""

$WRAPPER_SUDO tee "$WRAPPER_PATH" >/dev/null <<WRAPPER
#!/usr/bin/env bash
# AINode host wrapper — installed by install.sh. Not user-editable.
set -euo pipefail
AINODE_IMAGE="\${AINODE_IMAGE:-ghcr.io/getainode/ainode:latest}"
AINODE_SERVICE="ainode.service"

is_user_mode() {
    systemctl --user is-enabled "\$AINODE_SERVICE" >/dev/null 2>&1
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
        echo "==> Pulling \$AINODE_IMAGE"
        docker pull "\$AINODE_IMAGE"
        echo "==> Restarting \$AINODE_SERVICE"
        if is_user_mode || systemctl is-active --quiet "\$AINODE_SERVICE" 2>/dev/null; then
            restart_service
        else
            echo "   (service not running — start it with: sudo systemctl start ainode)"
        fi
        echo "==> Update complete. Version:"
        docker exec ainode ainode --version 2>/dev/null || \\
            docker run --rm --entrypoint ainode "\$AINODE_IMAGE" --version
        ;;
    "" | -h | --help)
        cat <<HELP
AINode host CLI. Commands that change the running container (update,
restart) run on the host; everything else is forwarded to the container.

Usage: ainode <command> [args...]

Host-side:
  update                  docker pull \$AINODE_IMAGE and restart service
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
            exec docker run --rm -it \\
                --entrypoint ainode \\
                -v "\$HOME/.ainode":/root/.ainode \\
                "\$AINODE_IMAGE" "\$@"
        fi
        ;;
esac
WRAPPER

$WRAPPER_SUDO chmod +x "$WRAPPER_PATH"

# -- Banner -----------------------------------------------------------------
printf '\n'
printf '    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '    \033[1;32m  AINode v%s installed!\033[0m\n' "${AINODE_VERSION}"
printf '    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\n'
printf '    Web:     http://localhost:3000\n'
printf '    API:     http://localhost:8000/v1\n'
printf '    Status:  ainode status\n'
printf '    Logs:    ainode logs -f\n'
printf '    Update:  ainode update\n'
printf '\n'
printf '    Powered by \033[0;34margentos.ai\033[0m\n'
printf '\n'
