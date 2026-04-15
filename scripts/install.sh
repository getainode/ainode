#!/usr/bin/env bash
# AINode installer — v0.4.0 container-native.
#
# Does four things:
#   1. sanity-check the host (Linux + docker + nvidia runtime)
#   2. docker pull ghcr.io/getainode/ainode:<version>
#   3. (optional, --setup-ssh or AINODE_PEERS set) passwordless SSH bootstrap
#      from this host to peer workers so distributed mode can launch them
#   4. write + enable the systemd unit that does `docker run ... ainode`
#
# Host Python venv + vLLM source build are gone — everything lives in the
# image. Upgrade is `docker pull` + `systemctl restart ainode`.
#
# Usage:
#   curl -fsSL https://ainode.dev/install | bash
#   AINODE_PEERS="10.0.0.2,10.0.0.3" curl -fsSL https://ainode.dev/install | bash
#   scripts/install.sh --setup-ssh           # explicit SSH bootstrap
#   scripts/install.sh --user                # install as --user systemd service

set -euo pipefail

# -- Defaults ---------------------------------------------------------------
AINODE_VERSION="${AINODE_VERSION:-0.4.0}"
AINODE_IMAGE="${AINODE_IMAGE:-ghcr.io/getainode/ainode:${AINODE_VERSION}}"
AINODE_HOME="${AINODE_HOME:-$HOME/.ainode}"
AINODE_PEERS="${AINODE_PEERS:-}"           # comma-separated IPs
AINODE_SSH_USER="${AINODE_SSH_USER:-$USER}"
SETUP_SSH="false"
USER_MODE="false"

# -- Arg parsing ------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --setup-ssh) SETUP_SSH="true" ;;
        --user)      USER_MODE="true" ;;
        -h|--help)
            sed -n '1,25p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$*"; }
die() { printf "\033[1;31mXX\033[0m %s\n" "$*" >&2; exit 1; }

# -- 1. Preflight -----------------------------------------------------------
log "Checking prerequisites"
[ "$(uname -s)" = "Linux" ] || die "AINode requires Linux (detected $(uname -s))"
command -v docker >/dev/null 2>&1 || die "docker not found. Install: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die "docker daemon not reachable — run 'sudo systemctl start docker' or add \$USER to the 'docker' group"

# GPU check (nvidia-container-toolkit). AINode targets NVIDIA GB10; skip if
# missing, let the container fail fast with a clear error.
if ! docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "Could not run a GPU container. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    warn "Continuing — AINode will error at start time if the GPU isn't accessible."
fi

# -- 2. Create config dir + pull image --------------------------------------
log "Preparing $AINODE_HOME"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

log "Pulling $AINODE_IMAGE (this is a one-time ~18 GB download)"
docker pull "$AINODE_IMAGE"

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

# Use the CLI baked into the image to write the unit file — the template
# lives in ainode/service/systemd.py and is version-locked with the image.
# IMPORTANT: override the entrypoint so we call `ainode` directly instead
# of going through docker-entrypoint.sh (which does `ainode start ...`).
docker run --rm \
    --entrypoint ainode \
    -v "$AINODE_HOME":/root/.ainode \
    -v "$HOME":"$HOME" \
    -e "AINODE_HOME=$AINODE_HOME" \
    -e "HOME=$HOME" \
    -e "AINODE_IN_CONTAINER=1" \
    "$AINODE_IMAGE" \
    service install "${SERVICE_ARGS[@]}"

if [ "$USER_MODE" = "true" ]; then
    systemctl --user daemon-reload
    systemctl --user enable --now ainode.service
    log "  (consider: sudo loginctl enable-linger $USER  — so the service survives logout)"
else
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
AINODE_IMAGE="\${AINODE_IMAGE:-$AINODE_IMAGE}"
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
            docker run --rm "\$AINODE_IMAGE" ainode --version
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
        docker exec ainode ainode --version 2>/dev/null || true
        ;;
    *)
        # Forward everything else into the running container. If the
        # container isn't up, fall back to a one-shot docker run so
        # \`ainode --help\`, \`ainode service install\`, etc. still work.
        if docker exec ainode true 2>/dev/null; then
            exec docker exec -it ainode ainode "\$@"
        else
            exec docker run --rm -it \\
                -v "\$HOME/.ainode":/root/.ainode \\
                "\$AINODE_IMAGE" ainode "\$@"
        fi
        ;;
esac
WRAPPER

$WRAPPER_SUDO chmod +x "$WRAPPER_PATH"

# -- Banner -----------------------------------------------------------------
cat <<BANNER

    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m
    \033[1;32m  AINode v${AINODE_VERSION} installed!\033[0m
    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m

    Web:     http://localhost:3000
    API:     http://localhost:8000/v1
    Status:  ainode status
    Logs:    ainode logs -f
    Update:  ainode update

    Powered by \033[0;34margentos.ai\033[0m

BANNER
