#!/usr/bin/env bash
# AINode uninstaller.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/getainode/ainode/main/scripts/uninstall.sh | bash
#   scripts/uninstall.sh              # keep data (models, training runs)
#   scripts/uninstall.sh --purge      # also delete ~/.ainode (DESTRUCTIVE)
#
# What this does:
#   1. Stops and disables the systemd service (system + user)
#   2. Removes the unit file(s)
#   3. Stops and removes the running container (if any)
#   4. Removes all AINode container images (GHCR + Docker Hub)
#   5. Removes the /usr/local/bin/ainode host wrapper
#   6. (--purge only) Deletes ~/.ainode — models, configs, training artifacts
#
# Data at ~/.ainode is NOT removed by default. Use --purge to also wipe it.

set -euo pipefail

PURGE=false
for arg in "$@"; do
    case "$arg" in
        --purge) PURGE=true ;;
        -h|--help)
            sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m !!\033[0m %s\n" "$*"; }

# ---------------------------------------------------------------------------
# 1. Stop + disable system-level service
# ---------------------------------------------------------------------------
if systemctl list-unit-files 2>/dev/null | grep -q '^ainode.service'; then
    log "Stopping system service"
    sudo systemctl stop ainode 2>/dev/null || true
    sudo systemctl disable ainode 2>/dev/null || true
fi

if [ -f /etc/systemd/system/ainode.service ]; then
    log "Removing /etc/systemd/system/ainode.service"
    sudo rm -f /etc/systemd/system/ainode.service
fi

sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl reset-failed 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Stop + disable user-level service
# ---------------------------------------------------------------------------
USER_UNIT_PATH="${HOME}/.config/systemd/user/ainode.service"
if systemctl --user list-unit-files 2>/dev/null | grep -q '^ainode.service'; then
    log "Stopping user service"
    systemctl --user stop ainode 2>/dev/null || true
    systemctl --user disable ainode 2>/dev/null || true
fi

if [ -f "$USER_UNIT_PATH" ]; then
    log "Removing $USER_UNIT_PATH"
    rm -f "$USER_UNIT_PATH"
fi

systemctl --user daemon-reload 2>/dev/null || true
systemctl --user reset-failed 2>/dev/null || true

# ---------------------------------------------------------------------------
# 3. Remove running container
# ---------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx 'ainode'; then
        log "Removing ainode container"
        docker rm -f ainode 2>/dev/null || true
    fi

    # ---------------------------------------------------------------------------
    # 4. Remove AINode images — all tags on GHCR and Docker Hub mirror
    # ---------------------------------------------------------------------------
    log "Removing AINode images (this may take a moment)"
    for repo in "ghcr.io/getainode/ainode" "argentaios/ainode"; do
        # List all local tags for this repo and remove each
        while IFS= read -r img; do
            [ -z "$img" ] && continue
            log "  Removing $img"
            docker rmi "$img" 2>/dev/null || true
        done < <(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep "^${repo}:" || true)
    done

    # Also remove the ainode-base image if present
    while IFS= read -r img; do
        [ -z "$img" ] && continue
        log "  Removing $img"
        docker rmi "$img" 2>/dev/null || true
    done < <(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep "^ghcr.io/getainode/ainode-base:" || true)
else
    warn "docker not found — skipping container/image cleanup"
fi

# ---------------------------------------------------------------------------
# 5. Remove host wrapper
# ---------------------------------------------------------------------------
if [ -f /usr/local/bin/ainode ]; then
    log "Removing /usr/local/bin/ainode"
    sudo rm -f /usr/local/bin/ainode
fi

# ---------------------------------------------------------------------------
# 6. (Optional) Remove data directory
# ---------------------------------------------------------------------------
AINODE_DATA="${AINODE_HOME:-${HOME}/.ainode}"

if [ "$PURGE" = "true" ]; then
    if [ -d "$AINODE_DATA" ]; then
        warn "Deleting $AINODE_DATA (models, config, training artifacts)"
        rm -rf "$AINODE_DATA"
        log "Data directory removed."
    fi
else
    if [ -d "$AINODE_DATA" ]; then
        warn "Data kept at $AINODE_DATA (models, config, training runs)."
        warn "To also remove data: run this script with --purge"
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<BANNER

    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m
    \033[1;32m  AINode uninstalled.\033[0m
    \033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m

    To reinstall:
      curl -fsSL https://ainode.dev/install | bash

BANNER
