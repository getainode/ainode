#!/usr/bin/env bash
# AINode container entrypoint.
# Delegates to `ainode start --in-container`. The CLI reads
# ~/.ainode/config.json (mounted via -v <host>/.ainode:/root/.ainode) and
# picks solo vs head via `distributed_mode`.
set -euo pipefail

: "${AINODE_HOME:=/root/.ainode}"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

# If the host mounted SSH keys at /root/.ssh (read-only bind of $HOME/.ssh),
# they'll be owned by the host's UID (uid 1000) and OpenSSH refuses to use
# them as root. Copy to an in-container root-owned .ssh dir and re-point.
if [ -d /root/.ssh ] && [ "$(stat -c %u /root/.ssh 2>/dev/null || echo 0)" != "0" ]; then
    mkdir -p /root/.ssh-runtime
    cp -r /root/.ssh/. /root/.ssh-runtime/ 2>/dev/null || true
    chown -R root:root /root/.ssh-runtime
    chmod 700 /root/.ssh-runtime
    find /root/.ssh-runtime -type f -exec chmod 600 {} +
    find /root/.ssh-runtime -type f -name '*.pub' -exec chmod 644 {} +
    # config file should be 600 (ssh is strict about this).
    [ -f /root/.ssh-runtime/config ] && chmod 600 /root/.ssh-runtime/config
    export HOME=/root
    # Replace /root/.ssh by a bind mount so tools pick it up transparently.
    if mountpoint -q /root/.ssh 2>/dev/null || [ -r /root/.ssh ]; then
        mount --bind /root/.ssh-runtime /root/.ssh 2>/dev/null || {
            # Fall back to re-pointing via GIT_SSH_COMMAND / ssh_config
            true
        }
    fi
fi

exec ainode start --in-container "$@"
