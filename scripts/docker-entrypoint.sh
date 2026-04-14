#!/usr/bin/env bash
# AINode container entrypoint.
# Delegates to `ainode start --in-container`. The CLI reads
# ~/.ainode/config.json (mounted via -v <host>/.ainode:/root/.ainode) and
# picks solo vs head via `distributed_mode`.
set -euo pipefail

: "${AINODE_HOME:=/root/.ainode}"
mkdir -p "$AINODE_HOME"/{models,logs,datasets,training}

exec ainode start --in-container "$@"
