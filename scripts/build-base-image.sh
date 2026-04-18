#!/usr/bin/env bash
# Build ghcr.io/getainode/ainode-base from eugr/spark-vllm-docker at a pinned
# commit. This is the "base image" that Dockerfile.ainode extends.
#
# End users never run this — only our CI runner (or maintainers rebuilding
# from source). Takes ~12 minutes on a DGX Spark with prebuilt vLLM wheels.
#
# Usage:
#   scripts/build-base-image.sh                 # build, tag locally
#   scripts/build-base-image.sh --push          # build + push to GHCR (requires docker login)
set -euo pipefail

# -- Pinned eugr reference ---------------------------------------------------
# Bumped by editing this file and committing; CI/build reproducibility
# depends on this being explicit.
EUGR_REPO="${EUGR_REPO:-https://github.com/eugr/spark-vllm-docker}"
EUGR_COMMIT="${EUGR_COMMIT:-c026c92bd0c1236f947ac212565b15a33ba1b4e7}"
EUGR_SHORT="${EUGR_COMMIT:0:7}"

# AINode version coupling (kept in sync with pyproject.toml).
AINODE_VERSION="${AINODE_VERSION:-0.4.0}"

REGISTRY="${REGISTRY:-ghcr.io/getainode}"
BASE_NAME="${BASE_NAME:-ainode-base}"

PUSH="false"
for arg in "$@"; do
    case "$arg" in
        --push) PUSH="true" ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$SCRIPT_DIR/_eugr"

echo "==> Preparing eugr worktree at $WORKTREE (commit $EUGR_SHORT)"
if [ -d "$WORKTREE/.git" ]; then
    git -C "$WORKTREE" fetch --depth=1 origin "$EUGR_COMMIT"
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
else
    rm -rf "$WORKTREE"
    git clone --depth=1 "$EUGR_REPO" "$WORKTREE"
    # Shallow clone can't check out an arbitrary sha in one shot; deepen.
    git -C "$WORKTREE" fetch --depth=50 origin "$EUGR_COMMIT" || \
        git -C "$WORKTREE" fetch --unshallow
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
fi

# -- Patch eugr's Dockerfile: vanilla NCCL for switched topology ------------
# Eugr's Dockerfile defaults to building NCCL from the ``dgxspark-3node-ring``
# branch of zyang-dev/nccl. That patch is for a 3-node direct-connect mesh
# topology only (see eugr's docs/NETWORKING.md § "3-node mesh", line 400+).
# For switched topology — our target: Sparks connected via a QSFP switch —
# eugr's own docs (NETWORKING.md § "Dual Sparks or Sparks via QSFP switch",
# line 355+) specify vanilla NCCL v2.28.3-1 from github.com/NVIDIA/nccl.git.
# Running the mesh patch on switched topology produced NCCL init hangs in
# ncclCommInitRank; see ops/runbooks/2026-04-18-v0.4.9-verification.md for
# the evidence trail.
#
# We swap the clone line in place rather than forking eugr's Dockerfile so
# EUGR_COMMIT bumps pull cleanly and this single line is the only re-audit.
# If eugr changes the NCCL source format and the grep below doesn't match,
# the script exits — a silently-skipped sed would leave the mesh NCCL in
# place and the hang would reappear.
DOCKERFILE="$WORKTREE/Dockerfile"
OLD_NCCL='git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git'
NEW_NCCL='git clone -b v2.28.3-1 https://github.com/NVIDIA/nccl.git'
echo "==> Patching eugr Dockerfile: vanilla NCCL for switched topology"
if grep -qF "$OLD_NCCL" "$DOCKERFILE"; then
    sed -i "s|$OLD_NCCL|$NEW_NCCL|" "$DOCKERFILE"
    echo "    swapped: dgxspark-3node-ring (mesh) -> v2.28.3-1 (vanilla, switched)"
elif grep -qF "$NEW_NCCL" "$DOCKERFILE"; then
    echo "    already patched (idempotent re-run)"
else
    echo "!! expected NCCL clone line not found in $DOCKERFILE" >&2
    echo "   Eugr may have changed the NCCL source; re-audit the patch in this script." >&2
    exit 1
fi

echo "==> Building eugr image (prebuilt vLLM wheels expected; ~12 min)"
pushd "$WORKTREE" >/dev/null
./build-and-copy.sh --tag "vllm-node:${EUGR_SHORT}"
popd >/dev/null

# Tag the base image for our registry.
# Two tags:
#   <registry>/<name>:<eugr-commit>       — the eugr content identity
#   <registry>/<name>:<version>-<eugr>    — correlates ainode version
COMMIT_TAG="$REGISTRY/$BASE_NAME:$EUGR_SHORT"
VERSION_TAG="$REGISTRY/$BASE_NAME:${AINODE_VERSION}-${EUGR_SHORT}"

docker tag "vllm-node:${EUGR_SHORT}" "$COMMIT_TAG"
docker tag "vllm-node:${EUGR_SHORT}" "$VERSION_TAG"
echo "==> Tagged:"
echo "    $COMMIT_TAG"
echo "    $VERSION_TAG"

if [ "$PUSH" = "true" ]; then
    echo "==> Pushing"
    docker push "$COMMIT_TAG"
    docker push "$VERSION_TAG"
fi

echo "==> Done. To build the ainode layer:"
echo "    docker build -f scripts/Dockerfile.ainode -t ainode:${AINODE_VERSION} ."
