#!/usr/bin/env bash
# Build the GLM-5.3-Flash engine image AINode launches (catalog entry
# "glm-5.3-flash-nvfp4-dflash2") and copy it to the peer node.
#
#   scripts/build-glm53-flash-image.sh RECIPE_DIR [PEER_SSH_TARGET]
#
# RECIPE_DIR is a checkout of coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
# at ab1102a, on the node that already has the base image (built there with
# `./serve-profile.sh build nvfp4-fp4-dcp2`). PEER_SSH_TARGET, when given, gets
# the finished image by `docker save | ssh PEER docker load`, because the mp
# launch runs the same tag on every rank and nothing pulls a local build.
#
# Refuses, before building anything, when the commit or any baked file differs
# from what the catalog entry was written against. Run it as the user that owns
# docker on the node, never with sudo.
set -euo pipefail

RECIPE_COMMIT=ab1102a8fd2c8a7102bdbfcda0f1402e37504073
BASE=${BASE:-glm53-v14:nvfp4-gscale-tooling}
IMAGE=${IMAGE:-glm53-flash-ainode:nvfp4-fp4-dcp2-ab1102a}
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

recipe=${1:?usage: build-glm53-flash-image.sh RECIPE_DIR [PEER_SSH_TARGET]}
peer=${2:-}

head=$(git -C "$recipe" rev-parse HEAD)
[[ "$head" == "$RECIPE_COMMIT" ]] || {
  echo "recipe checkout is at $head, expected $RECIPE_COMMIT" >&2; exit 1; }

(cd "$recipe" && sha256sum -c --quiet -) <<'SUMS'
642a52e59f4d3dadd6913e68c40eddbf6428b8b0586c2da467bc1aa73821178e  docker/sparse_attn_indexer_kpool_sm121.py
59876937beb90e37861d6652f2461197c777e53682c4a8d60cab6977143a65f4  docker/glm_decode_first_scheduler.py
419541bac92bf1015909175bc60a1ff276da469f9891530643558d88dc1a6e10  scheduler_profiles/adaptive.json
3464df90608f813c9e56215dfcac708bf0c90181850109391de1e4e398e00a42  chat_template_mm.jinja
0b58abcbaff2e30d2c51809895e15d44000e976f71cbd43945d1bb9d4347a97f  kv_calibration/results/nvfp4-gscale-v1.json
SUMS

docker image inspect "$BASE" >/dev/null 2>&1 || {
  echo "base image $BASE is not on this node; build it from the recipe first:" >&2
  echo "  (cd $recipe && ./serve-profile.sh build nvfp4-fp4-dcp2)" >&2
  exit 1; }

docker build -f "$HERE/Dockerfile.glm53-flash" --build-arg "BASE=$BASE" -t "$IMAGE" "$recipe"

if [[ -n "$peer" ]]; then
  echo "copying $IMAGE to $peer"
  docker save "$IMAGE" | ssh -o BatchMode=yes "$peer" docker load
  want=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  got=$(ssh -o BatchMode=yes "$peer" docker image inspect --format '{{.Id}}' "$IMAGE")
  [[ "$want" == "$got" ]] || { echo "peer image id $got != $want" >&2; exit 1; }
fi
echo "built $IMAGE"
