# scripts/run_cluster.sh — vendored provenance

- **Source:** https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
- **Upstream repo:** https://github.com/vllm-project/vllm (branch: `main`)
- **Date vendored:** 2026-04-19
- **SHA256:** `f37a33e80e231c9c52fa7ca3b93fb75f63f7b15fcd5ec860fcc4537cc5a0aaeb`
- **Size:** 4,536 bytes, 131 lines
- **Mode:** `0755` (executable — invoked directly by `NvidiaBackend`)

This is NVIDIA's + vLLM upstream's canonical Ray-cluster-in-Docker launcher
used by the `nvidia-vllm-engine` slice (see
`ops/slices/nvidia-vllm-engine/PLAN.md` § Phase 4). We ship a verbatim copy
inside the AINode image so installs are reproducible and offline-capable —
see D3 in the design diary. Do NOT modify the contents; if an upstream
change is needed, re-fetch and update SHA + date above in the same commit.

Update procedure:

```bash
curl -fsSL -o scripts/run_cluster.sh \
  https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod 755 scripts/run_cluster.sh
shasum -a 256 scripts/run_cluster.sh   # update SHA256 above
```
