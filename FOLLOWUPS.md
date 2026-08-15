# FOLLOWUPS (Dart-unreachable fallback — migrate to Dart board when authed)

## [ainode/lab] Replace the flaky 10G mgmt switch (root cause of the Jul 26–Aug 13 outage)
- **Filed:** 2026-08-13 (fleet restored this session; switch is revived but NOT trusted)
- **Owner:** Jason. The 10G copper switch feeding spark3+spark4 mgmt (+ the `192.168.0.100` device + ROSA `ether3`) flapped Jun 18 and Jul 6, died Jul 26 16:33 (same electrical event tripped spark2's outlet), and needed a power-cycle + one cable reseat (spark3's port) to revive on 2026-08-13. Classic dying PSU / failing unit.
- **Next action:** replace the switch (or at minimum its PSU); while at it, identify what owns `192.168.0.100` — still down after recovery, either on a dead port or powered off. Evidence + topology: `ops/runbooks/network-topology.md` outage note.
- **Proof of closure:** new/verified switch in place; 2 weeks with no synchronized link-flaps in spark3/4 `journalctl -k`; `.100` owner identified and documented in the runbook.

## [ainode] BUG: eject doesn't survive reboot (replay resurrects ejected instances) + failed loads leave phantom registry rows
- **Filed:** 2026-08-13, observed live on spark4 (0.5.3).
- **Repro:** (1) eject instance via `POST /api/server/models/<id>/eject` → OK; reboot node → ainode replay relaunches the ejected instance (Qwen2.5-0.5B came back). Eject removes from the in-memory registry but evidently not from the persisted replay set. (2) `POST /api/models/load` whose engine launch fails its memory pre-check leaves a `ready:false` / "launching" row in `/api/server/status` with NO container behind it — phantom, never reaped, no error surfaced to the caller.
- **Also (2026-08-14):** boot-time engine launch can wedge silently when the system clock NTP-jumps right after ainode starts (spark4 booted with a ~13h-stale clock; banner printed "Engine starting in background", no engine container was ever created, no error logged, and subsequent `/api/models/load` requests queued forever behind it). Engine-launch timers/timeouts should be monotonic-clock based, and a launch that produces no container within N minutes should be marked failed and released.
- **Also (2026-08-14, root-cause class):** engine containers launch with `docker run --rm -d` (nvidia.py `_build_solo_docker_cmd`) — an engine that dies during startup REMOVES ITSELF, leaving zero logs and zero `docker ps -a` corpse; `start_solo()` returns True if the docker CLI merely spawned (`poll() is None`), so AINode never notices. ~18 self-erased corpses found as bare container-ID hashes in `~/.ainode/logs/nvidia-vllm.log` on spark4. Drop `--rm` (the idempotent pre-launch stop/rm already handles leftovers) + have the manager health-check the container within N seconds of launch. Additionally: the boot-path banner launch on `engine_strategy: pip` falls into the legacy host-venv VLLMEngine inside the slim container and dies on `No module named 'vllm'` (see `~/.ainode/logs/vllm.log`) — boot replay should honor engine_backend=nvidia, same as the API path.
- **Also (2026-08-14, sizing):** VLM loads need modality-aware sizing — Qwen2.5-VL-7B at gmu 0.20 passes the 0.90 admission gate, loads 15.6 GiB of weights, then the vision **encoder cache** (profiled for max-size video, 114K-token budget) leaves KV at **-7.97 GiB** → engine dies post-admission with `No available memory for the cache blocks`, invisible to the caller. Working config: gmu 0.30 + max_model_len 32768. The stacked-load admission check should estimate weights+encoder overhead per modality (or at least surface the engine's death reason back through the API).
- **Also (2026-08-15, fleet-level symptom — the user-visible one):** a node whose engine died keeps advertising its model fleet-wide. spark-3's engine container is gone (only `ainode` running) yet `/api/nodes` still reports `models=chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4` and the master's **`/v1/models` menu on spark-1 lists it as available**; an actual request correctly 404s `model_not_found`. So the router is honest at request time but the *menu is a phantom* — a client picking from `/v1/models` gets a model that cannot be served. Node state should be reconciled against the live engine (heartbeat/health-check per instance) before it's advertised. Directly contradicts the 0.5.3 "truthful instances everywhere" goal.
- **Proof of closure:** eject → reboot → instance stays gone; failed load → status shows failure reason, no phantom row; simulated clock jump during launch doesn't wedge the loader; VLM load at undersized gmu is rejected at admission with a sizing hint (not a silent post-admission death); kill an engine container out-of-band → within one heartbeat the model disappears from the master's `/v1/models`.

## [ainode] Nemotron 3.5 Lightning native support — launch-path gaps
- **Filed:** 2026-08-13. Jason: "it would be really nice if AInode could do this natively."
- **Owner:** next AINode dev session. `NvidiaBackend._build_vllm_serve_args` (`ainode/engine/backends/nvidia.py:931`) cannot emit: `--moe-backend`, `--mamba-backend`, `--mamba-cache-mode`, `--speculative_config.*` (DSpark), `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`, `--enable-prefix-caching`. Engine image is fleet-global (`NVIDIA_VLLM_IMAGE`, default `scitrera/dgx-spark-vllm:0.17.0-t5`, vLLM 0.17.1) but the model needs `vllm/vllm-openai:v0.27.1`; `--enforce-eager` is hardwired; 0.17-era NVFP4 marlin env vars may conflict on 0.27.1.
- **Next action:** per-model `extra_vllm_args` passthrough + per-instance engine-image override in config/launch path; then serve Nemotron-3.5-Lightning through AINode (dogfood rule). Official recipe: HF model card, "1x DGX Spark (GB10)".
- **Proof of closure:** Nemotron 3.5 Lightning + DSpark launched from the AINode UI on spark4, visible in the dashboard, using the card recipe flags.

## [dell-r750] Second A40 DEFECTIVE — RMA in progress (2026-07-21)
- **Owner:** Richard (Jason sent him the evidence bundle 2026-07-21 evening). Card fails init via BOTH GSP (`0x62:0x65:2416`) and non-GSP (`0x25:0xffff:1480`) paths in validated slot 2 @ x16 with correct SIG_PWR_0 power; survived-cold-boot-unchanged; iDRAC reads PN/serial as N/A; BAR1 stuck at 256MB vs twin's 64GB. Verdict: dead firmware storage. Source: eBay item 187541687374.
- **Next action:** Richard files the eBay return; photograph physical serial sticker before shipping.
- **Proof of closure:** refund/replacement received; replacement card shows in `nvidia-smi -L` as GPU 1.

## [dell-r750] BOSS-S2 module replacement pending (FGNRW, $299, ETA ~2026-07-23/24)
- **Owner:** Jason. J_PWR_1 header pins snapped; currently running on a field repair (6-pin housing seated on the 3 surviving pins, orientation verified pin1=yellow both ends). Works, but unlatched+unretained.
- **Next action:** when module arrives — maintenance window: swap the two M.2 carriers into new module, connect 05HVX9 + signal cable, boot (BOSS-S2 auto-recognizes the RAID-1). Same window: reseat **PSU 1** AC cord/PDU outlet (iDRAC 2026-07-16: "PSU 1 is not receiving input power"; recurring since June).
- **Note:** working A40 currently runs GSP-firmware-OFF mode (side effect of diagnosis; fully supported, services healthy). Self-reverts to GSP default at that reboot — no action needed.
- **Proof of closure:** BOSS shows healthy in iDRAC storage, boot works, no PSU 1 AC-loss events for 2 weeks.

## [dell-r750] Orphaned Coolify proxy `yc8ck0w4ok4oc4gsgg4so40o-proxy` — stopped 2026-07-21
- Proxy container crash-looped (nginx upstream app container no longer exists — app was deleted, proxy left behind). Stopped it (`docker stop`); `unless-stopped` policy means it stays down across reboots. If the app is ever redeployed via Coolify it recreates its own proxy. **Next action:** delete the container + image via Coolify UI cleanup when convenient. Proof: `docker ps -a --filter name=yc8ck` shows Exited or nothing.

## [dell-r750] Foothold app: empty `levels/` content dir — container crash-looping since ~Jul 4
- **Filed:** 2026-07-21 (during 2nd-A40 install prep)
- **Owner:** Jason (app owner) — needs app knowledge Claude doesn't have
- **State:** Container `foothold-t11vcw03w9wnrsxngkkg3u2r-*` on the Dell R750 restart-loops.
  Two root causes found; first one FIXED this session:
  1. ~~SQLite volume `t11vcw03w9wnrsxngkkg3u2r_dta-data` owned root:root while container runs as `node` (uid 1000) → SQLITE_READONLY~~ — fixed with `chown -R 1000:1000` on `/data/docker/volumes/t11vcw03w9wnrsxngkkg3u2r_dta-data/_data` 2026-07-21.
  2. **OPEN:** host dir `/data/coolify/applications/t11vcw03w9wnrsxngkkg3u2r/Foothold/levels/` exists but is EMPTY; app needs `/levels/manifest.json` (read-only bind mount). Content was never deployed.
- **Next action:** Populate the `levels/` dir from the Foothold app repo (or redeploy via Coolify with the seed step), then confirm the container goes healthy.
- **Proof of closure:** `docker ps --filter name=foothold` shows `Up … (healthy)` and stays up >10 min.
