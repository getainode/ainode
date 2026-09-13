# RESUME — 2026-08-30 18:30 CDT

## Live incident: spark-1 down (memory starvation, my fault)

A `pip install .` of vLLM inside container `glmbuild2` on spark-1 ran **without**
`TORCH_CUDA_ARCH_LIST` / `MAX_JOBS`, so it compiled all architectures across ~20
jobs next to a resident Qwen engine. The box thrashed into starvation around
07:35 CDT.

**State of spark-1 (verified 18:13 CDT, probed from spark-2):**

| Path | Result |
|---|---|
| Tailscale `100.122.26.9` | dead (daemon starved) |
| Mgmt LAN `192.168.0.187` | dead, ARP FAILED |
| RoCE fabric `10.100.0.10` / `.11` | **ping UP both rails** |
| SSH over fabric | TCP connects, times out during banner exchange |

Kernel is alive; sshd is too starved to answer. No BMC/IPMI on a DGX Spark.

**Rescue in progress:** `~/spark1-rescue.sh` running on **spark-2** (started
18:18 CDT, pid 2469745, log `~/spark1-rescue.log`). 40 attempts, each a 10-minute
patient SSH to `sem@10.100.0.10`. On success it kills cc1plus/cicc/ptxas/nvcc/
ninja/pip, and if memory is still tight, `docker kill glmbuild2` (container
filesystem, including our patched source, survives a kill). Freeing memory should
bring sshd, Tailscale, the mgmt NIC, and AINode back on their own.

Fallback if all 40 fail: physical power button at the office. Jason is at the
house, over an hour away — **do not treat that as a quick option.**

## Blast radius

spark-1 hosts the **AINode master**, so AINode's control plane (dashboard, fleet
API, load/eject, master proxy :3000) is down, plus the Qwen lane and whisperlive.
Unaffected and serving: GLM pair :8020 (spark-2/3), Nemotron on spark-4, both Mac
endpoints, Benchy.

## Blocked work (unblocks the moment spark-1 returns)

The vLLM contribution is **written and waiting** in the stopped `glmbuild2`
container: `/src` at af282b77b with our fix to `vllm/platforms/cuda.py` (adds
`FLASHINFER_MLA_SPARSE_SM90` to the major==12 MLA candidate list, which routes
NoPE models around the `pe_dim must be 64 for fp8_ds_mla` wall) plus flashinfer
upgraded to 0.6.18.dev20260819.

Recovery sequence after the box is back:

1. Confirm AINode service self-started and 0.5.6 startup replay restored Qwen :8001.
2. `docker start glmbuild2`
3. **Pinned** rebuild (never omit these pins again):
   ```
   cd /src && TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 VLLM_TARGET_DEVICE=cuda \
     pip install --no-build-isolation --no-deps --force-reinstall .
   ```
4. Verify the cuda.py patch is present and flashinfer is 0.6.18; recommit the
   image; ship to spark-2/3.
5. Run `~/launch-armU-upstream.sh` — expect the SM90 sparse path to engage.
6. If green: suite + needle tests, then open the PR to ZJY0516/vllm (one-file
   cuda.py change, flashinfer >= 0.6.18 note, sm121 verification evidence,
   `jason-code-voice`). Also worth saying upstream should fail loudly rather than
   silently falling back to ds_mla.

## Decisions from today

- **Recovery network:** Jason is adding a 4-port switch as a private "recovery"
  network. My read, given to him: it covers the July failure (dead mgmt switch,
  healthy box) but NOT starvation, since a 4th NIC lands on the same starved
  kernel.
- **Remote power is the real fix.** Recommended: Kasa KP303 x2 for the 120V Spark
  shelf (local `python-kasa` API, no cloud, scriptable over tailnet); APC AP7921B
  or Tripp Lite PDUMH20HVNET for the 200-240V C4130 rack. **Check whether the
  Tripp Lite PDU already in transit is switched, not just metered.** Put the PDU
  on a network path independent of the shelf switch.
- **Build guardrail owed:** memory-capped cgroup wrapper + earlyoom on the Sparks.
  Held off tonight because spark-2/3 are serving GLM live; needs Jason's word.

## PR #53906 today

- New commit `36bb3795b` "Ignore cache opt-outs in partial-hit gating" — prefix
  cache only, nowhere near our fix. Mergeable flag has been flapping; currently
  CONFLICTING (upstream main moved, author owes a rebase). No action for us.
- stefanskiasan posted a large 8x MI350X tuning dossier. Checked against our
  launchers: we never set the two recipe-page env vars that cost him 30 %
  (`VLLM_SSM_CONV_STATE_LAYOUT=DS`, `VLLM_KV_CACHE_LAYOUT=HND`, which are for
  disaggregated serving only); our `--block-size 2304` is safe because the sm120
  kpool-alignment fix is in our image base; his MTP-depth crossover (deep spec
  wins at low concurrency, k=1 wins at saturation) validates our configs, since
  we run 1-6 users. Fold into the artifact on the next update.

## RETRACTED 2026-08-31 10:05 CDT — our cuda.py fix is a no-op on sm121

Rebuilt with proper pins after the reboot, committed the image as
`vllm-glm53-upstream:fix-sm90`, and probed backend selection directly with the
real GLM shape (`head_size=512`, `use_mla=True`, `use_sparse=True`,
`block_size=2304`, verified against the served checkpoint's config.json:
`kv_lora_rank 512`, `qk_rope_head_dim 0`, so NoPE confirmed). On the GB10:

| backend | verdict on sm121 |
|---|---|
| `TRITON_MLA` | rejected, "sparse not supported" |
| `FLASHINFER_MLA_SPARSE_SM90` | rejected, "compute capability not supported" |
| `FLASHINFER_MLA_SPARSE_SM120` | **ACCEPTED** (both kv auto and fp8_e4m3) |

So the SM120 sparse backend is already selected on GB10 with or without our
change, and the SM90 backend is hard-gated off by compute capability — adding it
to the candidate list cannot change anything. **Do not open the PR as planned.**
Upstream solved this with its own SM120 backend (added on this branch before our
base commit); tonyd2wild's SM90-plus-patched-flashinfer route was the workaround
from before that backend existed.

Still open: what actually produced the `pe_dim must be 64 for fp8_ds_mla` wall in
Sunday's pure-upstream serve test. That needs a fresh reproduction on the current
head, which needs two free nodes (evicting Qwen from spark-1 and Nemotron from
spark-4). Not attempted without Jason's word.

## Late-night PR findings (2026-08-31 01:20 CDT) — affects our submission

drakosha published a prebuilt Hopper image and, importantly, **independently
verified `FLASHINFER_MLA_SPARSE_SM90` as the working NoPE path**: 2x H200 NVL,
TP=2, NVFP4, MTP=3, needle 4/4 at 1M tokens, fp8 quality unchanged (gsm8k 0.888
vs 0.880). That corroborates the approach behind our fix.

Three things this changes for our PR:

1. **sm120 is still an open gap and nobody has posted our fix.** drakosha states
   his image "will not work on SM120" and would need ima-helikoptaaa's #10 on
   top (already merged as af282b77b). Our cuda.py selection change remains
   unclaimed.
2. **The branch already lists `FLASHINFER_MLA_SPARSE_SM120` in the `major == 12`
   candidates** (alongside TRITON_MLA). So our PR body must explain *why* that
   is not sufficient for NoPE — presumably SPARSE_SM120 is rejected by
   `supports_combination` for `qk_rope_head_dim == 0`, leaving TRITON_MLA, whose
   ds_mla fp8 cache layout is what trips `pe_dim must be 64`. **Verify this
   claim on the box before writing it down.**
3. **Next wall after ours, already named:** fp8 KV with the SM90 sparse path
   dies at startup with `MLA kv_data_type torch.uint8 is not supported` unless
   the fp8 view dtype is declared when planning. Fix lives in drakosha's open
   `ZJY0516/vllm#9`. Our production launchers run `--kv-cache-dtype fp8_e4m3`,
   so we need that patch too — cite it rather than duplicate it.

Also note: drakosha closed his comment with "AI assistance was used for this
work." That is the disclosure norm on this PR; match it.

Fork PR state: #9 open (drakosha, offloading + fp8 view dtype), #10 merged
(sm120 kpool alignment), #11 open (zigzagcai, short-context skip), #12 merged
(ZJY0516's simplification of #11).

## A/B owed on our hardware (from stefanskiasan's MXFP4 post, 05:20 CDT)

He now says flatly that **`--block-size 2304` from the GLM-5.3 deploy recipe
"actively hurts"** (earlier he qualified it as harmful only without the ROCm
kpool fix). All three of our launchers carry `--block-size 2304`. Worth a
straight A/B on our TP2 pair once spark-1 is back: same prompts, 3-rep medians,
2304 vs the default. Free win if it reproduces on CUDA.

Also from that post, no action but good validation: he measured the MoE running
at ~71 % of its bandwidth roofline, which is the same "bandwidth is the wall"
story as our GB10 design point, and moving FP8 -> 4-bit weights was the only
large lever left (+38 % throughput, +68 % KV). **We are already on NVFP4**, so
we sit on the right side of that lever already.

## Communication rule set today (Jason, firm)

Critical alerts get their **own short message**, first line = what broke + what
is needed, no surrounding essay, plus a phone push. He missed the outage the
first time because I buried it in a status wall. Also: **the Sparks are at the
office, over an hour from the house.** Never assume he can press a button.
