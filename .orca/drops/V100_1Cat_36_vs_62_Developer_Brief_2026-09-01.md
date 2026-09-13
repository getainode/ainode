# Why this four-V100 server is at 36 tok/s instead of 62+ tok/s

**Developer investigation brief — 2026-09-01**

## Bottom line

The present **36 tok/s is not established as the hardware limit**. It is much more consistent with an old or non-admitted 1Cat execution route.

The most important evidence is 1Cat's matched 128K A/B test:

| Exact 128K target-only decode | Result |
|---|---:|
| Previous E4M3 long-context route | 40.561 tok/s |
| New LUT merged-wave route | 61.834 tok/s |

Our 36 tok/s is close to the 40.561 tok/s control. That makes a missing long-context kernel, old binary, wrong KV format, or route-admission failure more likely than a fundamental V100 compute limit.

There is also a likely version trap: formal release **v1.3.0 was published August 17**, while the 61.834 tok/s long-decode change merged in **PR #285 on August 25**. A normal v1.3.0 wheel cannot contain that change. The DFlash2 q16 safety fix landed August 29, and the validated 1.5.0 wheel candidate was tested August 31 but was explicitly **not yet published or tagged**. Therefore, “we installed 1Cat 1.3” and “we are testing the 61.8/206 path” are mutually inconsistent.

One major qualification remains: the published 61.834 result used **four V100-SXM2-32GB GPUs with full NVLink connectivity**. This server appears to use four **PCIe V100 32GB** cards. Dell documents that C4130 PCIe configuration G provides peer-to-peer access only within GPU pairs 0/1 and 2/3, not across all four. TP4 must cross that boundary. The software gap should be fixed first, then the PCIe topology tax measured rather than guessed.

The **206 tok/s DFlash2 result is real but is not the same benchmark**. It is speculative decode: the target verifies blocks proposed by a drafter and emits multiple accepted tokens per engine round. The cited run averaged 3.599 emitted tokens per 17.463 ms round. Its speed depends heavily on acceptance rate and workload. It should be tested only after the target-only route is correct.

Finally, bonding two 50 Gb NICs does **not** improve local TP4 GPU-to-GPU communication. It helps network storage/offload when flows can use both links; it does not turn PCIe into NVLink.

## What the published numbers actually mean

| Number | Contract | Correct interpretation |
|---|---|---|
| 61.834 tok/s | Qwen3.8-27B-NVFP4; 4× V100-SXM2-32GB; TP4; E4M3 KV; Flash-V100; full CUDA Graph; no MTP; exact final 128K context; 64 generated tokens | Target-only pure decode. Prefill and TTFT excluded. This is the clean comparison target. |
| 50.376 tok/s | Same target-only contract at exact final 256K context | Measured endpoint, not a projection. |
| 206.06 tok/s | Qwen3.8-27B-NVFP4 + DFlash2; historical web prompt; 512 output tokens; 3.599 emitted/round | Workload-dependent speculative throughput, not single-token target speed. |
| 251.60 tok/s | DFlash2 on one high-acceptance MBPP request | Useful evidence that the path works, but not a universal rate. |
| Approximately 260 tok/s | Real-machine demo headline | Not a fixed service guarantee. 1Cat explicitly warns that its benchmark contracts differ. |

The reference 128K request finishes at exactly 131,072 context tokens after generating 64 tokens. Pure decode is measured over the 63 steady token intervals. Its recorded prefill took roughly 283 seconds and is excluded from the 61.834 tok/s figure. Do not compare that number with end-to-end request throughput.

## Ranked fault tree

| Rank | Suspected cause | Confidence | Fast discriminator |
|---:|---|---|---|
| 1 | Installed 1Cat source/wheel predates PR #285, or Python source is overlaid on older native extensions | Very high | Record package version, source commit, extension imports and hashes. A stock v1.3.0 wheel is too old by construction. |
| 2 | The intended NVFP4/FP8 projection routes are not admitted | High | Startup/route logs must identify QPN8/TurboMind routes rather than generic compressed-tensors, Marlin, cuBLAS, or another fallback. The model repository, revision and config hash must match the tested checkpoint contract. |
| 3 | E4M3 long-context XQA merged-wave route is absent or disabled | High | Run a matched A/B with its three route gates enabled and disabled. If performance and kernel traces do not change, the fast route is not entering. |
| 4 | TP4 communication is limited by the C4130 PCIe topology | High | Compare P2P matrices, 2-GPU within-pair collectives, and 4-GPU collectives. If the model fits, compare TP2 on 0/1 and 2/3 against TP4. |
| 5 | Full CUDA Graph replay is not active | High | Capture startup logs and verify graph mode. Eager or partial fallback is not the published contract. |
| 6 | Benchmark definitions differ | High | Match context tokens, output length, batch/concurrency, KV dtype, warmup policy, and pure-TPOT calculation. Exclude prefill/TTFT. |
| 7 | KV cache is FP16 or E5M2 rather than E4M3 for the target-only 61.8 comparison | High | Record the resolved KV dtype from engine logs. The 61.834 target-only test used E4M3; the recommended DFlash2 service command uses E5M2. Do not merge those contracts. |
| 8 | Custom all-reduce/P2P is disabled, blocked by ACS/IOMMU, or falling through host memory | Medium-high | Check `nvidia-smi topo`, NCCL diagnostics and 1Cat route logs. Compare with controlled collective tests. |
| 9 | Power, clocks, thermals, or competing processes | Medium | Verify P0 clocks, throttle reasons, power cap, GPU utilization and exclusive idle state during the run. This alone is unlikely to explain the full gap. |
| 10 | DFlash2 has low acceptance or a q8/q16 native-extension mismatch | High for the 200+ target | Verify grouped-verifier max Q, emitted tokens per round, round latency and acceptance length. Do this after target-only validation. |

## Why build mismatch is the leading diagnosis

The formal v1.3.0 release commit is `6ada86e` and predates PR #285. The long-context PR's exact endpoint A/B was:

- 128K: **40.561 → 61.834 tok/s**
- 256K: **27.456 → 50.376 tok/s**

The broader Qwen3.8 NVFP4 design record also shows an original mixed-checkpoint route at **29.35 tok/s**, then successive projection and attention-route admissions reaching 57–71 tok/s on its short-context frozen contract. This means a service can be “running 1Cat” while still spending most of its time in generic or older operators.

For the current DFlash2 line, 1Cat recommends Python 3.12, CUDA 12.8, PyTorch 2.10 and SM70. PR #427 tested a 1.5.0 release candidate from head `56db3c1` with final wheel SHA256 `9dbb1118d670f081563b202127e77af41f9261d9ec7f7a829ec1357f53037d71`, but states that the artifact was not published or tagged. If building locally, pin and record the exact commit; do not use an unrecorded floating `main`.

## Investigation procedure

### Gate 0 — Freeze the current state before changing it

Save the exact service command, environment and startup log. Then collect:

```bash
date -u
uname -a
nvidia-smi
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi topo -p2p n
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE
env | sort | grep -E '^(CUDA|NCCL|VLLM|TORCH)'
python -m pip freeze
python -m pip show -f vllm
```

If this driver does not support either `topo -p2p` form, record that and run `nvidia-smi topo -m` plus NVIDIA's `p2pBandwidthLatencyTest` instead.

Run 1Cat's native-capability check:

```bash
python - <<'PY'
import sys
import torch
import vllm
import flash_attn_v100
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA ABI:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("vLLM:", vllm.__version__)
print("vLLM path:", vllm.__file__)
print("flash_attn_v100:", flash_attn_v100.__version__)
print("flash_attn_v100 path:", flash_attn_v100.__file__)
print("DFlash2 grouped verify max Q:",
      flash_attn_grouped_verify_max_query_tokens())
PY
```

For DFlash2 q16, the capability result must be **16**. An older native extension reports 8. PR #422 documents a real failure caused by newer Python code driving an older q8-only native binary, so package version strings alone are insufficient.

Also record:

- exact target model repository/path and immutable revision;
- SHA256 of `config.json` (the private reference contract records `1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c`);
- quantization method and weight/scale layout resolved by the loader;
- every relevant `.so` path and SHA256;
- whether the install is a wheel, editable checkout, container layer, or Python source overlay;
- whether any other process occupies a GPU.

### Gate 1 — Prove the interconnect

The expected C4130 PCIe topology is two P2P pairs, not one four-GPU peer domain. Verify rather than assume it.

Build and run NVIDIA's collective test if it is not already present:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=0 CUDA_HOME=/usr/local/cuda

NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P \
./build/all_reduce_perf -b 4K -e 64M -f 2 -g 4
```

Then run 2-GPU tests separately on physical pairs 0/1 and 2/3, preserving the actual bus ordering shown by `nvidia-smi topo -m`. Retain both algorithm bandwidth and bus bandwidth. Check the kernel command line and logs for IOMMU/ACS behavior if P2P is unexpectedly absent; NVIDIA recommends diagnosing topology and P2P before tuning NCCL knobs.

This test will not by itself reproduce 1Cat's custom TP all-reduce, but it establishes whether cross-pair traffic is the likely hardware bottleneck.

### Gate 2 — Reproduce target-only decode before adding DFlash2

Use one pinned, source-independent build that contains PR #285 and its native Flash-V100 extension. Match this contract:

- Qwen3.8-27B-NVFP4 checkpoint and exact revision;
- four V100s, TP4;
- `FLASH_ATTN_V100`;
- FP16 activations/output;
- **E4M3 KV**;
- `max_model_len=262144`;
- `max_num_batched_tokens=8192`;
- `max_num_seqs=1`;
- prefix caching, aligned Mamba state, chunked prefill;
- `FULL_AND_PIECEWISE` CUDA graphs;
- no MTP, no DFlash, no concurrent requests;
- exact final context 128K, 64 generated tokens;
- temperature 1.0, top-p 0.95, top-k 20;
- exclude the first cold/JIT request and report the 63 steady decode intervals.

For a controlled route A/B, explicitly capture the resolved values of these three PR #285 gates:

```bash
VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO=1
VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS=1
VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH=1
```

Repeat with the relevant fast route disabled as a control. The point is not to guess a good environment-variable cocktail; it is to prove that the candidate binary produces a different admitted kernel route and measurable TPOT change.

Also run the frozen short-context guard (1,024 input / 256 output) after warmup. Interpret the pair of results as follows:

| Observation | Interpretation |
|---|---|
| Short test is roughly 29–40 tok/s | Projection routes or build are still on the original/fallback path. Do not blame long-context attention yet. |
| Short test is roughly 65–83 tok/s, but 128K remains near 36–41 | The general model route is healthy; the long E4M3 XQA route is absent, disabled, or not admitted. |
| Route-off is near 36–41 and route-on rises materially | Root cause found: old/disabled long-context route. |
| Route-on is logged and traced but remains slow | Measure TP4 communication and PCIe topology next. |
| TP2 on either physical pair beats TP4 at short context | Cross-pair PCIe communication is a real limiter. If 128K does not fit TP2, use the short test plus collective measurements. |
| Raw decode is fast but reported service throughput is low | Benchmark accounting, queueing, prefill, concurrency or client timing is the gap. |

Do not demand exactly 61.834 tok/s from PCIe hardware. A reasonable first target is to prove the fast route and get materially above the old 40.561 control. The residual delta to the SXM2 reference can then be attributed with TP all-reduce and kernel timing evidence.

### Gate 3 — Add DFlash2 only after Gate 2 passes

The current 1Cat README's validated service contract uses:

```bash
vllm serve /path/to/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b-dflash2 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 262144 \
  --performance-mode interactivity \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto"}' \
  --host 0.0.0.0 \
  --port 8000
```

Before interpreting tok/s, record:

- target-only TPOT from Gate 2;
- complete DFlash round latency;
- mean and distribution of emitted tokens per round;
- accepted length / acceptance rate by request type;
- q8/q16 grouped-verifier capability;
- warm versus cold prefix behavior;
- output quality on a fixed task set.

At 3.599 emitted tokens per 17.463 ms round, the arithmetic is approximately 206 emitted tok/s. If this workload accepts close to one token per round, DFlash2 cannot provide that multiplier and may add overhead. Treat 200+ as a workload-dependent product mode, not the baseline hardware rate.

## The result matrix the developer should return

| Build / route | TP | Context | KV | Speculation | Pure TPOT | Decode tok/s | All-reduce evidence | Route evidence |
|---|---:|---:|---|---|---:|---:|---|---|
| Current production | 4 | 1K | record | off |  |  |  |  |
| Current production | 4 | 128K | record | off |  |  |  |  |
| Pinned current 1Cat, fast route off | 4 | 128K | E4M3 | off |  |  |  |  |
| Pinned current 1Cat, fast route on | 4 | 128K | E4M3 | off |  |  |  |  |
| Pinned current 1Cat | 2 on pair 0/1 | 1K | E4M3 | off |  |  |  |  |
| Pinned current 1Cat | 2 on pair 2/3 | 1K | E4M3 | off |  |  |  |  |
| Pinned current 1Cat | 4 | representative workload | E5M2 | DFlash2 |  |  | emitted/round: | max Q: |

Attach raw logs and JSON, not only a spreadsheet summary. The minimum useful handback is:

1. exact launch command and environment;
2. source commit, package paths and native-library hashes;
3. model/checkpoint revision and `config.json` hash;
4. startup route/admission and CUDA Graph logs;
5. `nvidia-smi` topology/P2P output and 2-GPU/4-GPU collective results;
6. raw timing records separating prefill, TTFT and steady decode;
7. DFlash2 round latency and acceptance metrics, if enabled.

## Decision rule

Do not buy or change hardware to solve the 36 tok/s result until the pinned-build route A/B is complete.

- If the fast route produces a large gain, the primary defect was software/build configuration.
- If the fast route is proven active but TP4 remains far below the SXM2 result and TP2/collectives expose a cross-pair cliff, the remaining limitation is the PCIe C4130 topology.
- If target-only is healthy but DFlash2 disappoints, optimize acceptance and verifier compatibility rather than target kernels.

## Primary sources

- [1Cat-vLLM README and current benchmark contracts](https://github.com/1CatAI/1Cat-vLLM)
- [1Cat v1.3.0 release, August 17, 2026](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0)
- [PR #285: exact 128K/256K E4M3 long-decode A/B](https://github.com/1CatAI/1Cat-vLLM/pull/285)
- [Qwen3.8 NVFP4 SM70 decode design and route record](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/sm70_qwen38_nvfp4_decode.md)
- [PR #422: DFlash2 q8/q16 native capability fix and 206.06 tok/s record](https://github.com/1CatAI/1Cat-vLLM/pull/422)
- [PR #427: validated 1.5.0 release candidate and artifact hash](https://github.com/1CatAI/1Cat-vLLM/pull/427)
- [Official DFlash2 drafter model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- [Dell C4130 V100 PCIe/SXM2 topology and P2P comparison](https://dl.dell.com/manuals/all-products/esuprt_software/esuprt_it_ops_datcentr_mgmt/high-computing-solution-resources_white-papers27_en-us.pdf)
- [NVIDIA NCCL GPU/topology troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)

