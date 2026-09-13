#!/usr/bin/env python3
"""ainode-bench - measure what a spec sheet does not, on an AINode-served model.

    scripts/ainode-bench.py --url http://100.72.9.84:8000 \
        --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
        --ainode http://100.72.9.84:3000 --label dspark-recipe

    scripts/ainode-bench.py ... --only prefill,concurrency --depths 4000,32000
    scripts/ainode-bench.py --show bench/results/<file>.json

Five sections, each answering a question a tok/s headline does not:

  single       what one user feels right now: TTFT + decode on a short prompt.
  prefill      what long context costs. Decode is quoted at 4k and used at 120k;
               those are different numbers. x-axis is the SERVER's prompt_tokens.
  sustained    does throughput hold over one long unbroken generation, or sag as
               the KV cache grows and the node heats.
  concurrency  aggregate and per-stream throughput at 1/2/4/8/16 streams. This is
               the number that matters for agents and multi-user serving; on this
               hardware it is several times the single-stream number.
  reasoning    the reasoning tax: identical prompt, thinking on vs thinking off.

Honesty rules, all of them load-bearing:

  * Prompt token counts come from the server's ``usage.prompt_tokens`` via
    ``stream_options.include_usage``, never from a chars-per-token estimate. A
    bad guess would shift the entire prefill x-axis and nobody would see it.
  * Generated token counts come from ``usage.completion_tokens``, never from
    counting SSE chunks. Under speculative decoding (DSpark, MTP) one chunk can
    carry several accepted tokens, so chunk-counting silently halves the rate.
  * Every prompt carries a unique nonce at the FRONT, so ``--enable-prefix-caching``
    cannot serve a cached prefill and make depth look free. A trailing nonce
    would leave the whole prefix cacheable and we would be timing a cache hit.
  * Decode rate EXCLUDES prefill: the clock starts at the first content delta.
    Folding TTFT in turns a decode number into an agent-loop number.
  * TTFT includes queueing and scheduling, so ``prefill_tok_s`` derived from it
    (prompt_tokens / ttft) is a floor, not the engine's internal prefill rate.
    The OpenAI-compatible API does not expose engine-internal prefill timings.
  * Nothing is loaded, unloaded, restarted or deleted. Pure inference load
    against whatever is already serving.

Placement (node, GPU, engine image, vLLM flags, KV dtype) is read from the
AINode API when ``--ainode`` is given, so a result file says what it ran on
instead of trusting the operator's memory. Output is one JSON per run in
bench/results/ per bench/SCHEMA.md. stdlib only.

# ponytail: stdlib urllib + threads, same as scripts/bench-serve.py; a lab bench
# should run on a bare python3 with no pip step.
"""
import argparse
import json
import pathlib
import re
import statistics
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "bench" / "results"
SCHEMA = 1

# Inference requests: a 120k-token prefill on a bandwidth-bound node legitimately
# takes minutes, so this is deliberately generous. Control-plane reads are quick
# or they are broken.
REQ_TIMEOUT = 1800
CTL_TIMEOUT = 15

# Same filler text as scripts/bench-serve.py on purpose: depth numbers from the
# two scripts stay comparable because the tokenised content is identical.
FILLER = (
    "The memory bandwidth of a device sets a hard ceiling on single-stream decode, "
    "because every generated token requires reading the active weights out of memory. "
    "Quantization shrinks that read. Tensor parallelism splits it across nodes and "
    "charges for it in interconnect traffic. Speculative decoding is the only lever "
    "that changes the equation itself, by producing more than one token per pass. "
)

SHORT_TASK = ("Explain in about 150 words why memory bandwidth, not compute, sets the "
              "ceiling on single-stream token generation. Be specific and do not stop early.")

REASONING_Q = ("A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph on the "
               "same track. When does the second catch the first?")


# ---------------------------------------------------------------- transport

def get_json(url, timeout=CTL_TIMEOUT):
    """GET JSON. Returns {"_error": ...} instead of raising: a missing control
    endpoint must degrade a placement field, never kill a benchmark run."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:140]}"}


def stream_chat(url, model, prompt, max_tokens, thinking=None):
    """Stream one chat completion and time it.

    ``thinking`` None leaves the chat template's own default alone; True/False
    sends ``chat_template_kwargs.enable_thinking`` explicitly. Returns a dict
    with ok/error plus wall_s, ttft_s, decode_tok_s, gen_tokens, prompt_tokens.
    """
    payload = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # vLLM emits a final chunk with `choices: []` carrying usage. That empty
        # choices list is why the parse loop below cannot assume [0].
        "stream_options": {"include_usage": True},
    }
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    prompt_tokens = None
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                    completion_tokens = usage.get("completion_tokens") or completion_tokens
                choices = chunk.get("choices") or []
                if not choices:
                    continue                      # usage-only chunk
                d = choices[0].get("delta") or {}
                # A reasoning parser splits output into reasoning + content. Both
                # are generated tokens and both cost decode time, so counting only
                # `content` leaves ttft unset on a reply that is all thinking. The
                # key name varies by build: vLLM 0.27.1 emits `reasoning`, other
                # versions use `reasoning_content`.
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    chunks += 1
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:140]}",
                "wall_s": round(time.monotonic() - t0, 3)}
    wall = time.monotonic() - t0
    # Token count from the server's usage block, NOT from chunks: under spec
    # decoding a chunk can carry several accepted tokens at once.
    gen = completion_tokens
    if not gen:
        gen = chunks or None
    dec = None
    if ttft is not None and gen and gen > 1 and wall > ttft:
        dec = (gen - 1) / (wall - ttft)
    return {"ok": True, "wall_s": round(wall, 3),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "decode_tok_s": round(dec, 2) if dec else None,
            "gen_tokens": gen, "prompt_tokens": prompt_tokens,
            "usage_reported": completion_tokens is not None, "chunks": chunks}


# ---------------------------------------------------------------- prompts

def nonce():
    return f"[run {uuid.uuid4().hex}] "


def calibrate_cpt(url, model):
    """Measure this model's chars-per-token instead of assuming ~4.

    One cheap probe (max_tokens=1; we only want the usage block). Only used to
    SIZE a prompt: the reported x-axis is always the server's own prompt_tokens,
    so a bad calibration costs accuracy of the requested depth, never the
    truthfulness of the recorded depth.
    """
    probe = nonce() + FILLER * 40
    r = stream_chat(url, model, probe, 1)
    if r.get("ok") and r.get("prompt_tokens"):
        return len(probe) / r["prompt_tokens"]
    return 4.0


def depth_prompt(target_tokens, cpt):
    """A prompt of roughly target_tokens, unique nonce FIRST, asking for a long
    answer so the decode rate measured at this depth is not 7 tokens of noise."""
    head = nonce()
    reps = max(1, int((target_tokens * cpt - len(head)) / len(FILLER)))
    return head + (FILLER * reps) + (
        "\n\nWrite a detailed 400-word explanation of the tradeoffs described above. "
        "Be thorough and do not stop early.")


# ---------------------------------------------------------------- telemetry

class Telemetry:
    """Sample the serving node's GPU while the bench runs.

    A tok/s number without the load context is not evidence: 40 tok/s at 38 C on
    a third of the memory says something different from 40 tok/s at 41 C with the
    node full. Peaks are reported, because the peak is what the node had to
    survive. Reads AINode's /api/nodes, which is the only source here that works:
    /api/metrics and /api/metrics/gpu return {"error": "Unknown Error"} on GB10
    nodes where pynvml cannot read a unified-memory GPU.
    """

    INTERVAL = 2.0

    def __init__(self, ainode, node_id):
        self.ainode = (ainode or "").rstrip("/")
        self.node_id = node_id
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _read(self):
        d = get_json(f"{self.ainode}/api/nodes", timeout=8)
        for n in (d.get("nodes") or []):
            if n.get("node_id") == self.node_id:
                total = n.get("gpu_memory_gb")
                pct = n.get("gpu_memory_used_pct")
                used = round(total * pct / 100.0, 1) if (total and pct is not None) else None
                return {"gpu_util_pct": n.get("gpu_utilization"),
                        "temp_c": n.get("gpu_temp"),
                        "gpu_mem_used_gb": used,
                        "gpu_mem_total_gb": round(total, 1) if total else None}
        return None

    def _loop(self):
        while not self._stop.is_set():
            s = self._read()
            if s:
                self.samples.append(s)
            self._stop.wait(self.INTERVAL)

    def start(self):
        if not self.ainode or not self.node_id:
            return self
        s = self._read()
        if s:
            self.samples.append(s)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def result(self):
        """Peak of each series, plus the idle reading we opened with."""
        if not self.samples:
            return None
        def peak(key):
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            return max(vals) if vals else None
        out = {"gpu_mem_used_gb": peak("gpu_mem_used_gb"),
               "gpu_mem_total_gb": peak("gpu_mem_total_gb"),
               "gpu_util_pct": peak("gpu_util_pct"),
               "temp_c": peak("temp_c"),
               "samples": len(self.samples),
               "reading": "peak observed over the run; sampled from /api/nodes"}
        first = self.samples[0]
        if first.get("gpu_util_pct") is not None:
            out["gpu_util_pct_at_start"] = first["gpu_util_pct"]
        if first.get("temp_c") is not None:
            out["temp_c_at_start"] = first["temp_c"]
        return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------- discovery

def strip_paren(name):
    return re.sub(r"\s*\([^)]*\)\s*$", "", name or "").strip()


def parse_quant(model_id):
    for q in ("NVFP4", "FP8", "MXFP4", "AWQ", "GPTQ", "INT4", "INT8", "BF16"):
        if q.lower() in (model_id or "").lower():
            return q
    return None


def describe(ainode, engine_url, model):
    """Build the model + placement blocks from what the fleet actually reports.

    Every field is a read, never a guess. A field the API does not expose is
    omitted so bench/SCHEMA.md's "never fill a missing measurement with an
    estimate" holds for placement too. Returns (model_block, placement, node_id,
    warnings).
    """
    warn = []
    mb = {"id": model}
    pl = {"engine": "vllm"}
    node_id = None

    # --- engine itself: served context window (authoritative, it is serving)
    ml = get_json(engine_url.rstrip("/") + "/v1/models", timeout=CTL_TIMEOUT)
    for entry in (ml.get("data") or []):
        if entry.get("id") == model and entry.get("max_model_len"):
            pl["max_model_len"] = entry["max_model_len"]
    if ml.get("_error"):
        warn.append(f"engine /v1/models unreadable: {ml['_error']}")

    if not ainode:
        return mb, pl, node_id, warn
    base = ainode.rstrip("/")

    # --- the node we are hitting
    st = get_json(f"{base}/api/status")
    if st.get("_error"):
        warn.append(f"/api/status unreadable: {st['_error']}")
    else:
        node_id = st.get("node_id")
        if st.get("node_name"):
            pl["node"] = st["node_name"]
        if (st.get("gpu") or {}).get("name"):
            pl["gpu"] = st["gpu"]["name"]
        if st.get("version"):
            pl["ainode"] = st["version"]
        serves = [st.get("model")] + list(st.get("models_loaded") or [])
        if model not in serves:
            warn.append(f"{pl.get('node', 'this AINode')} does not report serving {model}; "
                        "placement and telemetry may describe the wrong node")

    # --- tensor parallelism and what else shares the node
    ss = get_json(f"{base}/api/server/status")
    stacked = []
    for m in (ss.get("loaded_models") or []):
        if m.get("id") == model:
            if m.get("parallel"):
                pl["tp"] = m["parallel"]
                pl["gpus"] = m["parallel"]
        elif node_id and m.get("node_id") == node_id:
            stacked.append(m["id"])
    pl["stacked_with"] = stacked
    pl.setdefault("tp", 1)
    pl.setdefault("gpus", 1)

    # --- launch flags. /api/config is the LIVE config of this node's primary
    # instance, but it is a shared mutable object that keeps the last load's
    # overrides, so it is only trustworthy when its own `model` is the model we
    # are benching. Otherwise fall back to the curated catalog recipe and say so.
    cfg = get_json(f"{base}/api/config")
    if not cfg.get("_error") and cfg.get("model") == model:
        if cfg.get("engine_image"):
            pl["engine_image"] = cfg["engine_image"]
        if cfg.get("extra_vllm_args"):
            pl["flags"] = list(cfg["extra_vllm_args"])
        if cfg.get("gpu_memory_utilization") is not None:
            pl["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
        if cfg.get("kv_cache_dtype"):
            pl["kv_cache_dtype"] = cfg["kv_cache_dtype"]
        if cfg.get("max_model_len"):
            pl["max_model_len"] = cfg["max_model_len"]
        if cfg.get("distributed_mode"):
            pl["distributed_mode"] = cfg["distributed_mode"]
        pl["flags_source"] = "live node config (/api/config)"
        if cfg.get("extra_env"):
            pl["extra_env"] = dict(cfg["extra_env"])

    # --- model metadata from the catalog, matched on hf_repo or catalog id
    cat = get_json(f"{base}/api/models", timeout=30)
    info = None
    for m in (cat.get("models") or []):
        if model in (m.get("hf_repo"), m.get("id")):
            info = m
            break
    if info:
        if info.get("name"):
            mb["name"] = strip_paren(info["name"])
        if info.get("params_b"):
            mb["params_b"] = info["params_b"]
        if info.get("quantization"):
            mb["quant"] = info["quantization"]
        if info.get("context_length"):
            mb["context"] = info["context_length"]
        if info.get("license"):
            mb["license"] = info["license"]
        mb["vision"] = "vision" in (info.get("capabilities") or [])
        if "flags" not in pl:
            if info.get("engine_image"):
                pl["engine_image"] = info["engine_image"]
            if info.get("extra_vllm_args"):
                pl["flags"] = list(info["extra_vllm_args"])
                pl["flags_source"] = ("curated catalog recipe (/api/models); the live "
                                      "container command line was not readable")
    else:
        warn.append(f"{model} is not in the AINode catalog; fill model metadata "
                    "(params_b, license, context) by hand")

    # Active params and architecture come off the model id, which states them:
    # the "A3B" in 30B-A3B is the vendor's own active-parameter count.
    mm = re.search(r"(?:^|[-_])A(\d+(?:\.\d+)?)B(?:[-_]|$)", model, re.I)
    if mm:
        mb["arch"] = "moe"
        mb["active_b"] = float(mm.group(1)) if "." in mm.group(1) else int(mm.group(1))
    elif mb.get("params_b"):
        mb["arch"] = "dense"
        mb["active_b"] = mb["params_b"]
    mb.setdefault("quant", parse_quant(model))
    if not mb.get("quant"):
        mb.pop("quant", None)
    return mb, pl, node_id, warn


# ---------------------------------------------------------------- sections

def sec_single(a, cpt):
    """One user, short prompt. The number a chat window feels."""
    print(f"\n  SINGLE STREAM  (short prompt, {a.max_tokens} tokens)")
    r = stream_chat(a.url, a.model, nonce() + SHORT_TASK, a.max_tokens,
                    thinking=False if a.no_think else None)
    if not r.get("ok") or not r.get("decode_tok_s"):
        print(f"    FAILED  {r.get('error') or 'no usable tokens; raise --max-tokens'}")
        return None
    print(f"    TTFT {r['ttft_s'] * 1000:.0f} ms   decode {r['decode_tok_s']:.1f} tok/s "
          f"({r['gen_tokens']} tok, prompt {r['prompt_tokens']})")
    out = {"decode_tok_s": r["decode_tok_s"], "gen_tokens": r["gen_tokens"]}
    if r.get("ttft_s") is not None:
        out["ttft_ms"] = round(r["ttft_s"] * 1000)
    if r.get("prompt_tokens"):
        out["prompt_tokens"] = r["prompt_tokens"]
    return out


def sec_prefill(a, cpt):
    """What long context costs. Unique nonce per request, so no cache freebies."""
    print(f"\n  PREFILL SCALING  (decode and TTFT against prompt length, "
          f"{cpt:.2f} chars/token)")
    print(f"    {'prompt tok':>11} {'TTFT':>10} {'prefill tok/s':>14} {'decode tok/s':>13}")
    rows = []
    for d in a.depths:
        r = stream_chat(a.url, a.model, depth_prompt(d, cpt), a.max_tokens,
                        thinking=False if a.no_think else None)
        if not r.get("ok"):
            print(f"    {d:>11} FAILED  {r.get('error')}")
            continue
        if r.get("ttft_s") is None:
            print(f"    {d:>11} no content deltas; raise --max-tokens or use --no-think")
            continue
        ptok = r.get("prompt_tokens")
        row = {"ttft_ms": round(r["ttft_s"] * 1000)}
        if ptok:
            row["prompt_tokens"] = ptok
            # A floor, not the engine's internal prefill rate: TTFT carries
            # queueing and scheduling as well as the forward pass.
            row["prefill_tok_s"] = round(ptok / r["ttft_s"], 1)
        if r.get("decode_tok_s"):
            row["decode_tok_s"] = r["decode_tok_s"]
        rows.append(row)
        print(f"    {ptok if ptok else '?':>11} {r['ttft_s'] * 1000:>8.0f}ms "
              f"{row.get('prefill_tok_s', 0):>14.1f} {row.get('decode_tok_s', 0):>13.1f}")
    if len(rows) > 1 and rows[0].get("decode_tok_s") and rows[-1].get("decode_tok_s"):
        # Can legitimately come out negative: on a spec-decode engine the
        # acceptance rate moves with content, and that swing can exceed the
        # depth penalty over a short sweep. Say which way it went.
        d = (1 - rows[-1]["decode_tok_s"] / rows[0]["decode_tok_s"]) * 100
        verb = "falls" if d >= 0 else "rises"
        print(f"    -> decode {verb} {abs(d):.0f}% from {rows[0].get('prompt_tokens')} to "
              f"{rows[-1].get('prompt_tokens')} prompt tokens")
    return rows or None


def sec_sustained(a, cpt):
    """One unbroken long generation: does the rate hold as KV grows?"""
    print(f"\n  SUSTAINED GENERATION  ({a.sustained_tokens} tokens, one request)")
    prompt = nonce() + (
        "Write a detailed technical explanation of how speculative decoding works in "
        "large language model inference. Cover the draft model, verification, acceptance "
        "rates, and why throughput varies with content. Do not stop early.")
    r = stream_chat(a.url, a.model, prompt, a.sustained_tokens,
                    thinking=False if a.no_think else None)
    if not r.get("ok") or not r.get("decode_tok_s"):
        print(f"    FAILED  {r.get('error') or 'no usable tokens'}")
        return None
    print(f"    {r['gen_tokens']} tokens in {r['wall_s']:.1f}s at {r['decode_tok_s']:.1f} tok/s")
    out = {"decode_tok_s": r["decode_tok_s"], "gen_tokens": r["gen_tokens"],
           "wall_s": r["wall_s"]}
    if r.get("ttft_s") is not None:
        out["ttft_ms"] = round(r["ttft_s"] * 1000)
    return out


def sec_concurrency(a, cpt):
    """Aggregate throughput as more people use the node at once."""
    print(f"\n  CONCURRENCY  ({a.max_tokens} tokens per stream)")
    print(f"    {'streams':>7} {'aggregate':>12} {'per-stream':>12} {'med TTFT':>10} "
          f"{'wall':>7}  ok")
    rows = []
    for n in a.streams:
        prompts = [nonce() + f"Question {i + 1}. " + SHORT_TASK for i in range(n)]
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(
                lambda p: stream_chat(a.url, a.model, p, a.max_tokens,
                                      thinking=False if a.no_think else None),
                prompts))
        wall = time.monotonic() - t0
        ok = [r for r in res if r.get("ok") and r.get("gen_tokens")]
        if not ok:
            err = next((r.get("error") for r in res if r.get("error")), "no tokens")
            print(f"    {n:>7} all failed  {err}")
            continue
        agg = sum(r["gen_tokens"] for r in ok) / wall
        decs = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        row = {"streams": n, "aggregate_tok_s": round(agg, 1), "ok": len(ok),
               "failed": len(res) - len(ok), "wall_s": round(wall, 2)}
        if decs:
            row["per_stream_tok_s"] = round(statistics.median(decs), 1)
        if ttfts:
            row["median_ttft_ms"] = round(statistics.median(ttfts) * 1000)
        rows.append(row)
        print(f"    {n:>7} {agg:>10.1f}/s {row.get('per_stream_tok_s', 0):>10.1f}/s "
              f"{row.get('median_ttft_ms', 0):>8} ms {wall:>6.1f}s  {len(ok)}/{n}")
    if len(rows) > 1 and rows[0]["aggregate_tok_s"]:
        print(f"    -> {rows[-1]['streams']} streams is "
              f"{rows[-1]['aggregate_tok_s'] / rows[0]['aggregate_tok_s']:.1f}x the "
              f"aggregate of {rows[0]['streams']}")
    return rows or None


def sec_reasoning(a, cpt):
    """The reasoning tax. Always measures BOTH states regardless of --no-think:
    the comparison is the whole point of the section."""
    print(f"\n  REASONING TAX  (same prompt, thinking off vs on, "
          f"{a.reasoning_tokens} token cap)")
    out = {}
    got = {}
    for name, flag in (("off", False), ("on", True)):
        r = stream_chat(a.url, a.model, nonce() + REASONING_Q, a.reasoning_tokens,
                        thinking=flag)
        if not r.get("ok"):
            print(f"    thinking {name:<3} FAILED  {r.get('error')}")
            continue
        got[name] = r
        print(f"    thinking {name:<3} {r['gen_tokens'] or 0:>5} tok  {r['wall_s']:>6.2f}s  "
              f"{r.get('decode_tok_s') or 0:>6.1f} tok/s")
    for name in ("off", "on"):
        if name in got:
            out[f"thinking_{name}_wall_s"] = got[name]["wall_s"]
            out[f"thinking_{name}_tokens"] = got[name]["gen_tokens"]
    if "off" in got and "on" in got and got["off"]["wall_s"]:
        ratio = got["on"]["wall_s"] / got["off"]["wall_s"]
        out["tax_x"] = round(ratio, 2)
        print(f"    -> reasoning costs {ratio:.1f}x the wall time of the same answer")
    return out or None


SECTIONS = {"single": ("single_stream", sec_single),
            "prefill": ("prefill", sec_prefill),
            "sustained": ("sustained", sec_sustained),
            "concurrency": ("concurrency", sec_concurrency),
            "reasoning": ("reasoning_tax", sec_reasoning)}


# ---------------------------------------------------------------- main

def slug(model_id):
    """bench/SCHEMA.md filenames: the part after the org, lowercased, dots to
    underscores so a version number stays readable as one token."""
    base = (model_id or "model").rsplit("/", 1)[-1].lower().replace(".", "_")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9_-]", "-", base)).strip("-")


def int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(prog="ainode-bench", description=__doc__.split("\n")[0])
    p.add_argument("--url", help="engine or AINode proxy base, e.g. http://host:8000")
    p.add_argument("--model", help="model id exactly as served")
    p.add_argument("--ainode", default="", help="AINode web base, e.g. http://host:3000 "
                                                "(telemetry + placement)")
    p.add_argument("--label", help="free text: what makes this run distinct")
    p.add_argument("--only", default="", help="comma list of "
                                             + ",".join(SECTIONS))
    p.add_argument("--depths", default="4000,16000,32000,64000,120000")
    p.add_argument("--streams", default="1,2,4,8,16")
    p.add_argument("--no-think", action="store_true",
                   help="enable_thinking=false for every section except reasoning, "
                        "which always measures both states")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="generation cap for single/prefill/concurrency")
    p.add_argument("--sustained-tokens", type=int, default=1500)
    p.add_argument("--reasoning-tokens", type=int, default=600)
    p.add_argument("--show", help="pretty-print a saved result and exit")
    a = p.parse_args()

    if a.show:
        print(json.dumps(json.loads(pathlib.Path(a.show).read_text()), indent=2))
        return 0
    for need in ("url", "model", "label"):
        if not getattr(a, need):
            p.error(f"--{need} is required")
    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(SECTIONS)
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        p.error(f"unknown section(s) {', '.join(bad)}; pick from {', '.join(SECTIONS)}")
    a.depths = int_list(a.depths)
    a.streams = int_list(a.streams)

    mb, pl, node_id, warn = describe(a.ainode, a.url, a.model)
    print(f"\n  ainode-bench  {a.model}")
    print(f"  label   : {a.label}")
    print(f"  endpoint: {a.url}")
    print(f"  node    : {pl.get('node', 'unknown')}  {pl.get('gpu', '')}  "
          f"tp={pl.get('tp', '?')}  ainode {pl.get('ainode', '?')}")
    print(f"  engine  : {pl.get('engine_image', 'unknown image')}  "
          f"kv={pl.get('kv_cache_dtype', '?')}  gmu={pl.get('gpu_memory_utilization', '?')}")
    if pl.get("stacked_with"):
        print(f"  stacked : {', '.join(pl['stacked_with'])}")
    for w in warn:
        print(f"  warn    : {w}")

    tel = Telemetry(a.ainode, node_id).start()
    cpt = calibrate_cpt(a.url, a.model)
    t_start = time.time()
    results = {}
    for name in want:
        key, fn = SECTIONS[name]
        try:
            val = fn(a, cpt)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    section {name} raised {type(e).__name__}: {e}")
            val = None
        if val:
            results[key] = val
    tel.stop()
    telemetry = tel.result()
    if telemetry:
        results["telemetry"] = telemetry
        print(f"\n  TELEMETRY  peak GPU {telemetry.get('gpu_util_pct')}%  "
              f"mem {telemetry.get('gpu_mem_used_gb')}/{telemetry.get('gpu_mem_total_gb')} GB  "
              f"{telemetry.get('temp_c')} C  ({telemetry.get('samples')} samples)")

    notes = [f"Measured by scripts/ainode-bench.py in {round(time.time() - t_start)}s; "
             f"nothing loaded, unloaded or restarted."]
    if "prefill" in results:
        notes.append("prefill_tok_s is prompt_tokens/TTFT, a floor: TTFT includes "
                     "queueing, and the OpenAI API exposes no internal prefill timing.")
    if a.no_think:
        notes.append("Sections other than reasoning_tax ran with enable_thinking=false.")
    rt = results.get("reasoning_tax") or {}
    if rt.get("thinking_on_tokens") == a.reasoning_tokens:
        notes.append(f"Thinking-on hit the {a.reasoning_tokens}-token cap, so the "
                     "reasoning tax is a floor: the answer had not finished.")
    # 0% is what the API returns for a GB10, not what the GPU was doing. Saying so
    # here is the difference between a caveat and a wrong number on the page.
    if telemetry and telemetry.get("gpu_util_pct") == 0:
        notes.append("GPU utilisation read 0% on every sample: AINode reports no "
                     "utilisation for a GB10 unified-memory GPU, so read it as unread "
                     "rather than idle. Memory and temperature are real reads.")
    if pl.get("flags_source", "").startswith("curated"):
        notes.append("Flags are the catalog recipe, not a read of the live container.")
    notes.extend(warn)

    settings = {"thinking": False if a.no_think else None,
                "max_tokens": a.max_tokens, "temperature": None,
                "sustained_tokens": a.sustained_tokens,
                "reasoning_tokens": a.reasoning_tokens,
                "depths_requested": a.depths, "streams_requested": a.streams,
                "chars_per_token_probe": round(cpt, 2)}
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rec = {"schema": SCHEMA, "stamp": stamp, "label": a.label, "model": mb,
           "placement": pl, "settings": settings, "results": results,
           "notes": notes, "source": "scripts/ainode-bench.py"}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{stamp}-{slug(a.model)}-{a.label}.json"
    f.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\n  saved {f}")
    print("  render with: python3 bench/report.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  interrupted; nothing was written")
        sys.exit(130)
