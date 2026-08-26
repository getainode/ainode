#!/usr/bin/env python3
"""Baseline serve benchmark for the AINode OpenAI-compatible endpoint.

Measures TTFT + decode tok/s single-stream at several prompt sizes, then a
concurrency sweep (the metric that actually matters for batched/agent workloads).
Non-destructive — pure inference load. stdlib only.

  python3 scripts/bench-serve.py --url http://100.122.26.9:3000 \
      --model nvidia/Qwen3-235B-A22B-NVFP4

Depth mode answers the question the peak number can't: what does decode actually
do as the KV cache fills? Decode reads the occupied KV every token, so tok/s at
4k and tok/s at 128k are different numbers and get quoted interchangeably.

  python3 scripts/bench-serve.py --url http://100.122.26.9:3000 \
      --model unsloth/Qwen3.8-27B-NVFP4 --mode depth \
      --depths 4000,16000,32000,64000,128000 --reps 3

Two things keep depth numbers honest. Each request carries a unique nonce at the
FRONT of the prompt so `--enable-prefix-caching` can't serve a cached prefill and
make depth look free. And the x-axis is the server's own `usage.prompt_tokens`,
never our estimate, so a bad chars-per-token guess shifts nothing.

# ponytail: stdlib urllib + threads; no httpx/asyncio dep for a lab bench.
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

PROMPTS = {
    "short": "Say hello in one sentence.",
    "med_2k": "Summarize the following, then list 5 implications.\n" + ("lorem ipsum dolor sit amet " * 300),
    "long_8k": "Read this and answer: what is the main theme?\n" + ("the quick brown fox jumps over the lazy dog " * 1500),
}


def one_request(url, model, prompt, max_tokens, want_usage=False, no_think=False):
    """Stream a completion.

    Returns (ttft_s, decode_toks_per_s, n_tokens, ok, prompt_tokens). The 5th
    element is appended rather than inserted so existing positional callers
    (the concurrency sweep) keep working unchanged.
    """
    payload = {
        "model": model, "stream": True, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if want_usage:
        # vLLM emits a final chunk with `choices: []` and a usage block. That
        # empty choices list is why the parse loop below can't assume [0].
        payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    n = 0
    prompt_tokens = None
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                    completion_tokens = usage.get("completion_tokens") or completion_tokens
                choices = chunk.get("choices") or []
                if not choices:
                    continue          # usage-only chunk
                d_obj = choices[0].get("delta", {}) or {}
                # A reasoning parser splits output into reasoning_content and
                # content. Both are generated tokens and both cost decode time,
                # so counting only `content` undercounts throughput and can
                # leave ttft unset entirely on a reply that is all thinking.
                # Key name varies by build: vLLM 0.27.1 emits `reasoning`,
                # other versions/parsers use `reasoning_content`. Check all three.
                delta = (d_obj.get("content") or d_obj.get("reasoning")
                         or d_obj.get("reasoning_content"))
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    n += 1
    except Exception as e:
        return (None, None, 0, f"ERR {e}", None)
    total = time.monotonic() - t0
    # Token count comes from the server's usage block, NOT from counting SSE
    # chunks. Under speculative decoding a chunk can carry several accepted
    # tokens at once, so chunk-counting reports roughly rate/acceptance-length
    # and silently halves the number on an MTP or DFlash config.
    gen = completion_tokens if completion_tokens else n
    # Decode rate EXCLUDES prefill: the clock starts at the first token, not at
    # request send. Including TTFT is what turns a decode number into an
    # agent-loop number.
    decode = (gen - 1) / (total - ttft) if ttft and gen > 1 and total > ttft else 0.0
    return (ttft, decode, gen, "ok", prompt_tokens)


# ---------------------------------------------------------------- depth sweep

FILLER = (
    "The memory bandwidth of a device sets a hard ceiling on single-stream decode, "
    "because every generated token requires reading the active weights out of memory. "
    "Quantization shrinks that read. Tensor parallelism splits it across nodes and "
    "charges for it in interconnect traffic. Speculative decoding is the only lever "
    "that changes the equation itself, by producing more than one token per pass. "
)


def calibrate_chars_per_token(url, model):
    """Measure this model's chars-per-token on FILLER instead of assuming ~4.

    One cheap probe. If it fails we fall back to 4.0, and it costs us nothing
    either way because the reported x-axis is the server's own prompt_tokens.
    """
    probe = FILLER * 40
    r = one_request(url, model, probe, 1, want_usage=True)  # 1 tok: we only want usage
    if r[3] == "ok" and r[4]:
        return len(probe) / r[4]
    return 4.0


def make_prompt(target_tokens, cpt):
    """Prompt of roughly target_tokens, with a unique nonce FIRST.

    The nonce leads so it invalidates the shared prefix. A trailing nonce would
    still let prefix caching serve almost the entire prefill and we would be
    timing a cache hit while calling it a depth measurement.
    """
    nonce = f"[run {uuid.uuid4().hex}] "
    reps = max(1, int((target_tokens * cpt - len(nonce)) / len(FILLER)))
    # Ask for a long answer on purpose: a 7-token reply makes the decode rate
    # noise. We want enough generated tokens for the rate to mean something.
    return nonce + (FILLER * reps) + (
        "\n\nWrite a detailed 400-word explanation of the tradeoffs described above. "
        "Be thorough and do not stop early.")


def depth_sweep(url, model, depths, reps, max_tokens, no_think=False):
    cpt = calibrate_chars_per_token(url, model)
    print(f"== decode vs context depth ({max_tokens} tok gen, {reps} rep(s), {cpt:.2f} chars/token) ==")
    print(f"  {'target':>8}  {'actual':>8}  {'TTFT':>9}  {'decode':>12}  {'gen':>5}")
    rows = []
    for d in depths:
        prompt = None
        decodes, ttfts, actuals, ntoks = [], [], [], []
        for _ in range(reps):
            prompt = make_prompt(d, cpt)   # new nonce each rep
            ttft, dec, n, ok, ptok = one_request(url, model, prompt, max_tokens,
                                                 want_usage=True, no_think=no_think)
            if ok != "ok":
                print(f"  {d:>8}  {ok}")
                break
            if ttft is None or n < 2:
                print(f"  {d:>8}  no usable tokens (n={n}); raise --max-tokens or use --no-think")
                break
            decodes.append(dec)
            ttfts.append(ttft)
            ntoks.append(n)
            if ptok:
                actuals.append(ptok)
        if not decodes:
            continue
        med_dec = statistics.median(decodes)
        med_ttft = statistics.median(ttfts)
        actual = int(statistics.median(actuals)) if actuals else 0
        rows.append((d, actual, med_ttft, med_dec))
        spread = f" (n={len(decodes)}, {min(decodes):.1f}-{max(decodes):.1f})" if len(decodes) > 1 else ""
        print(f"  {d:>8}  {actual:>8}  {med_ttft*1000:>7.0f}ms  {med_dec:>8.1f} t/s  {int(statistics.median(ntoks)):>5}{spread}")
    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        drop = (1 - last[3] / first[3]) * 100 if first[3] else 0
        print(f"\n  decode falls {drop:.0f}% from {first[1]} to {last[1]} prompt tokens "
              f"({first[3]:.1f} -> {last[3]:.1f} tok/s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--mode", choices=["legacy", "depth", "all"], default="legacy",
                    help="legacy = original single-stream + concurrency sections (default)")
    ap.add_argument("--depths", default="4000,16000,32000,64000,128000",
                    help="prompt sizes in tokens for --mode depth/all")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per depth; median is reported")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false for stable generation length")
    args = ap.parse_args()

    if args.mode in ("depth", "all"):
        depth_sweep(args.url, args.model, [int(x) for x in args.depths.split(",")],
                    args.reps, args.max_tokens, no_think=args.no_think)
        if args.mode == "depth":
            return
        print()

    print(f"== single-stream ({args.max_tokens} tok) ==")
    for name, p in PROMPTS.items():
        ttft, dec, n, ok, _ = one_request(args.url, args.model, p, args.max_tokens)
        if ok != "ok":
            print(f"  {name:8} {ok}")
        else:
            print(f"  {name:8} TTFT {ttft*1000:6.0f}ms  decode {dec:5.1f} tok/s  ({n} tok)")

    print("\n== concurrency sweep (short prompt, aggregate throughput) ==")
    for c in [int(x) for x in args.concurrency.split(",")]:
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda _: one_request(args.url, args.model, PROMPTS["short"], args.max_tokens), range(c)))
        wall = time.monotonic() - t0
        ok = [r for r in res if r[3] == "ok"]
        toks = sum(r[2] for r in ok)
        ttfts = [r[0] for r in ok if r[0]]
        agg = toks / wall if wall else 0
        med_ttft = statistics.median(ttfts) * 1000 if ttfts else 0
        print(f"  c={c:2}  {len(ok)}/{c} ok  agg {agg:6.1f} tok/s  median TTFT {med_ttft:6.0f}ms  wall {wall:4.1f}s")


if __name__ == "__main__":
    main()
