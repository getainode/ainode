"""Measurement core for the AINode bench - the part that actually times things.

This is the code that used to live inside ``scripts/ainode-bench.py``. It moved
here so the product can run the same benchmark from the browser and get numbers
that are comparable with the ones the CLI has been writing into
``bench/results/`` all along. The CLI is now a thin shim over this module, which
is the only way to keep "the web run and the terminal run measure the same
thing" true rather than aspirational.

Deliberately **stdlib only**, and deliberately blocking. Two reasons:

  * ``scripts/ainode-bench.py`` is documented as runnable on a bare ``python3``
    with no pip step, and it imports this module.
  * The numbers have to stay comparable with every result file already in
    ``bench/results/``. Swapping urllib for aiohttp would change the transport
    under the clock for no measurement gain.

Async callers do not block their event loop on it: :mod:`ainode.bench.runner`
drives every section through ``asyncio.to_thread``.

Honesty rules, all load-bearing, all inherited from the script:

  * Prompt token counts come from the server's ``usage.prompt_tokens`` via
    ``stream_options.include_usage``, never from a chars-per-token estimate.
  * Generated token counts come from ``usage.completion_tokens``, never from
    counting SSE chunks: under speculative decoding one chunk can carry several
    accepted tokens, so chunk-counting silently halves the rate.
  * Every prompt carries a unique nonce at the FRONT, so prefix caching cannot
    serve a cached prefill and make depth look free.
  * Decode rate EXCLUDES prefill: the clock starts at the first content delta.
  * ``prefill_tok_s`` is ``prompt_tokens / ttft``, a floor: TTFT carries
    queueing, and the OpenAI-compatible API exposes no internal prefill timing.
  * Nothing is loaded, unloaded, restarted or deleted. Pure inference load
    against whatever is already serving.
  * A request the node REFUSED is not a measurement. Both transports below hand a
    401 or a 429 to ``ainode.bench.auth``, which raises out of the run rather than
    letting the refusal land as a section of failures (see that module).
"""
from __future__ import annotations

import json
import re
import statistics
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ainode.bench import auth

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


# ---------------------------------------------------------------- options

@dataclass
class BenchOptions:
    """One run's inputs. Attribute names match the old argparse namespace so the
    section bodies below are the same code they were in the script."""

    url: str
    model: str
    label: str = "run"
    sections: list = field(default_factory=lambda: list(SECTION_ORDER))
    depths: list = field(default_factory=lambda: [4000, 16000, 32000, 64000, 120000])
    streams: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    no_think: bool = False
    max_tokens: int = 200
    sustained_tokens: int = 1500
    reasoning_tokens: int = 600
    #: Bearer token for a protected node. Empty for the in-process runner, which
    #: talks to an engine container that never sees AINode's middleware, and read
    #: from ``--api-key`` or ``$AINODE_API_KEY`` by the CLI. Never printed and
    #: never written into a record: ``build_record`` below names every setting it
    #: emits, and this is not one of them.
    api_key: str = ""


# ---------------------------------------------------------------- reporting

class Reporter:
    """Where a run's human-readable progress goes.

    The console reporter prints it (CLI); the job reporter appends to a deque and
    moves a step counter the web UI polls. Sections call ``cancelled()`` between
    requests, and ``stream_chat`` calls it while reading the SSE body, so a
    cancel from the UI lands inside a long generation instead of after it.
    """

    def section(self, key: str, title: str) -> None:
        self.log(title)

    def log(self, msg: str) -> None:
        pass

    def step(self, done: int, total: int, label: str = "") -> None:
        pass

    def cancelled(self) -> bool:
        return False


class ConsoleReporter(Reporter):
    """Prints the same text the script printed, blank line before each section."""

    def section(self, key: str, title: str) -> None:
        print("\n" + title)

    def log(self, msg: str) -> None:
        print(msg)


class Cancelled(Exception):
    """Raised out of a section when the reporter says the run was cancelled."""


# ---------------------------------------------------------------- transport

def get_json(url, timeout=CTL_TIMEOUT, api_key=""):
    """GET JSON. Returns {"_error": ...} instead of raising: a missing control
    endpoint must degrade a placement field, never kill a benchmark run.

    That is also why a 401 here does NOT raise the way one in ``stream_chat`` does:
    a control-plane read the node refused costs the run a placement field, not a
    measurement. The caller spells the refusal out with ``auth.explain`` so the
    warning says which key is missing rather than quoting urllib.
    """
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   **auth.bearer(api_key)})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:140]}"}


def stream_chat(url, model, prompt, max_tokens, thinking=None, should_stop=None,
                api_key=""):
    """Stream one chat completion and time it.

    ``thinking`` None leaves the chat template's own default alone; True/False
    sends ``chat_template_kwargs.enable_thinking`` and ``.thinking`` explicitly
    (the Qwen-family and DeepSeek V4 switch names). ``should_stop`` is
    an optional predicate checked while reading the stream, so a cancelled run
    stops inside a long generation rather than after it. Returns a dict with
    ok/error plus wall_s, ttft_s, decode_tok_s, gen_tokens, prompt_tokens.

    The one exception to "never raises" is a refusal: a 401 or a 429 leaves as
    :class:`ainode.bench.auth.EndpointRefused` rather than as an error dict, because
    a refused request is not a slow one and a section full of them is not a result.
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
        # Both switch names: Qwen and Nemotron templates read enable_thinking,
        # DeepSeek V4 reads thinking. A template ignores the one it does not use,
        # so sending both makes the reasoning section mean the same thing on
        # every model (before this, DeepSeek's "thinking on" never turned it on).
        payload["chat_template_kwargs"] = {
            "enable_thinking": bool(thinking),
            "thinking": bool(thinking),
        }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **auth.bearer(api_key)},
    )
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    prompt_tokens = None
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
            for raw in r:
                if should_stop is not None and should_stop():
                    return {"ok": False, "error": "cancelled", "cancelled": True,
                            "wall_s": round(time.monotonic() - t0, 3)}
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
        # A refusal raises out of the run; everything else is this request's row.
        auth.check_exception(e)
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


def calibrate_cpt(url, model, api_key=""):
    """Measure this model's chars-per-token instead of assuming ~4.

    One cheap probe (max_tokens=1; we only want the usage block). Only used to
    SIZE a prompt: the reported x-axis is always the server's own prompt_tokens,
    so a bad calibration costs accuracy of the requested depth, never the
    truthfulness of the recorded depth.
    """
    probe = nonce() + FILLER * 40
    r = stream_chat(url, model, probe, 1, api_key=api_key)
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
    survive.

    ``read`` is a callable returning one sample dict (or None). The CLI passes a
    reader that GETs the node API's ``/api/nodes``; the in-process runner passes
    one that reads the same fields straight off cluster state. Either way this is
    the source ``/api/nodes`` renders: ``/api/metrics`` and ``/api/metrics/gpu``
    return {"error": "Unknown Error"} on GB10 nodes where pynvml cannot read a
    unified-memory GPU.
    """

    INTERVAL = 2.0

    def __init__(self, read=None):
        self._read_fn = read
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _read(self):
        if self._read_fn is None:
            return None
        try:
            return self._read_fn()
        except Exception:
            # A sample that cannot be read is a missing sample, never a failed run.
            return None

    def _loop(self):
        while not self._stop.is_set():
            s = self._read()
            if s:
                self.samples.append(s)
            self._stop.wait(self.INTERVAL)

    def start(self):
        if self._read_fn is None:
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


def node_sample(node: dict):
    """One telemetry sample out of an ``/api/nodes`` row (either transport)."""
    total = node.get("gpu_memory_gb")
    pct = node.get("gpu_memory_used_pct")
    used = round(total * pct / 100.0, 1) if (total and pct is not None) else None
    return {"gpu_util_pct": node.get("gpu_utilization"),
            "temp_c": node.get("gpu_temp"),
            "gpu_mem_used_gb": used,
            "gpu_mem_total_gb": round(total, 1) if total else None}


def http_nodes_reader(ainode: str, node_id: str, api_key: str = ""):
    """Telemetry reader for an out-of-process caller: GET <ainode>/api/nodes."""
    base = (ainode or "").rstrip("/")
    if not base or not node_id:
        return None

    def read():
        d = get_json(f"{base}/api/nodes", timeout=8, api_key=api_key)
        for n in (d.get("nodes") or []):
            if n.get("node_id") == node_id:
                return node_sample(n)
        return None

    return read


# ---------------------------------------------------------------- sections

def _check(rep: Reporter):
    if rep.cancelled():
        raise Cancelled()


def sec_single(a, cpt, rep):
    """One user, short prompt. The number a chat window feels."""
    rep.section("single", f"  SINGLE STREAM  (short prompt, {a.max_tokens} tokens)")
    rep.step(0, 1, "short prompt")
    _check(rep)
    r = stream_chat(a.url, a.model, nonce() + SHORT_TASK, a.max_tokens,
                    thinking=False if a.no_think else None,
                    should_stop=rep.cancelled,
                    api_key=a.api_key)
    if r.get("cancelled"):
        raise Cancelled()
    if not r.get("ok") or not r.get("decode_tok_s"):
        rep.log(f"    FAILED  {r.get('error') or 'no usable tokens; raise max tokens'}")
        return None
    rep.step(1, 1, "short prompt")
    rep.log(f"    TTFT {r['ttft_s'] * 1000:.0f} ms   decode {r['decode_tok_s']:.1f} tok/s "
            f"({r['gen_tokens']} tok, prompt {r['prompt_tokens']})")
    out = {"decode_tok_s": r["decode_tok_s"], "gen_tokens": r["gen_tokens"]}
    if r.get("ttft_s") is not None:
        out["ttft_ms"] = round(r["ttft_s"] * 1000)
    if r.get("prompt_tokens"):
        out["prompt_tokens"] = r["prompt_tokens"]
    return out


def sec_prefill(a, cpt, rep):
    """What long context costs. Unique nonce per request, so no cache freebies."""
    rep.section("prefill", f"  PREFILL SCALING  (decode and TTFT against prompt length, "
                           f"{cpt:.2f} chars/token)")
    rep.log(f"    {'prompt tok':>11} {'TTFT':>10} {'prefill tok/s':>14} {'decode tok/s':>13}")
    rows = []
    for i, d in enumerate(a.depths):
        _check(rep)
        rep.step(i, len(a.depths), f"{d} prompt tokens")
        r = stream_chat(a.url, a.model, depth_prompt(d, cpt), a.max_tokens,
                        thinking=False if a.no_think else None,
                        should_stop=rep.cancelled, api_key=a.api_key)
        if r.get("cancelled"):
            raise Cancelled()
        if not r.get("ok"):
            rep.log(f"    {d:>11} FAILED  {r.get('error')}")
            continue
        if r.get("ttft_s") is None:
            rep.log(f"    {d:>11} no content deltas; raise max tokens or turn thinking off")
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
        rep.step(i + 1, len(a.depths), f"{d} prompt tokens")
        rep.log(f"    {ptok if ptok else '?':>11} {r['ttft_s'] * 1000:>8.0f}ms "
                f"{row.get('prefill_tok_s', 0):>14.1f} {row.get('decode_tok_s', 0):>13.1f}")
    if len(rows) > 1 and rows[0].get("decode_tok_s") and rows[-1].get("decode_tok_s"):
        # Can legitimately come out negative: on a spec-decode engine the
        # acceptance rate moves with content, and that swing can exceed the
        # depth penalty over a short sweep. Say which way it went.
        d = (1 - rows[-1]["decode_tok_s"] / rows[0]["decode_tok_s"]) * 100
        verb = "falls" if d >= 0 else "rises"
        rep.log(f"    -> decode {verb} {abs(d):.0f}% from {rows[0].get('prompt_tokens')} to "
                f"{rows[-1].get('prompt_tokens')} prompt tokens")
    return rows or None


def sec_sustained(a, cpt, rep):
    """One unbroken long generation: does the rate hold as KV grows?"""
    rep.section("sustained",
                f"  SUSTAINED GENERATION  ({a.sustained_tokens} tokens, one request)")
    rep.step(0, 1, f"{a.sustained_tokens} tokens")
    _check(rep)
    prompt = nonce() + (
        "Write a detailed technical explanation of how speculative decoding works in "
        "large language model inference. Cover the draft model, verification, acceptance "
        "rates, and why throughput varies with content. Do not stop early.")
    r = stream_chat(a.url, a.model, prompt, a.sustained_tokens,
                    thinking=False if a.no_think else None,
                    should_stop=rep.cancelled,
                    api_key=a.api_key)
    if r.get("cancelled"):
        raise Cancelled()
    if not r.get("ok") or not r.get("decode_tok_s"):
        rep.log(f"    FAILED  {r.get('error') or 'no usable tokens'}")
        return None
    rep.step(1, 1, f"{a.sustained_tokens} tokens")
    rep.log(f"    {r['gen_tokens']} tokens in {r['wall_s']:.1f}s at "
            f"{r['decode_tok_s']:.1f} tok/s")
    out = {"decode_tok_s": r["decode_tok_s"], "gen_tokens": r["gen_tokens"],
           "wall_s": r["wall_s"]}
    if r.get("ttft_s") is not None:
        out["ttft_ms"] = round(r["ttft_s"] * 1000)
    return out


def sec_concurrency(a, cpt, rep):
    """Aggregate throughput as more people use the node at once."""
    rep.section("concurrency", f"  CONCURRENCY  ({a.max_tokens} tokens per stream)")
    rep.log(f"    {'streams':>7} {'aggregate':>12} {'per-stream':>12} {'med TTFT':>10} "
            f"{'wall':>7}  ok")
    rows = []
    for i, n in enumerate(a.streams):
        _check(rep)
        rep.step(i, len(a.streams), f"{n} stream{'s' if n != 1 else ''}")
        prompts = [nonce() + f"Question {j + 1}. " + SHORT_TASK for j in range(n)]
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(
                lambda p: stream_chat(a.url, a.model, p, a.max_tokens,
                                      thinking=False if a.no_think else None,
                                      should_stop=rep.cancelled,
                                      api_key=a.api_key),
                prompts))
        wall = time.monotonic() - t0
        if any(r.get("cancelled") for r in res):
            raise Cancelled()
        ok = [r for r in res if r.get("ok") and r.get("gen_tokens")]
        if not ok:
            err = next((r.get("error") for r in res if r.get("error")), "no tokens")
            rep.log(f"    {n:>7} all failed  {err}")
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
        rep.step(i + 1, len(a.streams), f"{n} stream{'s' if n != 1 else ''}")
        rep.log(f"    {n:>7} {agg:>10.1f}/s {row.get('per_stream_tok_s', 0):>10.1f}/s "
                f"{row.get('median_ttft_ms', 0):>8} ms {wall:>6.1f}s  {len(ok)}/{n}")
    if len(rows) > 1 and rows[0]["aggregate_tok_s"]:
        rep.log(f"    -> {rows[-1]['streams']} streams is "
                f"{rows[-1]['aggregate_tok_s'] / rows[0]['aggregate_tok_s']:.1f}x the "
                f"aggregate of {rows[0]['streams']}")
    return rows or None


def sec_reasoning(a, cpt, rep):
    """The reasoning tax. Always measures BOTH states regardless of no_think:
    the comparison is the whole point of the section."""
    rep.section("reasoning", f"  REASONING TAX  (same prompt, thinking off vs on, "
                             f"{a.reasoning_tokens} token cap)")
    out = {}
    got = {}
    for i, (name, flag) in enumerate((("off", False), ("on", True))):
        _check(rep)
        rep.step(i, 2, f"thinking {name}")
        r = stream_chat(a.url, a.model, nonce() + REASONING_Q, a.reasoning_tokens,
                        thinking=flag, should_stop=rep.cancelled,
                        api_key=a.api_key)
        if r.get("cancelled"):
            raise Cancelled()
        if not r.get("ok"):
            rep.log(f"    thinking {name:<3} FAILED  {r.get('error')}")
            continue
        got[name] = r
        rep.step(i + 1, 2, f"thinking {name}")
        rep.log(f"    thinking {name:<3} {r['gen_tokens'] or 0:>5} tok  "
                f"{r['wall_s']:>6.2f}s  {r.get('decode_tok_s') or 0:>6.1f} tok/s")
    for name in ("off", "on"):
        if name in got:
            out[f"thinking_{name}_wall_s"] = got[name]["wall_s"]
            out[f"thinking_{name}_tokens"] = got[name]["gen_tokens"]
    if "off" in got and "on" in got and got["off"]["wall_s"]:
        ratio = got["on"]["wall_s"] / got["off"]["wall_s"]
        out["tax_x"] = round(ratio, 2)
        rep.log(f"    -> reasoning costs {ratio:.1f}x the wall time of the same answer")
    return out or None


SECTIONS = {"single": ("single_stream", sec_single),
            "prefill": ("prefill", sec_prefill),
            "sustained": ("sustained", sec_sustained),
            "concurrency": ("concurrency", sec_concurrency),
            "reasoning": ("reasoning_tax", sec_reasoning)}

SECTION_ORDER = list(SECTIONS)

SECTION_TITLES = {"single": "Single stream", "prefill": "Prefill scaling",
                  "sustained": "Sustained generation", "concurrency": "Concurrency",
                  "reasoning": "Reasoning tax"}


# ---------------------------------------------------------------- naming

def slug(model_id):
    """bench/SCHEMA.md filenames: the part after the org, lowercased, dots to
    underscores so a version number stays readable as one token."""
    base = (model_id or "model").rsplit("/", 1)[-1].lower().replace(".", "_")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9_-]", "-", base)).strip("-")


def int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def measure(opts: BenchOptions, rep: Reporter, cpt=None):
    """Run the selected sections and return (results, seconds, cpt).

    Blocking. Raises :class:`Cancelled` if the reporter reports a cancel, and
    :class:`ainode.bench.auth.EndpointRefused` if the node refused a request (401 or
    429): a refusal is not a section that measured badly, and swallowing one would
    write a record claiming five sections ran against a node that answered nothing.
    A section that raises anything else is logged and skipped: one broken section
    must not throw away the four that measured cleanly.
    """
    if cpt is None:
        cpt = calibrate_cpt(opts.url, opts.model, api_key=opts.api_key)
    t_start = time.time()
    results = {}
    for name in opts.sections:
        key, fn = SECTIONS[name]
        try:
            val = fn(opts, cpt, rep)
        except (Cancelled, auth.EndpointRefused):
            raise
        except Exception as e:
            rep.log(f"    section {name} raised {type(e).__name__}: {e}")
            val = None
        if val:
            results[key] = val
    return results, round(time.time() - t_start), cpt


def build_notes(opts: BenchOptions, results: dict, placement: dict, warnings: list,
                seconds: int, source: str):
    """The notes block: every caveat that would otherwise have to be remembered."""
    notes = [f"Measured by {source} in {seconds}s; nothing loaded, unloaded or restarted."]
    if "prefill" in results:
        notes.append("prefill_tok_s is prompt_tokens/TTFT, a floor: TTFT includes "
                     "queueing, and the OpenAI API exposes no internal prefill timing.")
    if opts.no_think:
        notes.append("Sections other than reasoning_tax ran with enable_thinking=false.")
    rt = results.get("reasoning_tax") or {}
    if rt.get("thinking_on_tokens") == opts.reasoning_tokens:
        notes.append(f"Thinking-on hit the {opts.reasoning_tokens}-token cap, so the "
                     "reasoning tax is a floor: the answer had not finished.")
    telemetry = results.get("telemetry") or {}
    # 0% is what the API returns for a GB10, not what the GPU was doing. Saying so
    # here is the difference between a caveat and a wrong number on the page.
    if telemetry and telemetry.get("gpu_util_pct") == 0:
        notes.append("GPU utilisation read 0% on every sample: AINode reports no "
                     "utilisation for a GB10 unified-memory GPU, so read it as unread "
                     "rather than idle. Memory and temperature are real reads.")
    if (placement.get("flags_source") or "").startswith("curated"):
        notes.append("Flags are the catalog recipe, not a read of the live container.")
    notes.extend(warnings or [])
    return notes


def build_record(opts: BenchOptions, model_block: dict, placement: dict, results: dict,
                 cpt: float, notes: list, stamp: str, source: str):
    """Assemble the schema-1 record. One place, so the CLI and the web run write
    byte-compatible files and bench/report.py never has to know which made it."""
    settings = {"thinking": False if opts.no_think else None,
                "max_tokens": opts.max_tokens, "temperature": None,
                "sustained_tokens": opts.sustained_tokens,
                "reasoning_tokens": opts.reasoning_tokens,
                "depths_requested": list(opts.depths),
                "streams_requested": list(opts.streams),
                "chars_per_token_probe": round(cpt, 2)}
    return {"schema": SCHEMA, "stamp": stamp, "label": opts.label,
            "model": model_block, "placement": placement, "settings": settings,
            "results": results, "notes": notes, "source": source}
