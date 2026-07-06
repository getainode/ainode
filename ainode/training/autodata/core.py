"""AutoData core — Challenger -> weak/strong Solvers -> Judge Δ-filter -> ShareGPT JSONL.

Δ = I_strong - I_weak. Keep only Δ == 1 (strong solves, weak fails): the high-value
"zone of proximal development". Δ == 0 is too-easy (both pass) or too-hard (both fail);
Δ == -1 (weak passes, strong fails) is judge noise / a bad task — dropped.

Everything is OpenAI-compatible HTTP, so the four roles are just AInode-served endpoints
(strong = a big model, weak = a small model or low-compute pass). `call_model` is module-
level so tests inject a fake and run the whole loop with no network. See demo() at bottom.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Endpoint:
    url: str                       # OpenAI base, e.g. http://localhost:8001/v1
    model: str
    max_tokens: int = 512
    temperature: float = 0.0
    api_key: str = "EMPTY"
    timeout: int = 120


@dataclass
class AutoDataConfig:
    task_spec: str                 # what kind of tasks to generate
    gen_prompt: str                # the Challenger system prompt (P)
    challenger: Endpoint
    weak: Endpoint
    strong: Endpoint
    judge: Endpoint
    n_tasks: int = 50
    system_prompt: str = ""        # optional system msg for solvers + emitted data
    judge_mode: str = "rubric"     # "rubric" (LLM), "verify" (math/numeric, rubric fallback), or "exact" (substring)
    concurrency: int = 8
    retries: int = 2               # transient HTTP/parse retries per model call
    out: str = ""                  # JSONL output path (optional)
    # v2.2 — Evalchemy-style val-set objective (see valset.py). Ignored unless
    # objective == "valset" in the meta-loop; the v2.1 Δ=1-yield proxy stays the default.
    objective: str = "yield"       # meta-loop reward: "yield" (v2.1 proxy) | "valset" (v2.2 lift)
    val_set: list = None           # held-out labeled probes [{"input","reference"}] for the objective
    val_shots: int = 3             # few-shot examples (from kept) used to prime the weak solver on val
    val_target: float = 0.10       # meta stop-threshold for the valset objective (absolute lift)
    # v2.2 significance guard — a raw lift on a small val set is mostly sampling noise, so the
    # meta-loop only ACCEPTS (early-stops) a round when the lift also clears a paired McNemar
    # test at `alpha` AND the val set has at least `val_min_n` probes. Below that, the objective
    # is reported but never treated as a genuine round-over-round win.
    alpha: float = 0.05            # significance level for the paired (McNemar) lift test
    val_min_n: int = 12            # minimum val-set size before a lift can be accepted as real

    def __post_init__(self):
        if self.val_set is None:
            self.val_set = []

    @staticmethod
    def from_dict(d: dict) -> "AutoDataConfig":
        ep = lambda k: Endpoint(**d[k])  # noqa: E731
        return AutoDataConfig(
            task_spec=d["task_spec"], gen_prompt=d["gen_prompt"],
            challenger=ep("challenger"), weak=ep("weak"), strong=ep("strong"), judge=ep("judge"),
            n_tasks=int(d.get("n_tasks", 50)), system_prompt=d.get("system_prompt", ""),
            judge_mode=d.get("judge_mode", "rubric"), concurrency=int(d.get("concurrency", 8)),
            retries=int(d.get("retries", 2)), out=d.get("out", ""),
            objective=d.get("objective", "yield"), val_set=list(d.get("val_set") or []),
            val_shots=int(d.get("val_shots", 3)), val_target=float(d.get("val_target", 0.10)),
            alpha=float(d.get("alpha", 0.05)), val_min_n=int(d.get("val_min_n", 12)),
        )


def _http_chat(ep: Endpoint, messages: list, json_mode: bool = False) -> str:
    body = {"model": ep.model, "messages": messages,
            "temperature": ep.temperature, "max_tokens": ep.max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        ep.url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {ep.api_key}"},
    )
    with urllib.request.urlopen(req, timeout=ep.timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


# Injection point: tests replace this with a fake (signature: (Endpoint, messages, json_mode) -> str).
call_model = _http_chat


def _retry(fn, attempts: int = 2, base_delay: float = 0.5):
    """Run fn(); retry on ANY exception up to `attempts` extra times with exp backoff.
    Used to absorb transient HTTP timeouts and malformed-JSON responses (each retry
    re-calls the model, so a bad parse gets a fresh generation)."""
    last = None
    for i in range(attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — transient model/HTTP/parse failures
            last = exc
            if i < attempts:
                time.sleep(base_delay * (2 ** i))
    raise last


def _norm(s) -> str:
    # coerce non-str (the Challenger may emit numeric/None references) before normalizing
    return " ".join(str("" if s is None else s).lower().split())


def generate_tasks(cfg: AutoDataConfig) -> list:
    """Challenger emits n candidate tasks as JSON: [{"input", "reference"?}]."""
    user = (
        f"{cfg.task_spec}\n\nGenerate {cfg.n_tasks} DIVERSE, challenging tasks. "
        'Return STRICT JSON only: {"tasks":[{"input":"<the task prompt the solver sees>",'
        '"reference":"<the correct answer, or null if open-ended>"}]}'
    )
    def _gen():
        out = call_model(cfg.challenger,
                         [{"role": "system", "content": cfg.gen_prompt}, {"role": "user", "content": user}],
                         True)
        return json.loads(out).get("tasks", [])
    tasks = _retry(_gen, cfg.retries)
    return [t for t in tasks if isinstance(t, dict) and t.get("input")][: cfg.n_tasks]


def solve(ep: Endpoint, system: str, x: str, retries: int = 2) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": x}]
    return _retry(lambda: call_model(ep, msgs, False), retries)


def judge_correct(cfg: AutoDataConfig, x: str, output: str, reference) -> int:
    """1 if `output` is correct for task `x`, else 0."""
    if cfg.judge_mode == "verify":
        # Torch-free value verifier (math/numeric): definitive on parseable answers, None on
        # open-ended/symbolic → fall through to the LLM rubric below. Fixes the thin-ZPD where
        # exact-match misgraded formatting variants (0.5 vs 1/2) as wrong.
        from .verify import is_correct
        v = is_correct(output, reference)
        if v is not None:
            return int(v)
    elif cfg.judge_mode == "exact" and reference:
        return int(_norm(reference) in _norm(output))
    ref = f"\n\nReference answer:\n{reference}" if reference else ""
    prompt = (f"Task:\n{x}\n\nCandidate answer:\n{output}{ref}\n\n"
              'Is the candidate answer correct and high-quality for this task? '
              'Return STRICT JSON only: {"correct": true|false}.')
    def _grade():
        out = call_model(cfg.judge, [{"role": "user", "content": prompt}], True)
        return int(bool(json.loads(out).get("correct")))
    try:
        return _retry(_grade, cfg.retries)
    except Exception:
        # last-ditch: one plain (non-JSON-mode) call with a lenient parse
        try:
            out = call_model(cfg.judge, [{"role": "user", "content": prompt}], False).lower()
            return int('"correct": true' in out or out.strip().startswith(("yes", "true")))
        except Exception:
            return 0


def run(config, on_progress=None) -> dict:
    """Run the Δ-filter loop. `config` is an AutoDataConfig or a dict. Returns
    {kept: [sharegpt...], report: {...}, out: path}."""
    cfg = config if isinstance(config, AutoDataConfig) else AutoDataConfig.from_dict(config)
    tasks = generate_tasks(cfg)
    report = {"total": len(tasks), "kept": 0, "too_easy": 0, "too_hard": 0,
              "strong_worse": 0, "errors": 0}
    kept = []

    def process(t):
        x, ref = t["input"], t.get("reference")
        w = solve(cfg.weak, cfg.system_prompt, x, cfg.retries)
        s = solve(cfg.strong, cfg.system_prompt, x, cfg.retries)
        return x, s, judge_correct(cfg, x, w, ref), judge_correct(cfg, x, s, ref)

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        for res in ex.map(lambda t: _safe(process, t), tasks):
            if res is None:
                report["errors"] += 1
                continue
            x, strong_out, i_weak, i_strong = res
            delta = i_strong - i_weak
            if delta == 1:                       # strong solves, weak fails — KEEP
                conv = ([{"from": "system", "value": cfg.system_prompt}] if cfg.system_prompt else [])
                conv += [{"from": "human", "value": x}, {"from": "gpt", "value": strong_out}]
                kept.append({"conversations": conv})
                report["kept"] += 1
            elif delta == 0:
                report["too_easy" if i_strong == 1 else "too_hard"] += 1
            else:                                # delta == -1
                report["strong_worse"] += 1
            if on_progress:
                on_progress(report)

    if cfg.out:
        with open(cfg.out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    report["yield_pct"] = round(100 * report["kept"] / max(report["total"], 1))
    return {"kept": kept, "report": report, "out": cfg.out}


def _safe(fn, t):
    try:
        return fn(t)
    except Exception:
        return None


def demo() -> None:
    """Self-check with a fake model: only the strong-solves/weak-fails task survives."""
    global call_model
    # 3 tasks: A (only strong solves), B (both solve = too easy), C (neither = too hard)
    def fake(ep: Endpoint, messages: list, json_mode: bool) -> str:
        last = messages[-1]["content"]
        if json_mode and "Generate" in last:        # challenger
            return json.dumps({"tasks": [
                {"input": "TASK_A", "reference": "A"},
                {"input": "TASK_B", "reference": "B"},
                {"input": "TASK_C", "reference": "C"}]})
        if json_mode and "Candidate answer" in last:  # judge (exact-ish via rubric path)
            # mark correct unless the answer literally says WRONG
            return json.dumps({"correct": "WRONG" not in last})
        # solver: weak gets only B right; strong gets A and B right, C wrong
        task = messages[-1]["content"]
        strong = ep.model == "strong"
        if task == "TASK_A":
            return "A" if strong else "WRONG"
        if task == "TASK_B":
            return "B"
        return "WRONG"  # TASK_C: both wrong

    call_model = fake
    try:
        cfg = AutoDataConfig(
            task_spec="x", gen_prompt="x", n_tasks=3, concurrency=1,
            challenger=Endpoint("u", "challenger"), weak=Endpoint("u", "weak"),
            strong=Endpoint("u", "strong"), judge=Endpoint("u", "judge"))
        out = run(cfg)
        rep = out["report"]
        assert rep["total"] == 3, rep
        assert rep["kept"] == 1, rep            # only TASK_A (strong yes / weak no)
        assert rep["too_easy"] == 1, rep        # TASK_B
        assert rep["too_hard"] == 1, rep        # TASK_C
        assert out["kept"][0]["conversations"][-1]["value"] == "A", out["kept"]

        # verify-mode: the strong solver returns the RIGHT VALUE in a different FORMAT ("1/2"
        # for reference "0.5"). exact-match misgrades it (-> too_hard); the verifier keeps it.
        def fake_num(ep, messages, json_mode):
            last = messages[-1]["content"]
            if json_mode and "Generate" in last:
                return json.dumps({"tasks": [{"input": "HALF", "reference": "0.5"}]})
            if json_mode and "Candidate answer" in last:   # rubric (not reached for clean numeric)
                return json.dumps({"correct": False})
            return "1/2" if ep.model == "strong" else "999"
        call_model = fake_num
        base = dict(task_spec="x", gen_prompt="x", n_tasks=1, concurrency=1,
                    challenger=Endpoint("u", "challenger"), weak=Endpoint("u", "weak"),
                    strong=Endpoint("u", "strong"), judge=Endpoint("u", "judge"))
        exact = run(AutoDataConfig(judge_mode="exact", **base))["report"]
        ver = run(AutoDataConfig(judge_mode="verify", **base))["report"]
        assert exact["kept"] == 0 and exact["too_hard"] == 1, exact   # format variant lost
        assert ver["kept"] == 1 and ver["too_hard"] == 0, ver         # verifier recovers it
        print("autodata demo OK:", rep,
              "| signal shift exact->verify: kept", exact["kept"], "->", ver["kept"],
              "too_hard", exact["too_hard"], "->", ver["too_hard"])
    finally:
        call_model = _http_chat


if __name__ == "__main__":
    demo()
