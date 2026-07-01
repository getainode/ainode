"""Torch-free correctness verifier — numeric/math equivalence over brittle substring match.

The v1 Δ-filter graded "exact" by substring (`reference in output`), which misgrades any
formatting variant ("0.5" vs "1/2", "x = 2" vs "2", "\\frac{3}{4}" vs "0.75"). That brittle
check is the documented root cause of the *thin ZPD*: real correct answers get marked wrong,
inflating `too_hard` and starving the Δ=1 yield. This grades the VALUE, not the string.

ponytail: stdlib only (`fractions` + `re`) — no new dependency, so the slim orchestrator's
no-torch property holds by construction. Handles numbers, fractions, percents, simple
"x = V" / "answer: V" equations, whitespace, and light LaTeX. Ceiling: full symbolic algebra
(trig, multivariable, inequalities, sqrt) returns None → caller falls back to the LLM rubric;
swap in math-verify (sympy) for that, deferred to Slice 2b. See demo() at the bottom.
"""
from __future__ import annotations

import re
from fractions import Fraction


def _strip_latex(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^\$+|\$+$", "", s)                                   # $...$
    for a, b in ((r"\(", ""), (r"\)", ""), (r"\[", ""), (r"\]", "")):  # \(...\) \[...\]
        s = s.replace(a, b)
    s = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)  # \frac{a}{b}->a/b
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace("\\", "")
    return s.strip()


def _rhs(s: str) -> str:
    """Pull the answer value out of an equation/label: 'x = 2' -> '2', 'boxed{2}' -> '2'."""
    m = re.search(r"boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip()
    for sep in ("=", ":"):
        if sep in s:
            s = s.split(sep)[-1]
    return s.strip()


def _frac(x: str) -> Fraction:
    return Fraction(x.strip().strip("()").strip())


def _to_number(s: str):
    """Best-effort parse to a Fraction; None if not a clean single number."""
    t = _rhs(_strip_latex(s)).strip().rstrip(".").replace(",", "").replace(" ", "")
    if not t:
        return None
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    try:
        val = _frac(t.split("/", 1)[0]) / _frac(t.split("/", 1)[1]) if "/" in t else _frac(t)
    except (ValueError, ZeroDivisionError, IndexError):
        try:
            val = Fraction(float(t)).limit_denominator(10 ** 9)   # 1e3, 3.0, ...
        except (ValueError, OverflowError):
            return None
    return val / 100 if pct else val


def _norm_text(s) -> str:
    return " ".join(_strip_latex("" if s is None else s).lower().split())


def is_correct(output: str, reference) -> "bool | None":
    """True/False when a comparison can be made confidently; None when it can't, so the caller
    falls back to the LLM rubric. Conservative on purpose: it only ever turns a formatting-variant
    CORRECT answer back into a pass (killing the false-negatives that inflate too_hard); it never
    invents a new false-negative — anything it can't cleanly judge defers to the rubric."""
    if reference is None or str(reference).strip() == "":
        return None
    b = _to_number(str(reference))
    if b is not None:
        a = _to_number(output)
        return (a == b) if a is not None else None   # numeric ref: judge only clean-numeric output
    na, nb = _norm_text(output), _norm_text(reference)
    if not na or not nb:
        return None
    return True if (na == nb or nb in na) else None   # text: positive call only; else rubric


def demo() -> None:
    """Self-check: equivalence the old substring check would have failed."""
    assert is_correct("1/2", "0.5") is True
    assert is_correct("x = 2", "2") is True
    assert is_correct("50%", "0.5") is True
    assert is_correct(r"\frac{3}{4}", "0.75") is True
    assert is_correct("$0.5$", "1/2") is True
    assert is_correct("1,000", "1000") is True
    assert is_correct("7", "8") is False                  # genuinely-different numbers -> wrong
    assert is_correct("paris", None) is None              # no reference -> rubric fallback
    assert is_correct("The answer is 2.", "2") is None    # prose around a number -> rubric (conservative)
    assert is_correct(r"\sqrt{4}", "2") is None           # symbolic -> rubric fallback (ceiling)
    assert is_correct("london", "paris") is None          # wrong text -> rubric decides, not a false-neg
    print("autodata verify demo OK")


if __name__ == "__main__":
    demo()
