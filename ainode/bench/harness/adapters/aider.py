"""aider - the reference harness for this task set.

Aider's own polyglot benchmark runs the Exercism exercises, so a number measured
here can be read against published numbers for other models, and a surprise in
another adapter is more likely to be that adapter than the model.

Verified end to end on aider 0.86.2 against an AINode fleet endpoint: exit 0,
6/6 hidden tests passing on ``isogram``, with

    OPENAI_API_KEY=ainode aider \\
        --model openai/<model id> \\
        --openai-api-base http://<node>:3000/v1 \\
        --yes-always --no-git --no-auto-commits --no-show-model-warnings \\
        --message "<instructions>" <stub>

``openai/`` is a litellm provider prefix, not part of the model id: it tells
litellm to speak the OpenAI protocol to ``--openai-api-base``, which is what vLLM
serves and what AINode proxies on port 3000.
"""
from __future__ import annotations

import re

from ainode.bench.harness.adapters import HarnessAdapter, HarnessRequest

# "Tokens: 709 sent, 87 received." - and, on a long task, "Tokens: 12k sent, 1.3k
# received. Cost: ...". One line per LLM exchange, which is also the turn count.
TOKENS_RE = re.compile(
    r"Tokens:\s*([\d.]+)\s*([kKmM]?)\s*sent,\s*([\d.]+)\s*([kKmM]?)\s*received",
    re.IGNORECASE)
SUFFIX = {"": 1, "k": 1_000, "m": 1_000_000}


def _count(number: str, suffix: str) -> int:
    return int(round(float(number) * SUFFIX[suffix.lower()]))


def parse_tokens(text: str) -> dict:
    """Aider's own accounting: summed over every exchange it reported.

    Reported rather than measured: these are aider's numbers for what it sent to
    the endpoint, not the engine's ``usage`` block. They are recorded because they
    say how much context the harness spent to get the same result, which is the
    number that separates two harnesses driving one model.
    """
    matches = TOKENS_RE.findall(text or "")
    if not matches:
        return {}
    sent = sum(_count(m[0], m[1]) for m in matches)
    received = sum(_count(m[2], m[3]) for m in matches)
    return {"tokens_sent": sent, "tokens_received": received, "turns": len(matches)}


class AiderAdapter(HarnessAdapter):
    name = "aider"
    binary = "aider"

    def command(self, req: HarnessRequest) -> list[str]:
        return [
            self.binary,
            "--model", f"openai/{req.model}",
            "--openai-api-base", req.endpoint,
            # Verified set: approve every edit, stay out of git, say nothing about
            # the unknown model.
            "--yes-always",
            "--no-git",
            "--no-auto-commits",
            "--no-show-model-warnings",
            # Added beyond the verified set, both about side effects rather than
            # behaviour: never phone home for a release check, never report
            # analytics. --no-pretty keeps the captured tails free of ANSI so the
            # token line parses (aider already drops pretty output when stdout is
            # not a tty; this makes it explicit).
            "--no-check-update",
            "--no-analytics",
            "--no-pretty",
            "--message", req.prompt,
            req.entry,
        ]

    def env(self, req: HarnessRequest) -> dict[str, str]:
        # An AINode endpoint does not authenticate, but litellm requires the
        # variable to be set before it will build an OpenAI client.
        return {"OPENAI_API_KEY": req.api_key}

    def parse(self, stdout: str, stderr: str) -> dict:
        return parse_tokens(f"{stdout}\n{stderr}")
