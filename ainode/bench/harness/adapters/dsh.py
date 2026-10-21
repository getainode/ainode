"""dsh - DeepSeek's harness launcher (``@deepseek-ai/dsh``).

``dsh --profile headless "<job>"`` answers one task in the invoking directory,
prints the final assistant message on stdout (reasoning goes to stderr) and exits,
which is exactly the shape this bench wants. It takes no endpoint or model flags: a
profile is an ordered stack of plugin config layers, and the last layer is
``--patch <file>``, a patch list applied over everything else. So the adapter
generates a two-entry overlay per run, one entry defining an OpenAI-compatible
provider on the ``llm-pi-ai`` plugin and one selecting it on
``agent-default-model``, and passes it with ``--patch``.

Verified end to end on dsh 0.1.5-rc.1: the stub edited and 6/6 hidden tests green on
``isogram`` against a model on the fleet. The composition itself is also verifiable
offline, with no endpoint involved:

    DSH_HOME=... dsh --profile headless --patch <overlay> --dump-config

prints the composed tree with both overrides landed and the file named as the
patching layer.

**The bench runs dsh in its own ``DSH_HOME``, and that is load-bearing.** dsh
validates *every* configured provider route at boot, so a single stale entry in a
person's own ``~/.dsh/settings.yaml`` - a provider pointing at a port that stopped
serving - ends every run with ``dsh: TRANSPORT: Connection error.`` after about 17 s,
no matter which provider the run actually selected. That cost an afternoon of chasing
the wrong thing. So the adapter points ``DSH_HOME`` at
``~/.ainode/bench/harness/dsh-home``, writes a minimal ``settings.yaml`` there with
one route, and never reads or writes the real one. The directory persists between
runs on purpose: the first use of a profile installs it, which takes minutes.

``$AINODE_HARNESS_DSH_HOME`` moves that home. If it points at a directory that
already has a ``settings.yaml``, the adapter leaves the file completely alone and
relies on the overlay, so pointing it at a curated home is safe; every route in that
file then has to resolve, for the reason above.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

from ainode.bench.harness.adapters import (
    PROVIDER,
    ConfigFile,
    HarnessAdapter,
    HarnessRequest,
)

ENV_HOME = "DSH_HOME"
ENV_HOME_OVERRIDE = "AINODE_HARNESS_DSH_HOME"
PROFILE = "headless"
PATCH_NAME = "dsh-harness.patch.yml"
SETTINGS_NAME = "settings.yaml"

#: The variable our own provider entry points at. Placeholder value throughout.
API_KEY_ENV = "AINODE_BENCH_API_KEY"

API_KEY_ENV_RE = re.compile(r"^\s*apiKeyEnv:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$",
                            re.MULTILINE)


def default_home() -> pathlib.Path:
    return pathlib.Path.home() / ".ainode" / "bench" / "harness" / "dsh-home"


def bench_home() -> pathlib.Path:
    """The ``DSH_HOME`` the bench drives dsh with. Never the real ``~/.dsh``."""
    override = os.environ.get(ENV_HOME_OVERRIDE)
    return pathlib.Path(override).expanduser() if override else default_home()


def settings_path(home: pathlib.Path | None = None) -> pathlib.Path:
    return (home or bench_home()) / SETTINGS_NAME


def api_key_envs(settings_text: str) -> list[str]:
    """Every ``apiKeyEnv`` named in a dsh settings file, first mention first.

    A text scan rather than a YAML parse on purpose: the bench package carries no
    dependency beyond the standard library, and the only thing wanted from the file
    is a list of variable names. Every one of them has to be set to something or the
    boot-time route check fails.
    """
    seen = []
    for name in API_KEY_ENV_RE.findall(settings_text or ""):
        if name not in seen:
            seen.append(name)
    return seen


def thinking_format(model: str) -> str | None:
    """dsh's ``compat.thinkingFormat`` for the model, when it needs one.

    DeepSeek models return reasoning in DeepSeek's own shape; the plugin has to be
    told so, or the reasoning arrives as content. Keyed off the model id because an
    OpenAI-compatible endpoint exposes nothing else to key off.
    """
    return "deepseek" if "deepseek" in (model or "").lower() else None


def _provider_lines(req: HarnessRequest, indent: str) -> list[str]:
    """The provider block's body, shared by the settings file and the overlay.

    Scalars go through ``json.dumps`` so a model id keeps its slashes and dots
    intact; YAML reads a JSON-quoted string as a string.
    """
    model = json.dumps(req.model)
    lines = [f"{indent}api: openai-completions",
             f"{indent}baseURL: {json.dumps(req.endpoint)}",
             f"{indent}apiKeyEnv: {API_KEY_ENV}"]
    fmt = thinking_format(req.model)
    if fmt:
        lines += [f"{indent}compat:", f"{indent}  thinkingFormat: {fmt}"]
    lines += [f"{indent}models:",
              f"{indent}  - id: {model}",
              f"{indent}    name: {model}",
              f"{indent}    contextWindow: {int(req.context_window)}",
              f"{indent}    maxTokens: {int(req.max_output_tokens)}"]
    return lines


def patch_overlay(req: HarnessRequest) -> str:
    """The ``--patch`` overlay: a patch list, provider first, selection second."""
    body = "\n".join(_provider_lines(req, " " * 8))
    return (
        "# Generated per run by ainode/bench/harness/adapters/dsh.py. Do not edit:\n"
        "# it is rewritten before every attempt and lives in the run's scratch dir.\n"
        "- id: llm-pi-ai\n"
        "  config:\n"
        "    providers:\n"
        f"      {PROVIDER}:\n"
        f"{body}\n"
        "- id: agent-default-model\n"
        "  config:\n"
        f"    provider: {PROVIDER}\n"
        f"    model: {json.dumps(req.model)}\n"
    )


def settings_yaml(req: HarnessRequest) -> str:
    """The minimal ``settings.yaml`` for the bench's isolated ``DSH_HOME``.

    One route and one default, so the boot-time check has exactly one thing to
    validate and a hand-run of ``dsh --profile headless`` in that home works too.
    """
    body = "\n".join(_provider_lines(req, " " * 6))
    return (
        "# Written by AINode's harness bench (ainode/bench/harness/adapters/dsh.py).\n"
        "# One route on purpose: dsh validates every configured provider at boot, so a\n"
        "# stale entry anywhere in this file would fail every run whatever it selected.\n"
        "llm-pi-ai:\n"
        "  providers:\n"
        f"    {PROVIDER}:\n"
        f"{body}\n"
        "agent-default-model:\n"
        f"  provider: {PROVIDER}\n"
        f"  model: {json.dumps(req.model)}\n"
    )


class DshAdapter(HarnessAdapter):
    name = "dsh"
    binary = "dsh"
    version_args = ("--version",)

    def patch_path(self, req: HarnessRequest) -> pathlib.Path:
        return req.scratch / PATCH_NAME

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        home = bench_home()
        settings = settings_path(home)
        files = []
        # A home somebody pointed us at and already curated is theirs: leave it, and
        # let the overlay carry the provider. Our own home we own and rewrite.
        if not (settings.is_file() and os.environ.get(ENV_HOME_OVERRIDE)):
            files.append(ConfigFile(settings, settings_yaml(req)))
        files.append(ConfigFile(self.patch_path(req), patch_overlay(req)))
        return files

    def command(self, req: HarnessRequest) -> list[str]:
        return [self.binary, "--profile", PROFILE,
                "--patch", str(self.patch_path(req)), req.prompt]

    def env(self, req: HarnessRequest) -> dict[str, str]:
        home = bench_home()
        env = {ENV_HOME: str(home), API_KEY_ENV: req.api_key}
        settings = settings_path(home)
        text = settings.read_text() if settings.is_file() else ""
        for name in api_key_envs(text):
            # Only fill the gaps: a real key already in the environment stays.
            if not os.environ.get(name):
                env.setdefault(name, req.api_key)
        return env
