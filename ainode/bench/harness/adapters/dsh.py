"""dsh - DeepSeek's harness launcher (``@deepseek-ai/dsh``).

``dsh --profile headless "<job>"`` answers one task in the invoking directory,
prints the final message and exits, which is exactly the shape this bench wants.
It takes no endpoint or model flags: a profile is an ordered stack of plugin
config layers, and the last layer is ``--patch <file>``, a patch list applied over
everything else. So the adapter generates a two-entry overlay per run - one entry
defining an OpenAI-compatible provider on the ``llm-pi-ai`` plugin, one selecting
it on ``agent-default-model`` - and passes it with ``--patch``.

Verified offline on dsh 0.1.5-rc.1, no endpoint involved:

    DSH_HOME=... dsh --profile headless --patch <overlay> --dump-config

prints the composed tree with both overrides landed and the file named as the
patching layer. The overlay route means the bench never edits
``$DSH_HOME/settings.yaml``, where a person's own providers live.

One thing the overlay cannot avoid: dsh validates **every** provider route at
boot, including the ones already in ``settings.yaml``, so each of their
``apiKeyEnv`` variables has to be set to something. The adapter scans that file
for the names and fills in any that are unset with a placeholder. An AINode
endpoint ignores the key; the boot check only wants the variable to exist.
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
PROFILE = "headless"
PATCH_NAME = "dsh-harness.patch.yml"
SETTINGS_NAME = "settings.yaml"

#: The variable our own provider entry points at. Placeholder value throughout.
API_KEY_ENV = "AINODE_BENCH_API_KEY"

API_KEY_ENV_RE = re.compile(r"^\s*apiKeyEnv:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$",
                            re.MULTILINE)


def dsh_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get(ENV_HOME) or (pathlib.Path.home() / ".dsh"))


def settings_path(home: pathlib.Path | None = None) -> pathlib.Path:
    return (home or dsh_home()) / SETTINGS_NAME


def api_key_envs(settings_text: str) -> list[str]:
    """Every ``apiKeyEnv`` named in a dsh settings file, first mention first.

    A text scan rather than a YAML parse on purpose: the bench package carries no
    dependency beyond the standard library, and the only thing wanted from the
    file is a list of variable names.
    """
    seen = []
    for name in API_KEY_ENV_RE.findall(settings_text or ""):
        if name not in seen:
            seen.append(name)
    return seen


def patch_overlay(req: HarnessRequest) -> str:
    """The ``--patch`` overlay: a patch list, provider first, selection second.

    Scalars go through ``json.dumps`` so a model id keeps its slashes and dots
    intact; YAML reads a JSON-quoted string as a string.
    """
    model = json.dumps(req.model)
    return (
        "# Generated per run by ainode/bench/harness/adapters/dsh.py. Do not edit:\n"
        "# it is rewritten before every attempt and lives in the run's scratch dir.\n"
        "- id: llm-pi-ai\n"
        "  config:\n"
        "    providers:\n"
        f"      {PROVIDER}:\n"
        "        api: openai-completions\n"
        f"        baseURL: {json.dumps(req.endpoint)}\n"
        f"        apiKeyEnv: {API_KEY_ENV}\n"
        "        models:\n"
        f"          - id: {model}\n"
        f"            name: {model}\n"
        f"            contextWindow: {int(req.context_window)}\n"
        f"            maxTokens: {int(req.max_output_tokens)}\n"
        "- id: agent-default-model\n"
        "  config:\n"
        f"    provider: {PROVIDER}\n"
        f"    model: {model}\n"
    )


class DshAdapter(HarnessAdapter):
    name = "dsh"
    binary = "dsh"
    version_args = ("--version",)

    def patch_path(self, req: HarnessRequest) -> pathlib.Path:
        return req.scratch / PATCH_NAME

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        return [ConfigFile(self.patch_path(req), patch_overlay(req))]

    def command(self, req: HarnessRequest) -> list[str]:
        return [self.binary, "--profile", PROFILE,
                "--patch", str(self.patch_path(req)), req.prompt]

    def env(self, req: HarnessRequest) -> dict[str, str]:
        env = {API_KEY_ENV: req.api_key}
        path = settings_path()
        text = path.read_text() if path.is_file() else ""
        for name in api_key_envs(text):
            # Only fill the gaps: a real key already in the environment stays.
            if not os.environ.get(name):
                env.setdefault(name, req.api_key)
        return env
