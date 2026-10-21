"""pi - the Pi coding agent CLI (``@mariozechner/pi-coding-agent``).

Verified against pi 0.73.1's own ``--help`` on this machine:

  * ``-p`` / ``--print`` is a boolean that makes the run non-interactive; the
    prompt is a positional argument, so it goes last.
  * ``--provider <name>`` selects a provider and ``--model <id>`` a model inside
    it. There is no base-URL flag: an OpenAI-compatible endpoint is a provider
    entry in ``~/.pi/agent/models.json``, whose shape was read off a working file
    (``baseUrl``, ``api: openai-completions``, ``apiKey``, ``models[]``).
  * pi has no permission prompts at all, so nothing else is needed to let it edit.

The provider entry is **merged** into whatever is already in that file: one key
under ``providers`` is added or replaced and the rest is left byte-for-byte alone.
``$AINODE_HARNESS_PI_HOME`` moves the file, which is how the tests write into a
tmp dir instead of the real one.
"""
from __future__ import annotations

import json
import os
import pathlib

from ainode.bench.harness.adapters import (
    PROVIDER,
    ConfigFile,
    HarnessAdapter,
    HarnessRequest,
)

ENV_HOME = "AINODE_HARNESS_PI_HOME"
CONFIG_REL = pathlib.PurePosixPath(".pi/agent/models.json")

#: Tools pi is allowed to use. Everything it needs to read the instructions, write
#: the solution and check its work, and nothing that reaches the network.
TOOLS = "read,grep,find,ls,edit,write,bash"


def config_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get(ENV_HOME) or pathlib.Path.home()).expanduser()


def config_path(home: pathlib.Path | None = None) -> pathlib.Path:
    return (home or config_home()) / CONFIG_REL


def provider_entry(req: HarnessRequest) -> dict:
    """The one provider block pi needs to route to an AINode endpoint."""
    return {
        "baseUrl": req.endpoint,
        "api": "openai-completions",
        "apiKey": req.api_key,
        "models": [{
            "id": req.model,
            "name": f"{req.model} (AINode harness bench)",
            "contextWindow": req.context_window,
            "maxTokens": req.max_output_tokens,
        }],
    }


def merge_models_json(existing: str, req: HarnessRequest) -> str:
    """Add our provider to an existing ``models.json`` without losing the rest.

    A file we cannot parse is an error, not something to overwrite: the bench does
    not get to destroy somebody's provider list because a comma was missing.
    """
    data = {}
    if existing.strip():
        data = json.loads(existing)
        if not isinstance(data, dict):
            raise ValueError("pi models.json must hold a JSON object")
    providers = dict(data.get("providers") or {})
    providers[PROVIDER] = provider_entry(req)
    data["providers"] = providers
    return json.dumps(data, indent=1) + "\n"


class PiAdapter(HarnessAdapter):
    name = "pi"
    binary = "pi"

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        path = config_path()
        existing = path.read_text() if path.is_file() else ""
        return [ConfigFile(path, merge_models_json(existing, req), merged=True)]

    def command(self, req: HarnessRequest) -> list[str]:
        return [
            self.binary,
            "--provider", PROVIDER,
            "--model", req.model,
            "--tools", TOOLS,
            # Nothing carries over between tasks or attempts beyond the working
            # directory, and no AGENTS.md / CLAUDE.md from an enclosing directory
            # is allowed to change what the model is told.
            "--no-session",
            "--no-context-files",
            "-p",
            req.prompt,
        ]

    def env(self, req: HarnessRequest) -> dict[str, str]:
        # pi reads the key out of the provider entry, so this only matters when
        # $AINODE_HARNESS_PI_HOME moved the config: HOME has to follow it or pi
        # would look for the file we did not write.
        home = os.environ.get(ENV_HOME)
        return {"HOME": str(pathlib.Path(home).expanduser())} if home else {}
