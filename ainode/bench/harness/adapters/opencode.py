"""opencode - the OpenCode CLI (``opencode-ai``).

Verified against opencode 1.18.31's own ``--help`` / ``run --help`` on this
machine: ``opencode run [message..]`` is the non-interactive entry point, ``-m``
takes ``provider/model``, ``--pure`` skips external plugins, and ``--auto``
approves permissions that are not explicitly denied. ``--auto`` is required here:
without a TTY there is nobody to approve the file write, so the run would sit
until the timeout.

The provider is a project-local ``opencode.json``, written into the working
directory. That is config, not a hint: it names the AINode endpoint as an
OpenAI-compatible provider through ``@ai-sdk/openai-compatible``. OpenCode expects
to be inside a git repo, so the working directory gets a bare ``git init`` with no
commit and no identity.
"""
from __future__ import annotations

import json

from ainode.bench.harness.adapters import (
    PROVIDER,
    ConfigFile,
    HarnessAdapter,
    HarnessRequest,
)

CONFIG_NAME = "opencode.json"
SCHEMA_URL = "https://opencode.ai/config.json"


def project_config(req: HarnessRequest) -> dict:
    return {
        "$schema": SCHEMA_URL,
        "provider": {
            PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AINode harness bench",
                "options": {"baseURL": req.endpoint, "apiKey": req.api_key},
                "models": {req.model: {"name": req.model,
                                       "limit": {"context": req.context_window,
                                                 "output": req.max_output_tokens}}},
            },
        },
    }


class OpencodeAdapter(HarnessAdapter):
    name = "opencode"
    binary = "opencode"
    needs_git = True

    def config(self, req: HarnessRequest) -> list[ConfigFile]:
        # In the working directory on purpose: opencode reads project config from
        # its cwd, and the file is the run's own provider wiring rather than
        # anything about the task. It is not a test file, so the isolation rule is
        # untouched.
        return [ConfigFile(req.workdir / CONFIG_NAME,
                           json.dumps(project_config(req), indent=1) + "\n")]

    def command(self, req: HarnessRequest) -> list[str]:
        return [
            self.binary, "run",
            "--pure",
            "--auto",
            "-m", f"{PROVIDER}/{req.model}",
            req.prompt,
        ]
