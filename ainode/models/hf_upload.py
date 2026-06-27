"""Push a local checkpoint (e.g. a quantized model) to the Hugging Face Hub.

Pure huggingface_hub (a core dep present in the slim orchestrator) — no torch —
so this runs in the orchestrator process, not a GPU container. Token resolution
mirrors _run_training.py: request override > SecretsManager 'huggingface_token' >
NodeConfig.hf_token > env. Pushes are PRIVATE by default; read-only tokens are
rejected up front rather than 403-ing after a multi-GB transfer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_hf_token(app, override: Optional[str] = None) -> Optional[str]:
    if override:
        return override
    secrets = app.get("secrets_manager")
    if secrets is not None:
        try:
            tok = secrets.get("huggingface_token")
            if tok:
                return tok
        except Exception:
            pass
    cfg = app.get("config")
    if cfg is not None and getattr(cfg, "hf_token", None):
        return cfg.hf_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _whoami(token: str) -> dict:
    from huggingface_hub import HfApi
    return HfApi(token=token).whoami()


def assert_write_scope(token: str) -> dict:
    """Raise if the token can't create/push repos. Returns the whoami dict (for
    the default namespace). Classic read tokens have role 'read'."""
    if not token:
        raise ValueError("no Hugging Face token configured (set one via `ainode config "
                         "--hf-token` or the Secrets UI)")
    who = _whoami(token)
    role = (((who.get("auth") or {}).get("accessToken") or {}).get("role") or "").lower()
    if role == "read":
        raise ValueError("the configured Hugging Face token is read-only — a write "
                         "(or write-scoped fine-grained) token is required to push")
    return who


def upload_checkpoint(
    local_dir: str,
    repo_name: str,
    token: str,
    namespace: Optional[str] = None,
    private: bool = True,
) -> str:
    """Create (if needed) and upload ``local_dir`` to ``<namespace>/<repo_name>``.
    Returns the repo URL. ``namespace`` defaults to the token owner."""
    from huggingface_hub import HfApi

    d = Path(local_dir)
    if not d.is_dir() or not any(d.iterdir()):
        raise ValueError(f"checkpoint dir is empty or missing: {local_dir}")

    who = assert_write_scope(token)
    ns = namespace or who.get("name")
    if not ns:
        raise ValueError("could not determine a Hugging Face namespace from the token")
    repo_id = repo_name if "/" in repo_name else f"{ns}/{repo_name}"

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(d),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload quantized checkpoint from AINode",
    )
    return f"https://huggingface.co/{repo_id}"


def demo() -> None:
    """ponytail self-check: namespace resolution + read-only rejection logic
    without hitting the network."""
    class _Who:
        @staticmethod
        def role(r):
            return {"auth": {"accessToken": {"role": r}}, "name": "me"}
    # read-only must raise
    import types
    g = globals()
    saved = g["_whoami"]
    try:
        g["_whoami"] = lambda t: _Who.role("read")
        try:
            assert_write_scope("x"); assert False, "read token should reject"
        except ValueError:
            pass
        g["_whoami"] = lambda t: _Who.role("write")
        assert assert_write_scope("x")["name"] == "me"
    finally:
        g["_whoami"] = saved
    print("ok")


if __name__ == "__main__":
    demo()
