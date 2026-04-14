"""Secrets storage for AINode.

Secrets are stored as JSON at ``~/.ainode/secrets.json`` with file mode 0600.
Values are obfuscated with an XOR against a per-host machine identifier before
being written to disk. This is NOT real encryption -- it just prevents casual
viewing if someone opens the file. The threat model here is "keep tokens from
being visible in ``cat`` output", not "defend against a determined attacker
with local access".

NEVER log secret values. NEVER include them in tracebacks. The API layer only
ever returns masked values (last 4 chars).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import socket
import stat
from pathlib import Path
from typing import Any, Dict, Optional

from ainode.core.config import AINODE_HOME


SECRETS_FILE = AINODE_HOME / "secrets.json"


# Known credential keys with metadata for the UI.
KNOWN_SECRETS: Dict[str, Dict[str, str]] = {
    "huggingface_token": {
        "label": "HuggingFace Token",
        "description": "Required to download gated models (Llama, Mistral, etc.) from the Hub.",
        "docs_url": "https://huggingface.co/settings/tokens",
        "prefix_hint": "hf_",
        "testable": "huggingface",
    },
    "nvidia_ngc_key": {
        "label": "NVIDIA NGC API Key",
        "description": "Used for pulling NGC-hosted models and containers.",
        "docs_url": "https://ngc.nvidia.com/setup/api-key",
        "prefix_hint": "",
        "testable": "",
    },
    "wandb_api_key": {
        "label": "Weights & Biases API Key",
        "description": "Optional -- stream training metrics to W&B.",
        "docs_url": "https://wandb.ai/authorize",
        "prefix_hint": "",
        "testable": "",
    },
    "openai_api_key": {
        "label": "OpenAI API Key",
        "description": "Optional -- used for benchmark comparisons against hosted OpenAI models.",
        "docs_url": "https://platform.openai.com/api-keys",
        "prefix_hint": "sk-",
        "testable": "",
    },
}


def _mask(value: str) -> str:
    """Return a masked rendering of a secret value.

    Examples
    --------
    ``hf_abcdef1234`` -> ``hf_...1234``
    ``short`` -> ``••••``
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    # Keep a small prefix hint (first 3 chars) if it's ascii-printable,
    # otherwise just dots.
    prefix = value[:3] if value[:3].isascii() and value[:3].isprintable() else ""
    return f"{prefix}…{value[-4:]}" if prefix else f"…{value[-4:]}"


def _machine_id() -> bytes:
    """Return a stable per-host identifier for XOR obfuscation.

    Tries /etc/machine-id first (Linux), then falls back to hostname+platform.
    Always returns at least 16 bytes.
    """
    sources: list[str] = []
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            data = Path(path).read_text().strip()
            if data:
                sources.append(data)
                break
        except OSError:
            pass
    sources.append(socket.gethostname())
    sources.append(platform.node() or "")
    sources.append(platform.system() or "")
    raw = "|".join(sources).encode("utf-8")
    # SHA-256 gives us 32 bytes of key material.
    return hashlib.sha256(raw).digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def _obfuscate(value: str) -> str:
    """XOR the value with the machine id and base64-encode."""
    if value is None:
        return ""
    key = _machine_id()
    scrambled = _xor_bytes(value.encode("utf-8"), key)
    return base64.b64encode(scrambled).decode("ascii")


def _deobfuscate(blob: str) -> str:
    """Reverse of :func:`_obfuscate`. Returns empty string on failure."""
    if not blob:
        return ""
    try:
        key = _machine_id()
        scrambled = base64.b64decode(blob.encode("ascii"))
        return _xor_bytes(scrambled, key).decode("utf-8")
    except Exception:
        return ""


class SecretsManager:
    """Manage credentials for AINode.

    All values on disk are obfuscated. The in-memory representation holds the
    raw values, but callers should prefer :meth:`all` (which masks) unless they
    specifically need to authenticate against an external service.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else SECRETS_FILE
        # Known-key values: {key: raw_value}
        self._values: Dict[str, str] = {}
        # User-provided custom keys: {name: raw_value}
        self._custom: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return

        known = data.get("known", {})
        custom = data.get("custom", {})
        for key, blob in known.items():
            if key in KNOWN_SECRETS and isinstance(blob, str):
                self._values[key] = _deobfuscate(blob)
        for name, blob in custom.items():
            if isinstance(blob, str):
                self._custom[name] = _deobfuscate(blob)

    def _save(self) -> None:
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        payload = {
            "known": {k: _obfuscate(v) for k, v in self._values.items() if v},
            "custom": {k: _obfuscate(v) for k, v in self._custom.items() if v},
        }
        # Write atomically to avoid torn reads.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Known-key API
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        return bool(self._values.get(key)) or bool(self._custom.get(key))

    def get(self, key: str) -> Optional[str]:
        """Return the raw value for *key* (known or custom), or ``None``."""
        if key in self._values:
            return self._values[key] or None
        if key in self._custom:
            return self._custom[key] or None
        return None

    def set(self, key: str, value: str) -> None:
        """Set a known secret. Raises ``KeyError`` for unknown keys."""
        if key not in KNOWN_SECRETS:
            raise KeyError(f"Unknown secret key: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Secret value must be a non-empty string")
        self._values[key] = value.strip()
        self._save()

    def delete(self, key: str) -> bool:
        """Remove a known secret. Returns True if anything was removed."""
        if key in self._values:
            del self._values[key]
            self._save()
            return True
        return False

    def all(self, include_values: bool = False) -> Dict[str, Any]:
        """Return all known secrets.

        By default values are MASKED. Pass ``include_values=True`` only when
        you need the raw value for an outgoing request -- NEVER include raw
        values in any HTTP response.
        """
        out: Dict[str, Any] = {}
        for key, meta in KNOWN_SECRETS.items():
            raw = self._values.get(key, "")
            entry = {
                "key": key,
                "label": meta["label"],
                "description": meta["description"],
                "docs_url": meta["docs_url"],
                "prefix_hint": meta.get("prefix_hint", ""),
                "testable": bool(meta.get("testable")),
                "is_set": bool(raw),
                "masked": _mask(raw) if raw else "",
            }
            if include_values:
                entry["value"] = raw
            out[key] = entry
        return out

    # ------------------------------------------------------------------
    # Custom-key API (arbitrary user-provided secrets, e.g. for training)
    # ------------------------------------------------------------------

    def custom_all(self, include_values: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, raw in self._custom.items():
            entry = {
                "name": name,
                "is_set": bool(raw),
                "masked": _mask(raw) if raw else "",
            }
            if include_values:
                entry["value"] = raw
            out[name] = entry
        return out

    def custom_set(self, name: str, value: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Custom secret name must be a non-empty string")
        name = name.strip()
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Custom secret name must be alphanumeric (plus _ and -)")
        if name in KNOWN_SECRETS:
            raise ValueError(f"'{name}' is a reserved known key; use set() instead")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Secret value must be a non-empty string")
        self._custom[name] = value.strip()
        self._save()

    def custom_delete(self, name: str) -> bool:
        if name in self._custom:
            del self._custom[name]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Safety: never leak values via repr / str
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SecretsManager(known={list(self._values.keys())}, "
            f"custom={list(self._custom.keys())})"
        )
