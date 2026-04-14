"""Secrets management for AINode.

Stores credentials (HuggingFace token, NVIDIA NGC key, W&B, OpenAI, custom
training keys, etc.) at ~/.ainode/secrets.json with mode 0600 and XOR
obfuscation against a host machine identifier.
"""

from ainode.secrets.manager import SecretsManager, KNOWN_SECRETS

__all__ = ["SecretsManager", "KNOWN_SECRETS"]
