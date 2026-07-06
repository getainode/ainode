"""AINode — Turn any NVIDIA GPU into a local AI platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the installed package version (pyproject.toml).
    # Prevents drift between __version__, pyproject, and deployed image tags.
    __version__ = version("ainode")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0+unknown"

__author__ = "Argentos AI"
__url__ = "https://ainode.dev"
