"""Web UI file serving — serves the embedded dashboard."""

from pathlib import Path

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"


def get_index_html() -> str:
    """Return the main dashboard HTML."""
    index = TEMPLATES_DIR / "index.html"
    return index.read_text()


def get_static_path() -> Path:
    """Return the path to static assets directory."""
    return STATIC_DIR
