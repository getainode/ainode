"""Web UI file serving — serves the embedded dashboard."""

import re
from pathlib import Path

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"


def _stamp_static_urls(html: str) -> str:
    """Append ``?v=<version>`` to every ``/static/...`` URL in a template.

    Static files are served without Cache-Control, so browsers apply heuristic
    freshness from Last-Modified and can keep an old stylesheet for weeks after
    an update: the master rolled to 0.5.7 and a client rendered the new chat
    markup with the 0.5.6 CSS (bare inputs, an unstyled model card). A version
    query makes every release a new URL, so the swap is atomic per page load.
    """
    from ainode import __version__
    # The path must end at a delimiter: a URL that already carries a query
    # (e.g. one stamped on a previous pass) is left alone rather than being
    # re-matched one character short and stamped twice.
    return re.sub(r'(["\'(])(/static/[^"\')?#\s]+)(?=["\')#\s]|$)', rf"\1\2?v={__version__}", html)


def get_index_html() -> str:
    """Return the main dashboard HTML, static URLs stamped with the version."""
    index = TEMPLATES_DIR / "index.html"
    return _stamp_static_urls(index.read_text())


def get_static_path() -> Path:
    """Return the path to static assets directory."""
    return STATIC_DIR
