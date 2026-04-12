"""Tests for ainode.web — template and static file serving."""

from pathlib import Path
from ainode.web.serve import get_index_html, get_static_path, STATIC_DIR, TEMPLATES_DIR


def test_templates_dir_exists():
    assert TEMPLATES_DIR.is_dir()


def test_static_dir_exists():
    assert STATIC_DIR.is_dir()


def test_get_index_html():
    html = get_index_html()
    assert "AINode" in html
    assert "argentos.ai" in html
    assert "dashboard" in html
    assert "chat" in html
    assert "models" in html
    assert "training" in html


def test_get_index_html_has_js():
    html = get_index_html()
    assert "app.js" in html


def test_get_index_html_has_css():
    html = get_index_html()
    assert "style.css" in html


def test_get_static_path():
    path = get_static_path()
    assert isinstance(path, Path)
    assert path.is_dir()


def test_css_exists():
    css = STATIC_DIR / "css" / "style.css"
    assert css.exists()
    content = css.read_text()
    assert "--bg-primary" in content


def test_js_exists():
    js = STATIC_DIR / "js" / "app.js"
    assert js.exists()
    content = js.read_text()
    assert "AINode" in content
    assert "fetchJSON" in content
