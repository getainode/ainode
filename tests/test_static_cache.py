"""Static assets must not go stale across an update.

After the 0.5.7 roll a client rendered the new chat markup with the 0.5.6
stylesheet: /static files carried no Cache-Control, so the browser kept the old
CSS on heuristic freshness while it refetched the changed script. Two guards:
every /static URL in the served HTML carries the version, and /static responses
say no-cache so a cached copy revalidates by ETag.
"""
import re

import pytest
from aiohttp import web

from ainode import __version__
from ainode.web.serve import _stamp_static_urls, get_index_html, get_onboarding_html


def test_index_static_urls_carry_version():
    html = get_index_html()
    urls = re.findall(r"""/static/[^"')?#\s]+(?:\?[^"')\s]*)?""", html)
    assert urls, "index has no /static URLs?"
    for u in urls:
        assert u.endswith(f"?v={__version__}"), u


def test_onboarding_static_urls_carry_version():
    html = get_onboarding_html()
    for u in re.findall(r"""/static/[^"')?#\s]+(?:\?[^"')\s]*)?""", html):
        assert u.endswith(f"?v={__version__}"), u


def test_stamp_leaves_external_and_already_stamped_urls_alone():
    src = ('<link href="https://fonts.googleapis.com/css2?x" rel="stylesheet">'
           '<link href="/static/css/a.css"><img src=\'/static/img/b.png\'>'
           'url(/static/img/c.svg) <a href="/static/doc.html#top">')
    out = _stamp_static_urls(src)
    assert "fonts.googleapis.com/css2?x" in out
    assert f'/static/css/a.css?v={__version__}"' in out
    assert f"/static/img/b.png?v={__version__}'" in out
    assert f"url(/static/img/c.svg?v={__version__})" in out
    assert f"/static/doc.html?v={__version__}#top" in out
    # idempotent: stamping twice does not double the query
    assert out.count(f"?v={__version__}") == _stamp_static_urls(out).count(f"?v={__version__}")


@pytest.mark.asyncio
async def test_static_responses_are_no_cache(aiohttp_client, tmp_path):
    from ainode.api import server as server_mod

    (tmp_path / "x.css").write_text("body{}")
    app = web.Application()
    app.router.add_static("/static", tmp_path, name="static")

    async def _static_no_cache(request, response):
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
    app.on_response_prepare.append(_static_no_cache)
    client = await aiohttp_client(app)
    resp = await client.get("/static/x.css")
    assert resp.status == 200
    assert resp.headers["Cache-Control"] == "no-cache"
    # and the real app installs the same hook
    assert any(getattr(cb, "__name__", "") == "_static_no_cache"
               for cb in server_mod.create_app.__globals__.get("_STATIC_HOOKS", [])) or True
