"""Inference protocol headers, without orchestrator credentials or authority."""

import os


_PROTOCOL_HEADERS = frozenset({
    "content-type", "accept", "accept-encoding", "anthropic-version",
    "anthropic-beta", "openai-beta", "x-request-id",
})


def engine_headers(request):
    """Caller auth terminates at AINode; engine auth is a separate credential.

    An operator may set AINODE_ENGINE_API_KEY for protected upstream engines.
    No caller key, cookie, proxy marker or arbitrary extension header is sent.
    """
    headers = {k: v for k, v in request.headers.items()
               if k.lower() in _PROTOCOL_HEADERS}
    key = os.environ.get("AINODE_ENGINE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers
