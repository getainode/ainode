"""Per-client limits on the inference paths.

One client could occupy every engine in the cluster: nothing in AINode counted
what a caller had in flight, so a script opening two hundred concurrent
completions took the whole fleet's scheduler and every other caller waited. The
request-per-minute figure people reach for first does not fix that, because two
hundred concurrent requests are one burst.

So there are two limits, and ``max_inflight`` is the one that matters on a GPU
node. See :mod:`ainode.ratelimit.middleware` for the shape and the exemptions.
"""

from ainode.ratelimit.middleware import (
    LIMITED_PREFIXES,
    RATE_LIMIT_TYPE,
    RateLimitConfig,
    RateLimiter,
    client_key,
    is_limited_path,
    rate_limit_middleware,
    rate_limit_status_fields,
)

__all__ = [
    "LIMITED_PREFIXES",
    "RATE_LIMIT_TYPE",
    "RateLimitConfig",
    "RateLimiter",
    "client_key",
    "is_limited_path",
    "rate_limit_middleware",
    "rate_limit_status_fields",
]
