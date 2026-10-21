"""Per-client limits on the inference paths.

A client can monopolize the engines with a burst of concurrent completions. A request-per-minute limit alone does not bound how many requests remain in flight.

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
