import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Fail closed: if no keys are configured, nothing is authorised."""
    if key:
        for valid in settings.api_key_list:
            if secrets.compare_digest(key.encode(), valid.encode()):
                return
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Missing or invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


class _RateLimiter:
    """Small in-process sliding-window limiter (per client IP).

    For multi-replica deployments put a real limiter (nginx, Traefik, API gateway) in front.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


_limiter = _RateLimiter()


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if settings.rate_limit_per_minute <= 0:
        return
    client = request.client.host if request.client else "unknown"
    if not _limiter.check(client, settings.rate_limit_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
