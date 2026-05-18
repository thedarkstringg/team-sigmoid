import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default TTL: 24 hours in seconds
_DEFAULT_TTL: int = 24 * 60 * 60

# Internal cache store: { ingredient_name: (data, expires_at) }
_cache: dict[str, tuple[Any, float]] = {}


def get(ingredient: str) -> Any | None:
    """Return cached nutrition data or None if missing/expired."""
    entry = _cache.get(ingredient)
    if entry is None:
        logger.info("cache.miss", extra={"ingredient": ingredient})
        return None

    data, expires_at = entry
    if time.monotonic() > expires_at:
        logger.info("cache.expired", extra={"ingredient": ingredient})
        del _cache[ingredient]
        return None

    logger.info("cache.hit", extra={"ingredient": ingredient})
    return data


def set(ingredient: str, data: Any, ttl: int = _DEFAULT_TTL) -> None:
    """Store nutrition data with a TTL (seconds)."""
    expires_at = time.monotonic() + ttl
    _cache[ingredient] = (data, expires_at)
    logger.debug("cache.set", extra={"ingredient": ingredient, "ttl": ttl})


def invalidate(ingredient: str) -> None:
    """Manually remove a single entry from the cache."""
    _cache.pop(ingredient, None)
    logger.debug("cache.invalidated", extra={"ingredient": ingredient})


def clear() -> None:
    """Wipe the entire cache — useful in tests."""
    _cache.clear()
    logger.debug("cache.cleared")


def size() -> int:
    """Return number of currently cached entries (including expired)."""
    return len(_cache)