import asyncio
import logging
import time

from ai.schemas import Ingredient, NutritionFacts
from ai.nutrition import USDAProvider
from src.config import settings
from src.services import nutrition_cache

logger = logging.getLogger(__name__)

# ← REMOVED: _SEMAPHORE = asyncio.Semaphore(settings.nutrition_concurrency_limit)


async def _fetch_one(
    ingredient: Ingredient,
    provider: USDAProvider,
    semaphore: asyncio.Semaphore,  # ← passed in, not module-level
) -> tuple[str, NutritionFacts | None]:
    async with semaphore:
        try:
            cached = nutrition_cache.get(ingredient.name)
            if cached is not None:
                logger.debug("pipeline.cache_hit", extra={"ingredient": ingredient.name})
                return ingredient.name, cached

            logger.debug("pipeline.fetch", extra={"ingredient": ingredient.name})
            facts = await asyncio.to_thread(provider.lookup, ingredient.name)

            if facts is not None:
                nutrition_cache.set(ingredient.name, facts, ttl=settings.cache_ttl_seconds)

            return ingredient.name, facts
        except Exception as e:
            logger.warning(
                "pipeline.fetch_failed",
                extra={"ingredient": ingredient.name, "error": str(e)},
            )
            return ingredient.name, None


async def fetch_nutrition_parallel(
    ingredients: list[Ingredient],
) -> dict[str, NutritionFacts]:
    """Fetch nutrition facts concurrently and keep only successful ingredient lookups."""
    t0 = time.monotonic()

    # ← created lazily here, inside a running event loop — safe in all contexts
    semaphore = asyncio.Semaphore(settings.nutrition_concurrency_limit)

    provider = USDAProvider(api_key=settings.usda_api_key)
    tasks = [_fetch_one(ing, provider, semaphore) for ing in ingredients]

    logger.info("pipeline.start", extra={"total": len(ingredients)})
    results = await asyncio.gather(*tasks, return_exceptions=False)

    facts_by_name: dict[str, NutritionFacts] = {}
    for name, facts in results:
        if facts is not None:
            facts_by_name[name] = facts

    logger.info(
        "pipeline.done",
        extra={
            "total": len(ingredients),
            "fetched": len(facts_by_name),
            "failed": len(ingredients) - len(facts_by_name),
            "duration_ms": round((time.monotonic() - t0) * 1000),
        },
    )
    return facts_by_name