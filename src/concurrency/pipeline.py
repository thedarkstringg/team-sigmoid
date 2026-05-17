import asyncio
import logging

from ai.schemas import Ingredient, NutritionFacts
from ai.nutrition import USDAProvider
from src.config import settings

logger = logging.getLogger(__name__)

_SEMAPHORE = asyncio.Semaphore(settings.nutrition_concurrency_limit)


async def _fetch_one(
    ingredient: Ingredient,
    provider: USDAProvider,
) -> tuple[str, NutritionFacts | None]:
    """Fetch nutrition for a single ingredient with semaphore."""
    async with _SEMAPHORE:
        try:
            logger.debug("pipeline.fetch", extra={"ingredient": ingredient.name})
            facts = await asyncio.to_thread(provider.get_nutrition, ingredient.name)
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
    """
    Fetch nutrition facts for all ingredients in parallel.
    Uses asyncio.gather + Semaphore(10) to cap concurrent USDA requests.
    Returns dict keyed by ingredient name — missing lookups are omitted.
    """
    provider = USDAProvider(api_key=settings.usda_api_key)

    tasks = [_fetch_one(ing, provider) for ing in ingredients]
    results = await asyncio.gather(*tasks)

    facts_by_name: dict[str, NutritionFacts] = {}
    for name, facts in results:
        if facts is not None:
            facts_by_name[name] = facts

    logger.info(
        "pipeline.done",
        extra={"total": len(ingredients), "fetched": len(facts_by_name)},
    )
    return facts_by_name