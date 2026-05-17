import asyncio
import logging
from typing import Any

import httpx

from ai.schemas import IngredientList

logger = logging.getLogger(__name__)

# Max concurrent USDA requests — stays within free-tier rate limit
_SEMAPHORE = asyncio.Semaphore(10)
_USDA_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


async def _fetch_nutrition(
    client: httpx.AsyncClient,
    ingredient: str,
    api_key: str,
) -> dict[str, Any]:
    """Fetch nutrition data for a single ingredient from USDA."""
    async with _SEMAPHORE:
        logger.debug("fetching nutrition for '%s'", ingredient)
        response = await client.get(
            _USDA_URL,
            params={"query": ingredient, "api_key": api_key, "pageSize": 1},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        foods = data.get("foods", [])
        return {"ingredient": ingredient, "data": foods[0] if foods else {}}


async def fetch_all_nutrition(
    ingredients: IngredientList,
    api_key: str,
) -> list[dict[str, Any]]:
    """Fetch nutrition for all ingredients in parallel."""
    names = [item.name for item in ingredients.items]
    logger.info("fetching nutrition for %d ingredients", len(names))

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_nutrition(client, name, api_key)
            for name in names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes from failures — don't crash on partial failure
    output = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.warning("failed to fetch '%s': %s", name, result)
            output.append({"ingredient": name, "data": {}})
        else:
            output.append(result)

    logger.info("nutrition fetch complete: %d ok, %d failed",
        sum(1 for r in results if not isinstance(r, Exception)),
        sum(1 for r in results if isinstance(r, Exception)),
    )
    return output
