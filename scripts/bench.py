from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.schemas import Ingredient, NutritionFacts
from src.concurrency import pipeline


DELAY_SECONDS = 0.05
ITEM_COUNT = 12


class FakeUSDAProvider:
    """Deterministic fake provider for offline benchmarking."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_nutrition(self, ingredient_name: str) -> NutritionFacts:
        time.sleep(DELAY_SECONDS)
        base = 50 + (sum(ord(ch) for ch in ingredient_name) % 100)
        return NutritionFacts(
            name=ingredient_name,
            kcal_per_100g=float(base),
            protein_g_per_100g=5.0,
            carbs_g_per_100g=20.0,
            fat_g_per_100g=2.0,
            source="fake-benchmark",
        )


def make_ingredients() -> list[Ingredient]:
    return [
        Ingredient(
            name=f"ingredient-{i}",
            estimated_grams=100.0,
            confidence=0.95,
        )
        for i in range(ITEM_COUNT)
    ]


def fetch_nutrition_sequential(
    ingredients: list[Ingredient],
) -> dict[str, NutritionFacts]:
    provider = FakeUSDAProvider(api_key="offline")
    facts_by_name: dict[str, NutritionFacts] = {}

    for ingredient in ingredients:
        facts_by_name[ingredient.name] = provider.get_nutrition(ingredient.name)

    return facts_by_name


async def fetch_nutrition_concurrent(
    ingredients: list[Ingredient],
) -> dict[str, NutritionFacts]:
    original_provider = pipeline.USDAProvider
    pipeline.USDAProvider = FakeUSDAProvider

    try:
        return await pipeline.fetch_nutrition_parallel(ingredients)
    finally:
        pipeline.USDAProvider = original_provider


def main() -> None:
    ingredients = make_ingredients()

    t0 = time.perf_counter()
    sequential_result = fetch_nutrition_sequential(ingredients)
    sequential_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    concurrent_result = asyncio.run(fetch_nutrition_concurrent(ingredients))
    concurrent_seconds = time.perf_counter() - t1

    speedup = sequential_seconds / concurrent_seconds if concurrent_seconds else 0.0

    assert len(sequential_result) == ITEM_COUNT
    assert len(concurrent_result) == ITEM_COUNT

    print("Sequential vs concurrent nutrition lookup benchmark")
    print()
    print(f"{'mode':<12} {'items':>5} {'duration_seconds':>18}")
    print("-" * 39)
    print(f"{'sequential':<12} {ITEM_COUNT:>5} {sequential_seconds:>18.3f}")
    print(f"{'concurrent':<12} {ITEM_COUNT:>5} {concurrent_seconds:>18.3f}")
    print("-" * 39)
    print(f"speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
