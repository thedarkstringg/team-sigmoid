import asyncio

from ai.schemas import Ingredient, NutritionFacts
from src.concurrency import pipeline


class FakeUSDAProvider:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_nutrition(self, ingredient_name: str) -> NutritionFacts:
        if ingredient_name == "bad ingredient":
            raise RuntimeError("simulated provider failure")

        return NutritionFacts(
            name=ingredient_name,
            kcal_per_100g=100,
            protein_g_per_100g=10,
            carbs_g_per_100g=20,
            fat_g_per_100g=5,
            source="fake",
        )


def test_fetch_nutrition_parallel_skips_failed_lookup(monkeypatch):
    monkeypatch.setattr(pipeline, "USDAProvider", FakeUSDAProvider)

    ingredients = [
        Ingredient(name="rice", estimated_grams=200, confidence=0.9),
        Ingredient(name="bad ingredient", estimated_grams=100, confidence=0.5),
        Ingredient(name="chicken", estimated_grams=150, confidence=0.8),
    ]

    result = asyncio.run(pipeline.fetch_nutrition_parallel(ingredients))

    assert set(result.keys()) == {"rice", "chicken"}
    assert result["rice"].kcal_per_100g == 100
    assert result["chicken"].protein_g_per_100g == 10
    assert "bad ingredient" not in result


def test_fetch_nutrition_parallel_empty_list_returns_empty_dict(monkeypatch):
    monkeypatch.setattr(pipeline, "USDAProvider", FakeUSDAProvider)

    result = asyncio.run(pipeline.fetch_nutrition_parallel([]))

    assert result == {}
