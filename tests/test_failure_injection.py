import asyncio
import logging

from ai.schemas import Ingredient, NutritionFacts
from src.concurrency import pipeline


class FailureInjectingUSDAProvider:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_nutrition(self, ingredient_name: str) -> NutritionFacts:
        if ingredient_name == "provider 5xx":
            raise RuntimeError("simulated provider 5xx")
        if ingredient_name == "timeout item":
            raise TimeoutError("simulated provider timeout")
        if ingredient_name == "malformed item":
            raise ValueError("simulated malformed provider response")

        return NutritionFacts(
            name=ingredient_name,
            kcal_per_100g=100,
            protein_g_per_100g=10,
            carbs_g_per_100g=20,
            fat_g_per_100g=5,
            source="failure-injection-test",
        )


def test_parallel_nutrition_degrades_when_some_provider_calls_fail(monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "USDAProvider", FailureInjectingUSDAProvider)
    caplog.set_level(logging.WARNING)

    ingredients = [
        Ingredient(name="rice", estimated_grams=200, confidence=0.9),
        Ingredient(name="provider 5xx", estimated_grams=100, confidence=0.8),
        Ingredient(name="timeout item", estimated_grams=100, confidence=0.8),
        Ingredient(name="malformed item", estimated_grams=100, confidence=0.8),
        Ingredient(name="chicken", estimated_grams=150, confidence=0.9),
    ]

    result = asyncio.run(pipeline.fetch_nutrition_parallel(ingredients))

    assert set(result.keys()) == {"rice", "chicken"}
    assert result["rice"].kcal_per_100g == 100
    assert result["chicken"].protein_g_per_100g == 10
    assert "provider 5xx" not in result
    assert "timeout item" not in result
    assert "malformed item" not in result
    assert "pipeline.fetch_failed" in caplog.text


def test_parallel_nutrition_returns_empty_dict_when_all_provider_calls_fail(monkeypatch, caplog):
    monkeypatch.setattr(pipeline, "USDAProvider", FailureInjectingUSDAProvider)
    caplog.set_level(logging.WARNING)

    ingredients = [
        Ingredient(name="provider 5xx", estimated_grams=100, confidence=0.8),
        Ingredient(name="timeout item", estimated_grams=100, confidence=0.8),
        Ingredient(name="malformed item", estimated_grams=100, confidence=0.8),
    ]

    result = asyncio.run(pipeline.fetch_nutrition_parallel(ingredients))

    assert result == {}
    assert "pipeline.fetch_failed" in caplog.text
