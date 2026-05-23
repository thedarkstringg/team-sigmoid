import asyncio

import pytest
from unittest.mock import AsyncMock

from ai.schemas import Ingredient, Nutrition, NutritionFacts
from src.core import analyzer
from src.models import AnalysisRecord


INGREDIENTS = [
    Ingredient(name="white rice", estimated_grams=180.0, confidence=0.95),
    Ingredient(name="grilled chicken", estimated_grams=150.0, confidence=0.90),
]

FACTS = {
    "white rice": NutritionFacts(
        name="white rice",
        kcal_per_100g=130.0,
        protein_g_per_100g=2.7,
        carbs_g_per_100g=28.0,
        fat_g_per_100g=0.3,
        source="fake",
    ),
    "grilled chicken": NutritionFacts(
        name="grilled chicken",
        kcal_per_100g=165.0,
        protein_g_per_100g=31.0,
        carbs_g_per_100g=0.0,
        fat_g_per_100g=3.6,
        source="fake",
    ),
}

TOTALS = Nutrition(kcal=481.5, protein_g=51.4, carbs_g=50.4, fat_g=5.9)


def test_run_analysis_happy_path(monkeypatch):
    monkeypatch.setattr(analyzer, "identify_ingredients", lambda image_path: INGREDIENTS)

    fake_fetch = AsyncMock(return_value=FACTS)
    monkeypatch.setattr(analyzer, "fetch_nutrition_parallel", fake_fetch)

    monkeypatch.setattr(analyzer, "compute_totals", lambda ingredients, facts: TOTALS)

    result = asyncio.run(analyzer.run_analysis("data/rice_chicken.png"))

    assert isinstance(result, AnalysisRecord)
    assert result.image_path == "data/rice_chicken.png"
    assert result.meal_recognized is True
    assert result.total_kcal == pytest.approx(481.5)
    assert result.total_protein == pytest.approx(51.4)
    assert result.total_carbs == pytest.approx(50.4)
    assert result.total_fat == pytest.approx(5.9)
    assert len(result.ingredients) == 2

    fake_fetch.assert_awaited_once_with(INGREDIENTS)


def test_run_analysis_builds_scaled_ingredient_rows(monkeypatch):
    monkeypatch.setattr(analyzer, "identify_ingredients", lambda image_path: INGREDIENTS)
    monkeypatch.setattr(analyzer, "fetch_nutrition_parallel", AsyncMock(return_value=FACTS))
    monkeypatch.setattr(analyzer, "compute_totals", lambda ingredients, facts: TOTALS)

    result = asyncio.run(analyzer.run_analysis("img.png"))

    rice_row = next(row for row in result.ingredients if row.name == "white rice")
    chicken_row = next(row for row in result.ingredients if row.name == "grilled chicken")

    assert rice_row.estimated_grams == 180.0
    assert rice_row.confidence == 0.95
    assert rice_row.kcal == pytest.approx(234.0)
    assert rice_row.protein == pytest.approx(4.9)
    assert rice_row.carbs == pytest.approx(50.4)
    assert rice_row.fat == pytest.approx(0.5)

    assert chicken_row.estimated_grams == 150.0
    assert chicken_row.kcal == pytest.approx(247.5)
    assert chicken_row.protein == pytest.approx(46.5)
    assert chicken_row.carbs == pytest.approx(0.0)
    assert chicken_row.fat == pytest.approx(5.4)


def test_run_analysis_unknown_meal_returns_structured_record(monkeypatch):
    monkeypatch.setattr(analyzer, "identify_ingredients", lambda image_path: [])

    fake_fetch = AsyncMock()
    monkeypatch.setattr(analyzer, "fetch_nutrition_parallel", fake_fetch)

    result = asyncio.run(analyzer.run_analysis("data/no_meal_blue.png"))

    assert isinstance(result, AnalysisRecord)
    assert result.image_path == "data/no_meal_blue.png"
    assert result.meal_recognized is False
    assert result.ingredients == []
    assert result.total_kcal == 0.0
    assert result.total_protein == 0.0
    assert result.total_carbs == 0.0
    assert result.total_fat == 0.0

    fake_fetch.assert_not_called()


def test_run_analysis_partial_nutrition_failure_keeps_successful_rows(monkeypatch):
    partial_facts = {
        "white rice": FACTS["white rice"],
    }
    partial_totals = Nutrition(kcal=234.0, protein_g=4.9, carbs_g=50.4, fat_g=0.5)

    monkeypatch.setattr(analyzer, "identify_ingredients", lambda image_path: INGREDIENTS)
    monkeypatch.setattr(analyzer, "fetch_nutrition_parallel", AsyncMock(return_value=partial_facts))
    monkeypatch.setattr(analyzer, "compute_totals", lambda ingredients, facts: partial_totals)

    result = asyncio.run(analyzer.run_analysis("img.png"))

    assert result.meal_recognized is True
    assert result.total_kcal == pytest.approx(234.0)

    rice_row = next(row for row in result.ingredients if row.name == "white rice")
    chicken_row = next(row for row in result.ingredients if row.name == "grilled chicken")

    assert rice_row.kcal == pytest.approx(234.0)
    assert chicken_row.kcal == 0.0
    assert chicken_row.protein == 0.0
    assert chicken_row.carbs == 0.0
    assert chicken_row.fat == 0.0


def test_run_analysis_passes_ingredients_and_facts_to_compute_totals(monkeypatch):
    captured = {}

    def fake_compute_totals(ingredients, facts):
        captured["ingredients"] = ingredients
        captured["facts"] = facts
        return TOTALS

    monkeypatch.setattr(analyzer, "identify_ingredients", lambda image_path: INGREDIENTS)
    monkeypatch.setattr(analyzer, "fetch_nutrition_parallel", AsyncMock(return_value=FACTS))
    monkeypatch.setattr(analyzer, "compute_totals", fake_compute_totals)

    asyncio.run(analyzer.run_analysis("img.png"))

    assert captured["ingredients"] == INGREDIENTS
    assert captured["facts"] == FACTS
