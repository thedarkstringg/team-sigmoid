import time

import pytest
from tenacity import wait_none

from ai.schemas import Ingredient, Nutrition, NutritionFacts
from src.services import ai_service


@pytest.fixture
def ingredients():
    return [
        Ingredient(name="white rice", estimated_grams=180.0, confidence=0.95),
        Ingredient(name="grilled chicken", estimated_grams=150.0, confidence=0.9),
    ]


@pytest.fixture
def facts_by_name():
    return {
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


@pytest.fixture(autouse=True)
def remove_retry_sleep(monkeypatch):
    monkeypatch.setattr(ai_service.identify_ingredients.retry, "wait", wait_none())
    monkeypatch.setattr(ai_service.compute_totals.retry, "wait", wait_none())


def test_call_with_timeout_returns_function_result():
    result = ai_service._call_with_timeout(lambda x, y: x + y, 2, 3, timeout_seconds=1)

    assert result == 5


def test_call_with_timeout_reraises_worker_exception():
    def broken_call():
        raise ValueError("bad provider response")

    with pytest.raises(ValueError, match="bad provider response"):
        ai_service._call_with_timeout(broken_call, timeout_seconds=1)


def test_call_with_timeout_raises_timeout_error():
    def slow_call():
        time.sleep(0.2)

    with pytest.raises(TimeoutError, match="Call timed out"):
        ai_service._call_with_timeout(slow_call, timeout_seconds=0.01)


def test_identify_ingredients_returns_vlm_result(monkeypatch, ingredients):
    def fake_call_with_timeout(fn, *args, timeout_seconds=30):
        assert fn is ai_service.vlm.identify_ingredients
        assert args == ("data/rice_chicken.png",)
        return ingredients

    monkeypatch.setattr(ai_service, "_call_with_timeout", fake_call_with_timeout)

    result = ai_service.identify_ingredients("data/rice_chicken.png")

    assert result == ingredients
    assert result[0].name == "white rice"


def test_identify_ingredients_retries_transient_connection_error(monkeypatch, ingredients):
    calls = {"count": 0}

    def flaky_call_with_timeout(fn, *args, timeout_seconds=30):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionError("temporary provider failure")
        return ingredients

    monkeypatch.setattr(ai_service, "_call_with_timeout", flaky_call_with_timeout)

    result = ai_service.identify_ingredients("image.png")

    assert result == ingredients
    assert calls["count"] == 2


def test_identify_ingredients_reraises_after_retry_limit(monkeypatch):
    calls = {"count": 0}

    def always_fails(fn, *args, timeout_seconds=30):
        calls["count"] += 1
        raise ConnectionError("provider still down")

    monkeypatch.setattr(ai_service, "_call_with_timeout", always_fails)

    with pytest.raises(ConnectionError, match="provider still down"):
        ai_service.identify_ingredients("image.png")

    assert calls["count"] == 3


def test_compute_totals_returns_calculator_result(monkeypatch, ingredients, facts_by_name):
    expected = Nutrition(kcal=481.5, protein_g=51.4, carbs_g=50.4, fat_g=5.9)

    def fake_call_with_timeout(fn, *args, timeout_seconds=30):
        assert fn is ai_service.calculator.compute_totals
        assert args == (ingredients, facts_by_name)
        return expected

    monkeypatch.setattr(ai_service, "_call_with_timeout", fake_call_with_timeout)

    result = ai_service.compute_totals(ingredients, facts_by_name)

    assert result == expected
    assert result.kcal == pytest.approx(481.5)


def test_compute_totals_reraises_transient_failure_after_retry_limit(monkeypatch, ingredients, facts_by_name):
    calls = {"count": 0}

    def always_fails(fn, *args, timeout_seconds=30):
        calls["count"] += 1
        raise TimeoutError("calculator timed out")

    monkeypatch.setattr(ai_service, "_call_with_timeout", always_fails)

    with pytest.raises(TimeoutError, match="calculator timed out"):
        ai_service.compute_totals(ingredients, facts_by_name)

    assert calls["count"] == 3
