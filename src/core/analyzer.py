import logging
from datetime import datetime, timezone

from ai.schemas import Ingredient, NutritionFacts, Nutrition
from src.services.ai_service import identify_ingredients, compute_totals
from src.concurrency.pipeline import fetch_nutrition_parallel
from src.models import AnalysisRecord, IngredientRow

logger = logging.getLogger(__name__)


async def run_analysis(image_path: str) -> AnalysisRecord:
    """
    Orchestrates the full meal analysis pipeline:
    1. identify_ingredients via ai_service wrapper
    2. parallel nutrition lookups via asyncio pipeline
    3. compute_totals via ai_service wrapper
    Returns a complete AnalysisRecord ready to be saved.
    """
    logger.info("analyzer.start", extra={"image_path": image_path})

    # Step 1 — identify ingredients via ai_service wrapper
    ingredients: list[Ingredient] = identify_ingredients(image_path)

    if not ingredients:
        logger.warning("analyzer.no_meal_recognized", extra={"image_path": image_path})
        return AnalysisRecord(
            timestamp=datetime.now(timezone.utc),
            image_path=image_path,
            ingredients=[],
            total_kcal=0.0,
            total_protein=0.0,
            total_carbs=0.0,
            total_fat=0.0,
            meal_recognized=False,
        )

    # Step 2 — parallel nutrition lookups via pipeline
    facts_by_name: dict[str, NutritionFacts] = await fetch_nutrition_parallel(ingredients)

    # Step 3 — compute totals via ai_service wrapper
    totals: Nutrition = compute_totals(ingredients, facts_by_name)

    # Build ingredient rows with per-ingredient nutrition
    ingredient_rows: list[IngredientRow] = []
    for ing in ingredients:
        nf = facts_by_name.get(ing.name)
        if nf:
            factor = ing.estimated_grams / 100.0
            row = IngredientRow(
                name=ing.name,
                estimated_grams=ing.estimated_grams,
                confidence=ing.confidence,
                kcal=round(nf.kcal_per_100g * factor, 1),
                protein=round(nf.protein_g_per_100g * factor, 1),
                carbs=round(nf.carbs_g_per_100g * factor, 1),
                fat=round(nf.fat_g_per_100g * factor, 1),
            )
        else:
            row = IngredientRow(
                name=ing.name,
                estimated_grams=ing.estimated_grams,
                confidence=ing.confidence,
                kcal=0.0,
                protein=0.0,
                carbs=0.0,
                fat=0.0,
            )
        ingredient_rows.append(row)

    logger.info("analyzer.done", extra={
        "image_path": image_path,
        "ingredients": len(ingredient_rows),
        "total_kcal": totals.kcal,
    })

    return AnalysisRecord(
        timestamp=datetime.now(timezone.utc),
        image_path=image_path,
        ingredients=ingredient_rows,
        total_kcal=totals.kcal,
        total_protein=totals.protein_g,
        total_carbs=totals.carbs_g,
        total_fat=totals.fat_g,
        meal_recognized=True,
    )