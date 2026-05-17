import logging
from datetime import datetime, timezone

from ai import identify_ingredients, compute_totals
from ai.nutrition import USDAProvider
from src.config import settings
from src.models import AnalysisRecord, IngredientRow

logger = logging.getLogger(__name__)


async def run_analysis(image_path: str) -> AnalysisRecord:
    """
    Orchestrates the full meal analysis pipeline:
    1. identify_ingredients via VLM
    2. parallel nutrition lookups (delegated to concurrency/pipeline.py)
    3. compute_totals
    Returns a complete AnalysisRecord ready to be saved.
    """
    logger.info("analyzer.start", extra={"image_path": image_path})

    # Step 1 — identify ingredients via VLM
    ingredients = identify_ingredients(image_path)

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

    # Step 2 — nutrition lookups (parallel, via Emin's pipeline)
    # Once emin/asyncio-pipeline merges, replace below with:
    # from src.concurrency.pipeline import fetch_nutrition_parallel
    # facts = await fetch_nutrition_parallel(ingredients)
    provider = USDAProvider(api_key=settings.usda_api_key)
    facts = {}
    for ing in ingredients:
        try:
            facts[ing.name] = provider.get_nutrition(ing.name)
        except Exception as e:
            logger.warning("analyzer.nutrition_failed", extra={"ingredient": ing.name, "error": str(e)})

    # Step 3 — compute totals
    totals = compute_totals(ingredients, facts)

    # Build ingredient rows with per-ingredient nutrition
    ingredient_rows = []
    for ing in ingredients:
        nf = facts.get(ing.name)
        if nf:
            factor = ing.estimated_grams / 100.0
            row = IngredientRow(
                name=ing.name,
                estimated_grams=ing.estimated_grams,
                confidence=ing.confidence,
                kcal=round(nf.kcal * factor, 1),
                protein=round(nf.protein * factor, 1),
                carbs=round(nf.carbs * factor, 1),
                fat=round(nf.fat * factor, 1),
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
        total_protein=totals.protein,
        total_carbs=totals.carbs,
        total_fat=totals.fat,
        meal_recognized=True,
    )