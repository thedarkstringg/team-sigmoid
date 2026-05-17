import logging
from ai import vlm, calculator
from ai.schemas import IngredientList, NutritionTotals

logger = logging.getLogger(__name__)


def identify_ingredients(image_path: str) -> IngredientList:
    """Call the VLM to identify ingredients in an image."""
    logger.info("identifying ingredients", extra={"image": image_path})
    result = vlm.identify_ingredients(image_path)
    logger.debug("identified %d ingredients", len(result.items))
    return result


def compute_totals(ingredients: IngredientList) -> NutritionTotals:
    """Compute nutrition totals from identified ingredients."""
    logger.info("computing nutrition totals")
    result = calculator.compute_totals(ingredients)
    logger.debug("totals: %s", result)
    return result