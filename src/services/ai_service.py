import logging
from ai import vlm, calculator
from ai.schemas import Ingredient, NutritionFacts, Nutrition

logger = logging.getLogger(__name__)


def identify_ingredients(image_path: str) -> list[Ingredient]:
    logger.info("identifying ingredients", extra={"image": image_path})
    result = vlm.identify_ingredients(image_path)
    logger.debug("identified %d ingredients", len(result))
    return result


def compute_totals(
    ingredients: list[Ingredient],
    facts_by_name: dict[str, NutritionFacts],
) -> Nutrition:
    logger.info("computing nutrition totals")
    result = calculator.compute_totals(ingredients, facts_by_name)
    logger.debug("totals: %s", result)
    return result