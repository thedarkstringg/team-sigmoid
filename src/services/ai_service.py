import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    retry_if_exception_type,
)
from ai import vlm, calculator
from ai.schemas import IngredientList, NutritionTotals

logger = logging.getLogger(__name__)

# Retry policy: up to 3 attempts, exponential backoff 1s→10s, +jitter
_RETRY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@retry(**_RETRY)
def identify_ingredients(image_path: str) -> IngredientList:
    """Call the VLM to identify ingredients — retries on failure."""
    logger.info("identifying ingredients", extra={"image": image_path})
    result = vlm.identify_ingredients(image_path)
    logger.debug("identified %d ingredients", len(result.items))
    return result


@retry(**_RETRY)
def compute_totals(ingredients: IngredientList) -> NutritionTotals:
    """Compute nutrition totals — retries on failure."""
    logger.info("computing nutrition totals")
    result = calculator.compute_totals(ingredients)
    logger.debug("totals: %s", result)
    return result