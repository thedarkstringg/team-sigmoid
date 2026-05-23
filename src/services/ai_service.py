import logging
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    retry_if_exception_type,
)
from ai import vlm, calculator
from ai.schemas import Ingredient, NutritionFacts, Nutrition

logger = logging.getLogger(__name__)

_RETRY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@retry(**_RETRY)
def identify_ingredients(image_path: str) -> list[Ingredient]:
    """Call the VLM to identify ingredients — retries on failure."""
    t0 = time.monotonic()
    logger.info("ai_service.identify.start", extra={"image": image_path})
    result = vlm.identify_ingredients(image_path)
    logger.info(
        "ai_service.identify.done",
        extra={
            "count": len(result),
            "duration_ms": round((time.monotonic() - t0) * 1000),
        },
    )
    return result


@retry(**_RETRY)
def compute_totals(
    ingredients: list[Ingredient],
    facts_by_name: dict[str, NutritionFacts],
) -> Nutrition:
    """Compute nutrition totals — retries on failure."""
    t0 = time.monotonic()
    logger.info("ai_service.compute.start")
    result = calculator.compute_totals(ingredients, facts_by_name)
    logger.info(
        "ai_service.compute.done",
        extra={"duration_ms": round((time.monotonic() - t0) * 1000)},
    )
    return result