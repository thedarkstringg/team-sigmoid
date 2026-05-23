import logging
import time
import threading
import httpx
import requests
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

_TIMEOUT_SECONDS = 30

_RETRY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type((
        httpx.HTTPError,
        requests.RequestException,
        TimeoutError,
        ConnectionError,
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _call_with_timeout(fn, *args, timeout_seconds: int = _TIMEOUT_SECONDS):
    """Run a synchronous function with a hard timeout."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        raise TimeoutError(f"Call timed out after {timeout_seconds}s")
    if error[0]:
        raise error[0]
    return result[0]


@retry(**_RETRY)
def identify_ingredients(image_path: str) -> list[Ingredient]:
    """Call the VLM to identify ingredients — retries on transient failures."""
    t0 = time.monotonic()
    logger.info("ai_service.identify.start", extra={"image": image_path})
    result = _call_with_timeout(vlm.identify_ingredients, image_path)
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
    """Compute nutrition totals — retries on transient failures."""
    t0 = time.monotonic()
    logger.info("ai_service.compute.start")
    result = _call_with_timeout(calculator.compute_totals, ingredients, facts_by_name)
    logger.info(
        "ai_service.compute.done",
        extra={"duration_ms": round((time.monotonic() - t0) * 1000)},
    )
    return result