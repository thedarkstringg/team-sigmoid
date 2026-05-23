import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_env() -> None:
    database_url = input(
        "Enter DATABASE_URL or press Enter for sqlite default: "
    ).strip()

    if not database_url:
        database_url = "sqlite+aiosqlite:///./foodanalyzer.db"

    os.environ["DATABASE_URL"] = database_url

    usda_api_key = input(
        "Enter USDA_API_KEY or press Enter to skip: "
    ).strip()

    if usda_api_key:
        os.environ["USDA_API_KEY"] = usda_api_key
    else:
        os.environ["USDA_API_KEY"] = ""

    print("Environment configured.")


from ai.schemas import NutritionFacts
from src.services import nutrition_cache


def main() -> None:
    configure_env()
    nutrition_cache.clear()

    fake = NutritionFacts(
        name="rice",
        kcal_per_100g=130.0,
        protein_g_per_100g=2.7,
        carbs_g_per_100g=28.2,
        fat_g_per_100g=0.3,
        source="USDA",
    )

    result1 = nutrition_cache.get("rice")
    print(f"Before set - cache.get('rice'): {result1}")

    nutrition_cache.set("rice", fake, ttl=86400)
    print(f"Cache size after set: {nutrition_cache.size()}")

    result2 = nutrition_cache.get("rice")
    print(f"After set - cache.get('rice'): {result2}")

    print(f"\nCache working: {result2 is not None}")
    print(f"Cache hit returns correct kcal: {result2.kcal_per_100g}")


if __name__ == "__main__":
    main()
