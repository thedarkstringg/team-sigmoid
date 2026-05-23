import asyncio
import logging
import argparse
import sys

from ai.providers.base import ProviderError
from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="foodanalyzer")
    sub = parser.add_subparsers(dest="command")

    analyze_p = sub.add_parser("analyze", help="Analyze a meal image")
    analyze_p.add_argument("image_path", help="Path to JPEG or PNG image")

    sub.add_parser("history", help="Show recent analyses")

    args = parser.parse_args()

    try:
        if args.command == "analyze":
            asyncio.run(_analyze(args.image_path))
        elif args.command == "history":
            asyncio.run(_history())
        else:
            parser.print_help()
    except ProviderError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print(
            "Set the required API key in .env or your shell, then try again. "
            "For the default OpenAI/OpenRouter provider, set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


async def _analyze(image_path: str) -> None:
    from src.core.analyzer import run_analysis
    from src.storage.repository import AnalysisRepository

    repo = AnalysisRepository()
    await repo.init_db()

    record = await run_analysis(image_path)
    saved = await repo.save(record)

    print(f"\nAnalyzing: {image_path}\n")
    print(f"{'ingredient':<25} {'g':>6} {'kcal':>6} {'protein':>8} {'carbs':>6} {'fat':>6}")
    print("-" * 60)
    for ing in saved.ingredients:
        print(f"{ing.name:<25} {ing.estimated_grams:>6.0f} {ing.kcal:>6.0f} {ing.protein:>8.1f} {ing.carbs:>6.1f} {ing.fat:>6.1f}")
    print("-" * 60)
    print(f"{'TOTAL':<25} {'':>6} {saved.total_kcal:>6.0f} {saved.total_protein:>8.1f} {saved.total_carbs:>6.1f} {saved.total_fat:>6.1f}")
    print(f"\nMeal recognized: {saved.meal_recognized}")


async def _history() -> None:
    from src.storage.repository import AnalysisRepository

    repo = AnalysisRepository()
    await repo.init_db()
    records = await repo.list_all()

    if not records:
        print("No analyses yet.")
        return

    for r in records:
        print(f"[{r.id}] {r.timestamp} | {r.image_path} | {r.total_kcal:.0f} kcal | recognized: {r.meal_recognized}")


if __name__ == "__main__":
    main()
