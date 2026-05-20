import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from src.models import AnalysisRecord, IngredientRow
from src.storage import repository as repository_module


def test_repository_save_and_list_all_roundtrip(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_food_analyzer.db").replace("\\", "/")
    db_url = f"sqlite+aiosqlite:///{db_path}"

    monkeypatch.setattr(
        repository_module,
        "settings",
        SimpleNamespace(database_url=db_url),
    )

    async def scenario():
        repo = repository_module.AnalysisRepository()
        try:
            await repo.init_db()

            record = AnalysisRecord(
                timestamp=datetime.now(timezone.utc),
                image_path="data/rice_chicken.png",
                ingredients=[
                    IngredientRow(
                        name="rice",
                        estimated_grams=200,
                        confidence=0.9,
                        kcal=260,
                        protein=5.4,
                        carbs=56,
                        fat=0.6,
                    )
                ],
                total_kcal=260,
                total_protein=5.4,
                total_carbs=56,
                total_fat=0.6,
                meal_recognized=True,
            )

            saved = await repo.save(record)
            records = await repo.list_all()

            assert saved.id is not None
            assert len(records) == 1
            assert records[0].id == saved.id
            assert records[0].image_path == "data/rice_chicken.png"
            assert records[0].ingredients[0].name == "rice"
            assert records[0].total_kcal == 260
            assert records[0].meal_recognized is True
        finally:
            await repo._engine.dispose()

    asyncio.run(scenario())
