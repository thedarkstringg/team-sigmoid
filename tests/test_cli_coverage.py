import asyncio
import sys
import types
from types import SimpleNamespace

import src.cli as cli
import src.storage.repository as repository_module


def test_cli_no_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["foodanalyzer"])

    cli.main()

    out = capsys.readouterr().out
    assert "usage:" in out
    assert "analyze" in out
    assert "history" in out


def test_history_prints_no_analyses_message(monkeypatch, capsys):
    class FakeRepo:
        async def init_db(self):
            pass

        async def list_all(self):
            return []

    monkeypatch.setattr(repository_module, "AnalysisRepository", FakeRepo)

    asyncio.run(cli._history())

    assert "No analyses yet." in capsys.readouterr().out


def test_history_prints_saved_records(monkeypatch, capsys):
    class FakeRepo:
        async def init_db(self):
            pass

        async def list_all(self):
            return [
                SimpleNamespace(
                    id=7,
                    timestamp="2026-05-21T12:00:00Z",
                    image_path="data/rice_chicken.png",
                    total_kcal=450,
                    meal_recognized=True,
                )
            ]

    monkeypatch.setattr(repository_module, "AnalysisRepository", FakeRepo)

    asyncio.run(cli._history())

    out = capsys.readouterr().out
    assert "[7]" in out
    assert "data/rice_chicken.png" in out
    assert "450 kcal" in out
    assert "recognized: True" in out


def test_analyze_prints_table_using_mocked_dependencies(monkeypatch, capsys):
    fake_analyzer = types.ModuleType("src.core.analyzer")

    async def fake_run_analysis(image_path):
        return SimpleNamespace(image_path=image_path)

    fake_analyzer.run_analysis = fake_run_analysis
    monkeypatch.setitem(sys.modules, "src.core.analyzer", fake_analyzer)

    class FakeRepo:
        async def init_db(self):
            pass

        async def save(self, record):
            return SimpleNamespace(
                id=1,
                image_path=record.image_path,
                ingredients=[
                    SimpleNamespace(
                        name="rice",
                        estimated_grams=200,
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

    monkeypatch.setattr(repository_module, "AnalysisRepository", FakeRepo)

    asyncio.run(cli._analyze("data/rice_chicken.png"))

    out = capsys.readouterr().out
    assert "Analyzing: data/rice_chicken.png" in out
    assert "rice" in out
    assert "TOTAL" in out
    assert "Meal recognized: True" in out
