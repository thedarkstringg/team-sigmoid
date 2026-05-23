import sys

import pytest


def test_cli_history_command_runs_history(monkeypatch):
    import src.cli as cli

    called = {"history": False}

    async def fake_history():
        called["history"] = True

    monkeypatch.setattr(cli, "_history", fake_history)
    monkeypatch.setattr(sys, "argv", ["foodanalyzer", "history"])

    cli.main()

    assert called["history"] is True


def test_cli_analyze_command_passes_image_path(monkeypatch):
    import src.cli as cli

    called = {"image_path": None}

    async def fake_analyze(image_path: str):
        called["image_path"] = image_path

    monkeypatch.setattr(cli, "_analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["foodanalyzer", "analyze", "data/rice_chicken.png"])

    cli.main()

    assert called["image_path"] == "data/rice_chicken.png"


def test_cli_reports_provider_errors_without_traceback(monkeypatch, capsys):
    import src.cli as cli
    from ai.providers.base import ProviderError

    async def fake_analyze(image_path: str):
        raise ProviderError("OPENAI_API_KEY (or LLM_API_KEY) is not set.")

    monkeypatch.setattr(cli, "_analyze", fake_analyze)
    monkeypatch.setattr(sys, "argv", ["foodanalyzer", "analyze", "data/no_meal_blue.png"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "Configuration error" in err
    assert "OPENAI_API_KEY" in err
