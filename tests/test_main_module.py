import importlib
import sys

import src.cli as cli


def test_main_module_calls_cli_main(monkeypatch):
    called = {"main": False}

    def fake_main():
        called["main"] = True

    monkeypatch.setattr(cli, "main", fake_main)
    sys.modules.pop("src.__main__", None)

    importlib.import_module("src.__main__")

    assert called["main"] is True
