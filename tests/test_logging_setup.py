import logging

from src import logging_setup


def test_configure_logging_sets_root_and_third_party_levels(monkeypatch):
    monkeypatch.setattr(logging_setup.settings, "log_level", "DEBUG")

    logging_setup.configure_logging()

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING


def test_configure_logging_falls_back_to_info_for_invalid_level(monkeypatch):
    monkeypatch.setattr(logging_setup.settings, "log_level", "NOT_A_REAL_LEVEL")

    logging_setup.configure_logging()

    assert logging.getLogger().level == logging.INFO
