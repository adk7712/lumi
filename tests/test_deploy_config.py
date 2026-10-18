import logging

import pytest

import persistence


@pytest.fixture
def clean(monkeypatch):
    for k in ("LUMI_ENV", "DB_URL", "DATABASE_URL", "LUMI_TESTING"):
        monkeypatch.delenv(k, raising=False)
    persistence._get_cached_db_url.cache_clear()
    monkeypatch.setattr(persistence, "_get_db_url", lambda: None)
    yield monkeypatch
    persistence._get_cached_db_url.cache_clear()


def test_default_env_is_development(clean):
    assert persistence.get_env() == "development"
    assert persistence.storage_warning() is None


def test_env_var_production_case_insensitive(clean):
    clean.setenv("LUMI_ENV", " Production ")
    assert persistence.get_env() == "production"


def test_unknown_env_value_is_development(clean):
    clean.setenv("LUMI_ENV", "staging")
    assert persistence.get_env() == "development"


def test_production_without_db_warns_and_logs_error(clean, caplog):
    clean.setenv("LUMI_ENV", "production")
    with caplog.at_level(logging.ERROR, logger="lumi.persistence"):
        msg = persistence.storage_warning()
    assert msg and "DB_URL" in msg
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_production_with_db_no_warning(clean):
    clean.setenv("LUMI_ENV", "production")
    clean.setattr(persistence, "_get_db_url", lambda: "postgresql://u:p@h:6543/db")
    assert persistence.storage_warning() is None


def test_testing_mode_never_warns(clean):
    clean.setenv("LUMI_ENV", "production")
    clean.setenv("LUMI_TESTING", "1")
    assert persistence.storage_warning() is None
