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


def test_secrets_example_top_level_keys_parse_at_top_level():
    """Top-level secrets must sit above the first [table] header, or TOML nests them."""
    import tomllib
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml.example"
    text = example.read_text(encoding="utf-8")
    with open(example, "rb") as f:
        data = tomllib.load(f)

    for key in ("env", "dev_auth", "DB_URL"):
        assert key in data, f"{key} must be a top-level key in secrets.toml.example"
    assert "DB_URL" not in data["auth"]
    assert "DB_URL" not in data["auth"]["google"]

    # Commented-out POSTHOG_API_KEY placeholder must also live above the first table header.
    first_header = next(i for i, line in enumerate(text.splitlines()) if line.startswith("["))
    posthog_line = next(i for i, line in enumerate(text.splitlines()) if "POSTHOG_API_KEY" in line)
    assert posthog_line < first_header


def test_inject_posthog_finds_key_nested_under_table():
    """A key placed below [auth.google] (old template order) is still found, with a warning."""
    from unittest.mock import MagicMock
    import ui_utils

    st_obj = MagicMock()
    st_obj.secrets = {"auth": {"google": {"POSTHOG_API_KEY": "phc_test"}}}
    ui_utils.inject_posthog(st_obj)
    st_obj.html.assert_called_once()
    assert "phc_test" in st_obj.html.call_args[0][0]


def test_inject_posthog_skips_without_key(monkeypatch):
    from unittest.mock import MagicMock
    import ui_utils

    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    st_obj = MagicMock()
    st_obj.secrets = {"auth": {"redirect_uri": "x"}}
    ui_utils.inject_posthog(st_obj)
    st_obj.html.assert_not_called()
