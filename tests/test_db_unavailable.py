"""get_db_connection must not silently fall back to SQLite in production when Postgres fails."""
import logging
import sqlite3

import pytest

import persistence
from lumi_db import pool as pool_mod

URL = "postgresql://u:p@localhost:5432/db"


class _FailingPool:
    def getconn(self, *a, **k):
        raise pool_mod.PoolError("Connection pool timeout after 3.0s")


@pytest.fixture
def pg_down(tmp_path, monkeypatch):
    db = tmp_path / "fallback.db"
    monkeypatch.setattr(persistence, "DB_PATH", db)
    monkeypatch.setattr(pool_mod, "DB_PATH", db)
    monkeypatch.setattr(pool_mod, "_get_db_url", lambda: URL)
    monkeypatch.setattr(pool_mod, "_get_pg_pool", lambda url: _FailingPool())
    monkeypatch.setattr(pool_mod, "_log_db_perf", lambda *a, **k: None)
    monkeypatch.setattr(pool_mod, "_fallback_warned", False)
    return db


def test_timeout_constant():
    assert persistence.POOL_WAIT_TIMEOUT == 3.0


def test_production_raises_and_writes_nothing(pg_down, monkeypatch):
    monkeypatch.setenv("LUMI_ENV", "production")
    with pytest.raises(persistence.DatabaseUnavailableError):
        persistence.get_db_connection()
    assert not pg_down.exists()


def test_production_db_errors_are_not_refusals_or_misses(pg_down, monkeypatch):
    monkeypatch.setenv("LUMI_ENV", "production")
    # None = error (retry later); False is reserved for "owned by another user".
    assert persistence.save_session("s1", "f.csv", [], [], set(), user_id="u@x.com") is None
    # A failed lookup must not look like "session not found" (callers would delete the cookie).
    with pytest.raises(persistence.SessionLoadError):
        persistence.load_session("s1", "u@x.com")
    assert not pg_down.exists()


def test_development_falls_back_with_single_warning(pg_down, monkeypatch, caplog):
    monkeypatch.setenv("LUMI_ENV", "development")
    with caplog.at_level(logging.WARNING, logger="lumi.persistence"):
        for _ in range(3):
            conn = persistence.get_db_connection()
            assert isinstance(conn, sqlite3.Connection)
            conn.close()
    warns = [r for r in caplog.records if "falling back to local SQLite" in r.getMessage()]
    assert len(warns) == 1
    assert pg_down.exists()


def test_no_url_uses_sqlite_even_in_production(tmp_path, monkeypatch):
    db = tmp_path / "plain.db"
    monkeypatch.setattr(persistence, "DB_PATH", db)
    monkeypatch.setattr(pool_mod, "_get_db_url", lambda: None)
    monkeypatch.setattr(pool_mod, "_log_db_perf", lambda *a, **k: None)
    monkeypatch.setenv("LUMI_ENV", "production")
    conn = persistence.get_db_connection()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
