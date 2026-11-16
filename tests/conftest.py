"""Test isolation: never touch the repo-root ``lumi.db`` or the real DB perf log."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_sqlite_db(tmp_path_factory, request):
    """Point persistence at a fresh temporary SQLite DB (and perf log) for every test.

    Runs before a test's own fixtures, so fixtures/tests that set their own DB path still win.
    Module-level ``persistence.DB_PATH = ...`` assignments in test files are overridden here
    for the duration of each test and restored afterwards.
    """
    import persistence

    d = tmp_path_factory.mktemp("lumi_db")
    saved = (persistence.DB_PATH, persistence.DB_PERF_LOG, persistence._DB_INITIALIZED)
    persistence.DB_PATH = d / "lumi.db"
    persistence.DB_PERF_LOG = d / "db_perf_log.jsonl"
    persistence._DB_INITIALIZED = False
    try:
        yield
    finally:
        persistence.DB_PATH, persistence.DB_PERF_LOG, persistence._DB_INITIALIZED = saved
