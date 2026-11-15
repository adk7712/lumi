"""Regression tests: workspace cache off the save thread, guest-row ownership, pool warm-up,
dropped-connection retry, cache sort with datetimes, migration lock/backoff, landing escaping."""
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("LUMI_TESTING", "1")

import psycopg2
import pytest
import streamlit as st

import persistence
import state_manager
from lumi_db import pool as pool_mod, schema as schema_mod, sessions as sessions_mod
from lumi_state import session_io

DB = Path("test_lumi_a40.db")
GUEST = "guest_" + "a" * 32 + "@lumi.ai"
GUEST2 = "guest_" + "b" * 32 + "@lumi.ai"


def _rm():
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            Path(str(DB) + suffix).unlink()
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    monkeypatch.setattr(persistence, "DB_PATH", DB)
    _rm()
    persistence._DB_INITIALIZED = False
    schema_mod._reset_migration_backoff()
    persistence.init_db()
    st.session_state.clear()
    yield
    st.session_state.clear()
    persistence._DB_INITIALIZED = False
    schema_mod._reset_migration_backoff()
    _rm()


def _owner(sid):
    conn = sqlite3.connect(str(DB))
    try:
        row = conn.execute("SELECT user_id FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# 1. Workspace cache is maintained on the main thread; limit uses a real DB count

def test_save_session_on_background_thread_never_touches_session_state():
    assert persistence.get_user_projects("alice@x.com") == []  # cache primed (empty)
    res = []
    t = threading.Thread(target=lambda: res.append(
        persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com", "A")))
    t.start()
    t.join()
    assert res == [True]
    assert st.session_state["_cache_user_projects_alice@x.com"] == []  # untouched off-thread
    # Off the script thread the cache is neither read nor written.
    t = threading.Thread(target=lambda: res.append(persistence.get_user_projects("alice@x.com")))
    t.start()
    t.join()
    assert [p["session_id"] for p in res[1]] == ["s1"]
    assert st.session_state["_cache_user_projects_alice@x.com"] == []


def _prime_session(sid="ws-new", user="alice@x.com", monkeypatch=None):
    import ui_utils
    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: user)
    st.session_state.session_id = sid
    st.session_state.filename = "new.csv"
    st.session_state.cleaning_recipe = [{"action": "drop"}]
    st.session_state.rules = []
    st.session_state.scanned_columns = set()


def test_save_db_session_sync_adds_new_workspace_to_cache(monkeypatch):
    assert persistence.get_user_projects("alice@x.com") == []  # cache primed (empty)
    _prime_session(monkeypatch=monkeypatch)
    state_manager.save_db_session()
    ids = [p["session_id"] for p in st.session_state["_cache_user_projects_alice@x.com"]]
    assert ids == ["ws-new"]
    assert st.session_state["_cache_user_projects_alice@x.com"][0]["step_count"] == 1


def test_save_db_session_async_updates_cache_on_calling_thread(monkeypatch):
    assert persistence.get_user_projects("alice@x.com") == []
    _prime_session(monkeypatch=monkeypatch)
    jobs = []
    monkeypatch.setattr(session_io, "_is_sync_save", lambda: False)
    monkeypatch.setattr(session_io, "_submit_save", lambda sid, job: jobs.append((sid, job)))
    state_manager.save_db_session()
    assert [p["session_id"] for p in st.session_state["_cache_user_projects_alice@x.com"]] == ["ws-new"]
    # The background job itself must not touch st.session_state.
    before = dict(st.session_state)
    t = threading.Thread(target=lambda: jobs[0][1]("ws-new"))
    t.start()
    t.join()
    assert dict(st.session_state).keys() == before.keys()
    assert _owner("ws-new") == "alice@x.com"


def test_rotation_or_failure_invalidates_project_cache(monkeypatch):
    import ui_utils
    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: "alice@x.com")
    st.session_state.session_id = "old"
    st.session_state["_cache_user_projects_alice@x.com"] = [{"session_id": "old"}]
    with session_io._bg_lock:
        session_io._rotated_sessions["old"] = "new"
    monkeypatch.setattr(session_io, "_set_session_cookie", lambda sid: None)
    session_io._apply_rotated_session()
    assert st.session_state.session_id == "new"
    assert "_cache_user_projects_alice@x.com" not in st.session_state


def test_count_user_sessions_uses_db_not_stale_cache():
    assert persistence.get_user_projects("alice@x.com") == []  # stale empty cache
    for i in range(3):
        persistence.save_session(f"s{i}", "a.csv", [], [], [], "alice@x.com")
    st.session_state["_cache_user_projects_alice@x.com"] = []
    assert persistence.count_user_sessions("alice@x.com") == 3
    assert persistence.count_user_sessions(GUEST) == 0
    assert persistence.count_user_sessions(None) == 0


# 2. Guest rows are not bearer records

def test_is_guest_id_requires_generated_format():
    assert persistence.is_guest_id(GUEST)
    assert not persistence.is_guest_id("guest_foo@gmail.com")
    assert not persistence.is_guest_id("guest_" + "a" * 32 + "@lumi.ai.evil.com")
    assert not persistence.is_guest_id(None)


def test_guest_row_overwritten_only_by_same_guest():
    assert persistence.save_session("g", "a.csv", [], [], [], GUEST) is True
    assert persistence.save_session("g", "evil.csv", [], [], [], GUEST2) is False
    assert persistence.save_session("g", "evil.csv", [], [], [], None) is False
    assert _owner("g") == GUEST
    assert persistence.save_session("g", "b.csv", [], [], [], GUEST) is True


def test_registered_user_can_still_claim_guest_row():
    persistence.save_session("g", "a.csv", [], [], [], GUEST)
    assert persistence.load_session("g", "alice@x.com") is not None
    assert persistence.reconcile_session("g", "alice@x.com") is True
    assert _owner("g") == "alice@x.com"
    assert persistence.reconcile_session("g", "bob@x.com") is False


def test_guest_row_not_loadable_by_other_guest_or_anonymous():
    persistence.save_session("g", "a.csv", [], [], [], GUEST)
    assert persistence.load_session("g", None) is None
    assert persistence.load_session("g", GUEST2) is None
    assert persistence.load_session("g", GUEST)["user_id"] == GUEST


def test_registered_email_with_guest_prefix_is_not_a_guest():
    owner = "guest_foo@gmail.com"
    persistence.save_session("r", "a.csv", [], [], [], owner)
    assert persistence.save_session("r", "evil.csv", [], [], [], "alice@x.com") is False
    assert persistence.reconcile_session("r", "alice@x.com") is False
    assert persistence.load_session("r", "alice@x.com") is None
    assert persistence.load_session("r", None) is None
    assert _owner("r") == owner


def test_guest_prefixed_caller_cannot_claim_guest_row():
    persistence.save_session("g", "a.csv", [], [], [], GUEST)
    assert persistence.reconcile_session("g", "guest_foo@gmail.com") is False
    assert _owner("g") == GUEST


def test_null_owned_rows_keep_previous_behaviour():
    persistence.save_session("n", "a.csv", [], [], [], None)
    assert persistence.load_session("n", None) is not None
    assert persistence.save_session("n", "b.csv", [], [], [], None) is True
    assert persistence.reconcile_session("n", "alice@x.com") is True


def test_guest_sql_predicate_per_dialect():
    pg = sessions_mod._guest_owner_sql(mock.MagicMock(), "sessions.user_id")
    assert "~ '^guest_[0-9a-f]{32}@lumi\\.ai$'" in pg
    assert "?" not in pg and "%" not in pg
    lite = sessions_mod._guest_owner_sql(sqlite3.connect(":memory:"), "user_id")
    assert "GLOB" in lite and "?" not in lite


# 3. Pool warm-up holds at most one connection at a time

def test_warm_pool_one_connection_at_a_time(monkeypatch):
    state = {"out": 0, "max_out": 0, "gets": 0}

    class FakePool:
        maxconn = 10

        def getconn(self, timeout=None):
            state["out"] += 1
            state["gets"] += 1
            state["max_out"] = max(state["max_out"], state["out"])
            return object()

        def putconn(self, conn, close=False):
            state["out"] -= 1

    monkeypatch.setattr(pool_mod, "_get_db_url", lambda: "postgresql://fake")
    monkeypatch.setattr(pool_mod, "_get_pg_pool", lambda url: FakePool())
    t = pool_mod._warm_pool_async()
    t.join(5)
    assert state["max_out"] == 1
    assert 1 <= state["gets"] <= 3
    assert state["out"] == 0


# 4. One retry on a connection the server dropped

class _DroppedCursor:
    def execute(self, *a, **k):
        raise psycopg2.OperationalError("server closed the connection unexpectedly")


class _DroppedConn:
    closed = 0

    def cursor(self):
        return _DroppedCursor()

    def rollback(self):
        pass

    def close(self):
        self.closed = 1


def _conn_sequence(monkeypatch, conns):
    it = iter(conns)
    calls = []

    def fake():
        calls.append(1)
        c = next(it)
        if c == "sqlite":
            c = sqlite3.connect(str(DB))
            c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(persistence, "get_db_connection", fake)
    return calls


def test_save_retries_once_on_dropped_connection(monkeypatch):
    calls = _conn_sequence(monkeypatch, [_DroppedConn(), "sqlite"])
    assert persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com") is True
    assert len(calls) == 2
    assert _owner("s1") == "alice@x.com"


def test_load_retries_once_on_dropped_connection(monkeypatch):
    persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com")
    calls = _conn_sequence(monkeypatch, [_DroppedConn(), "sqlite"])
    assert persistence.load_session("s1", "alice@x.com")["filename"] == "a.csv"
    assert len(calls) == 2


def test_refusal_is_not_retried(monkeypatch):
    persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com")
    calls = _conn_sequence(monkeypatch, ["sqlite", "sqlite"])
    assert persistence.save_session("s1", "b.csv", [], [], [], "bob@x.com") is False
    assert len(calls) == 1


def test_second_drop_gives_up(monkeypatch):
    _conn_sequence(monkeypatch, [_DroppedConn(), _DroppedConn()])
    assert persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com") is None
    _conn_sequence(monkeypatch, [_DroppedConn(), _DroppedConn()])
    with pytest.raises(persistence.SessionLoadError):
        persistence.load_session("s1", "alice@x.com")


# 5. updated_at normalised to ISO strings; pin sort never raises

class _RowsCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self.rows


class _RowsConn:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _RowsCursor(self.rows)

    def close(self):
        pass


def test_get_user_projects_normalises_datetime(monkeypatch):
    rows = [{"session_id": "a", "filename": "a.csv", "project_name": "A",
             "updated_at": datetime(2026, 1, 2, 3, 4, 5), "step_count": 0, "pinned": 0}]
    monkeypatch.setattr(persistence, "get_db_connection", lambda: _RowsConn(rows))
    projects = persistence.get_user_projects("alice@x.com")
    assert projects[0]["updated_at"] == "2026-01-02T03:04:05"


def test_toggle_cached_pin_with_mixed_types_does_not_raise():
    st.session_state["_cache_user_projects_alice@x.com"] = [
        {"session_id": "a", "pinned": 0, "updated_at": datetime(2026, 1, 1)},
        {"session_id": "b", "pinned": 0, "updated_at": None},
        {"session_id": "c", "pinned": 0, "updated_at": "2026-01-03T00:00:00"},
    ]
    persistence.toggle_cached_project_pin("alice@x.com", "b")
    cache = st.session_state["_cache_user_projects_alice@x.com"]
    assert cache[0]["session_id"] == "b" and cache[0]["pinned"] == 1


# 6. Migrations: lock, failure backoff, advisory lock per migration on Postgres

class _PgCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        self.conn.log.append((sql.strip().split()[0:3], self.conn.autocommit))
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("boom")

    def fetchone(self):
        return {"v": None}

    def fetchall(self):
        return []


class _PgConn:
    """Not a sqlite3 module object, so schema code takes the Postgres path."""
    def __init__(self, fail_on=None):
        self.autocommit = True
        self.log = []
        self.commits = 0
        self.rollbacks = 0
        self.fail_on = fail_on

    def cursor(self):
        return _PgCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_postgres_migrations_take_advisory_lock_per_migration(monkeypatch):
    conn = _PgConn()
    monkeypatch.setattr(persistence, "get_db_connection", lambda: conn)
    monkeypatch.setattr(persistence, "release_db_connection", lambda c, **k: None)
    persistence._DB_INITIALIZED = False
    persistence.init_db()
    assert persistence._DB_INITIALIZED
    locks = [entry for entry in conn.log if "pg_advisory_xact_lock" in " ".join(entry[0])]
    assert len(locks) == len(persistence.MIGRATIONS)
    assert all(ac is False for _, ac in locks)  # inside an explicit transaction
    assert conn.commits == len(persistence.MIGRATIONS)
    assert conn.autocommit is True  # restored


def test_failed_migration_backs_off(monkeypatch):
    conn = _PgConn(fail_on="pg_advisory_xact_lock")
    gets = []
    monkeypatch.setattr(persistence, "get_db_connection", lambda: gets.append(1) or conn)
    monkeypatch.setattr(persistence, "release_db_connection", lambda c, **k: None)
    persistence._DB_INITIALIZED = False
    persistence.init_db()
    assert not persistence._DB_INITIALIZED
    assert conn.autocommit is True and conn.rollbacks >= 1
    n = len(gets)
    persistence.init_db()
    persistence.init_db()
    assert len(gets) == n  # within backoff window: no retry
    schema_mod._reset_migration_backoff()
    persistence.init_db()
    assert len(gets) > n


def test_concurrent_init_runs_migrations_once(monkeypatch):
    _rm()
    persistence._DB_INITIALIZED = False
    calls = []
    orig = schema_mod._run_migrations
    gate = threading.Event()

    def slow():
        calls.append(1)
        gate.wait(0.2)
        return orig()

    monkeypatch.setattr(schema_mod, "_run_migrations", slow)
    threads = [threading.Thread(target=persistence.init_db) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert calls == [1]
    assert persistence._DB_INITIALIZED


# 7. Landing escapes DB-derived values in raw HTML

def test_landing_escapes_resume_values(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui_utils

    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: None)
    monkeypatch.setattr(ui_utils, "read_session_cookie", lambda *a, **k: None)
    monkeypatch.setattr(persistence, "load_session", lambda sid, user=None: {
        "session_id": sid, "user_id": None, "filename": "<img src=x onerror=alert(1)>.csv",
        "project_name": "<script>alert(2)</script>", "scanned_columns": set(), "step_count": 0,
    })

    def script():
        import streamlit as st
        from views.landing import render_landing_page
        st.session_state.resume_session_id = "sid1"
        render_landing_page()

    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert "<script>alert(2)</script>" not in html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;.csv" in html


# Extra: oversized / zip-bomb XLSX (UnsafeFileError) handled at the two read_tabular callers

def test_load_full_data_turns_unsafe_file_into_value_error(monkeypatch):
    from lumi_state import core
    def boom(*a, **k):
        raise core.UnsafeFileError("This Excel file expands to 900 MB of data")
    monkeypatch.setattr(core, "read_tabular", boom)
    st.session_state.full_source = {"name": "big.xlsx", "bytes": b"PK"}
    with pytest.raises(ValueError, match="900 MB"):
        state_manager.load_full_data()


def test_landing_resume_shows_error_for_unsafe_file(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ui_utils
    from lumi_state import core

    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: None)
    monkeypatch.setattr(ui_utils, "read_session_cookie", lambda *a, **k: None)
    monkeypatch.setattr(persistence, "load_session", lambda sid, user=None: {
        "session_id": sid, "user_id": None, "filename": "big.xlsx", "project_name": "P",
        "scanned_columns": {"a"}, "step_count": 0,
    })

    def boom(*a, **k):
        raise core.UnsafeFileError("This Excel file is compressed far more than a normal spreadsheet")
    monkeypatch.setattr(state_manager, "read_tabular", boom)
    restored = []
    monkeypatch.setattr(state_manager, "load_db_session", lambda *a, **k: restored.append(1) or True)

    class _Upload:
        name = "big.xlsx"

        def seek(self, *a):
            return 0

    real_uploader = st.file_uploader
    monkeypatch.setattr(st, "file_uploader", lambda label, *a, key=None, **k: _Upload() if key == "resume_uploader" else real_uploader(label, *a, key=key, **k))

    def script():
        import streamlit as st
        from views.landing import render_landing_page
        st.session_state.resume_session_id = "sid1"
        render_landing_page()

    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("compressed far more" in e.value for e in at.error)
    assert not [b for b in at.button if b.key == "proceed_anyway_btn"]
    assert restored == []
