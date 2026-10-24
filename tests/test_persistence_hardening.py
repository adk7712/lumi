import sys
import sqlite3
import unittest
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import persistence

HARDENING_DB = Path("test_lumi_hardening.db")
persistence.DB_PATH = HARDENING_DB


def _rm():
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            Path(str(HARDENING_DB) + suffix).unlink()
        except FileNotFoundError:
            pass


class _Base(unittest.TestCase):
    def setUp(self):
        persistence.DB_PATH = HARDENING_DB
        persistence._DB_INITIALIZED = False
        _rm()

    def tearDown(self):
        _rm()

    def _q(self, sql, params=()):
        conn = sqlite3.connect(str(HARDENING_DB))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


class TestOwnerCheck(_Base):
    def setUp(self):
        super().setUp()
        persistence.init_db()

    def _owner(self, sid):
        return self._q("SELECT user_id FROM sessions WHERE session_id=?", (sid,))[0][0]

    def test_cross_user_overwrite_refused(self):
        self.assertTrue(persistence.save_session("s1", "a.csv", [{"a": 1}], [], [], "alice@x.com", "Alice"))
        ok = persistence.save_session("s1", "evil.csv", [], [], [], "bob@x.com", "Hacked")
        self.assertFalse(ok)
        row = self._q("SELECT filename, user_id, project_name, step_count FROM sessions WHERE session_id='s1'")[0]
        self.assertEqual(row, ("a.csv", "alice@x.com", "Alice", 1))

    def test_anonymous_overwrite_of_owned_row_refused(self):
        persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com")
        self.assertFalse(persistence.save_session("s1", "b.csv", [], [], [], None))
        self.assertFalse(persistence.save_session("s1", "b.csv", [], [], [], "guest_1"))
        self.assertEqual(self._owner("s1"), "alice@x.com")

    def test_same_user_allowed(self):
        persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com")
        self.assertTrue(persistence.save_session("s1", "b.csv", [{"x": 1}], [], [], "alice@x.com"))
        self.assertEqual(self._q("SELECT filename, step_count FROM sessions")[0], ("b.csv", 1))

    def test_guest_to_user_claim_allowed(self):
        persistence.save_session("s1", "a.csv", [], [], [], "guest_abc")
        self.assertTrue(persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com"))
        self.assertEqual(self._owner("s1"), "alice@x.com")

    def test_reconcile_guest_ok_and_foreign_refused(self):
        persistence.save_session("g", "a.csv", [], [], [], "guest_abc")
        self.assertTrue(persistence.reconcile_session("g", "alice@x.com"))
        self.assertEqual(self._owner("g"), "alice@x.com")
        self.assertFalse(persistence.reconcile_session("g", "bob@x.com"))
        self.assertEqual(self._owner("g"), "alice@x.com")

    def test_cache_not_updated_when_refused(self):
        import streamlit as st
        st.session_state.clear()
        persistence.save_session("s1", "a.csv", [], [], [], "alice@x.com", "Alice")
        persistence.save_session("s1", "a.csv", [], [], [], "bob@x.com", "Bob")
        persistence.get_user_projects("bob@x.com")  # populate bob's (empty) cache
        persistence.save_session("s1", "evil.csv", [], [], [], "bob@x.com", "Hacked")
        self.assertEqual(st.session_state.get("_cache_user_projects_bob@x.com"), [])
        st.session_state.clear()


class TestMigrations(_Base):
    def _make_old_db(self):
        conn = sqlite3.connect(str(HARDENING_DB))
        conn.execute("""CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, filename TEXT, cleaning_recipe TEXT, step_count INTEGER,
            rules TEXT, scanned_columns TEXT, user_id TEXT, project_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("INSERT INTO sessions (session_id, filename, cleaning_recipe, step_count, user_id) VALUES ('old','o.csv','[]',0,'u@x.com')")
        conn.commit()
        conn.close()

    def _cols(self):
        return [r[1] for r in self._q("PRAGMA table_info(sessions)")]

    def test_upgrades_old_schema_without_pinned(self):
        self._make_old_db()
        self.assertNotIn("pinned", self._cols())
        persistence.init_db()
        self.assertIn("pinned", self._cols())
        self.assertEqual(self._q("SELECT session_id, pinned FROM sessions"), [("old", 0)])
        self.assertEqual(persistence.toggle_pin_session("old", "u@x.com"), True)

    def test_schema_version_recorded(self):
        persistence.init_db()
        versions = [r[0] for r in self._q("SELECT version FROM schema_version ORDER BY version")]
        self.assertEqual(versions, [v for v, _ in persistence.MIGRATIONS])
        self.assertEqual(versions[-1], persistence.LATEST_SCHEMA_VERSION)

    def test_indexes_exist(self):
        persistence.init_db()
        names = {r[0] for r in self._q("SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertIn("idx_sessions_user_id", names)
        self.assertIn("idx_sessions_updated_at", names)

    def test_idempotent(self):
        self._make_old_db()
        persistence.init_db()
        persistence._DB_INITIALIZED = False
        persistence.init_db()
        self.assertTrue(persistence.migrate_db())
        self.assertEqual(len(self._q("SELECT * FROM schema_version")), len(persistence.MIGRATIONS))
        self.assertEqual(self._cols().count("pinned"), 1)

    def test_fast_path_skips_work_when_current(self):
        persistence.init_db()
        persistence._DB_INITIALIZED = False
        calls = []
        orig = persistence._apply_migration
        persistence._apply_migration = lambda *a, **k: calls.append(a)
        try:
            persistence.init_db()
        finally:
            persistence._apply_migration = orig
        self.assertEqual(calls, [])

    def test_preexisting_current_schema_without_version_table(self):
        # Mimics prod: sessions table with pinned, no schema_version.
        conn = sqlite3.connect(str(HARDENING_DB))
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, filename TEXT, cleaning_recipe TEXT, step_count INTEGER, rules TEXT, scanned_columns TEXT, user_id TEXT, project_name TEXT, pinned INTEGER DEFAULT 0, created_at TIMESTAMP, updated_at TIMESTAMP)")
        conn.commit()
        conn.close()
        persistence.init_db()
        self.assertEqual(len(self._q("SELECT * FROM schema_version")), 3)

    def test_postgres_sql_file_mirrors_migrations(self):
        sql = (Path(__file__).parent.parent / "migrations" / "postgres.sql").read_text()
        for stmt in (persistence._PG_ADD_PINNED, *persistence._INDEX_STATEMENTS):
            self.assertIn(stmt, sql)
        for v, name in persistence.MIGRATIONS:
            self.assertIn(f"({v}, '{name}')", sql)

    def test_init_failure_logged_at_error(self):
        orig = persistence._apply_migration
        def boom(*a, **k):
            raise RuntimeError("boom")
        persistence._apply_migration = boom
        try:
            with self.assertLogs("lumi.persistence", level="ERROR"):
                persistence.init_db()
        finally:
            persistence._apply_migration = orig
        self.assertFalse(persistence._DB_INITIALIZED)


if __name__ == "__main__":
    unittest.main()


def test_db_url_miss_is_not_cached_but_hit_is(monkeypatch):
    from lumi_db import pool

    pool._get_cached_db_url.cache_clear()
    calls = []
    results = iter([None, "postgresql://u:p@h/db", "postgresql://other@h/db"])

    def fake_lookup(is_testing):
        calls.append(is_testing)
        return next(results)

    monkeypatch.setattr(pool, "_lookup_db_url", fake_lookup)
    try:
        assert pool._get_cached_db_url(False) is None
        # a miss is remembered briefly: no re-lookup within the TTL
        assert pool._get_cached_db_url(False) is None
        assert len(calls) == 1
        monkeypatch.setattr(pool, "_DB_URL_MISS_TTL", 0.0)
        assert pool._get_cached_db_url(False) == "postgresql://u:p@h/db"
        # found URL is cached: no further lookup
        assert pool._get_cached_db_url(False) == "postgresql://u:p@h/db"
        assert len(calls) == 2
    finally:
        pool._get_cached_db_url.cache_clear()
