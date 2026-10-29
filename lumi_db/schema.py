"""Versioned schema migrations and init_db."""
import logging
import threading
import time

from .pool import get_db_connection, release_db_connection, _commit

logger = logging.getLogger("lumi.persistence")

def migrate_db():
    """Backward-compatible wrapper: applies any pending schema migrations (see init_db)."""
    with _MIGRATION_LOCK:
        return _run_migrations()

_DB_INITIALIZED = False

# One migration run per process at a time; Postgres additionally serialises across processes
# (several app replicas starting together) with a transaction-scoped advisory lock.
_MIGRATION_LOCK = threading.Lock()
_PG_MIGRATION_LOCK_KEY = 0x4C554D49  # "LUMI"
# After a failed run against Postgres, init_db() (called on every rerun) waits this long before
# retrying, so an outage does not add a failing round trip to every page load.
MIGRATION_RETRY_BACKOFF = 60.0
_migration_failed_at = None


def _reset_migration_backoff():
    global _migration_failed_at
    _migration_failed_at = None

# ---------------------------------------------------------------------------
# Versioned schema migrations
# ---------------------------------------------------------------------------
# Every migration must be idempotent: production Postgres already has the
# sessions table (with `pinned`) but no schema_version table.
# The Postgres DDL is mirrored in migrations/postgres.sql.

_SCHEMA_VERSION_DDL = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        name TEXT,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

_SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        filename TEXT,
        cleaning_recipe TEXT,
        step_count INTEGER,
        rules TEXT,
        scanned_columns TEXT,
        user_id TEXT,
        project_name TEXT,
        pinned INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

_PG_ADD_PINNED = "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS pinned INTEGER DEFAULT 0"
_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)",
)

# (version, name). SQL is applied by _apply_migration per dialect.
MIGRATIONS = (
    (1, "create_sessions_table"),
    (2, "add_sessions_pinned_column"),
    (3, "add_sessions_indexes"),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1][0]


def _is_sqlite(conn) -> bool:
    return "sqlite" in type(conn).__module__


def _apply_migration(conn, cursor, version: int):
    sqlite = _is_sqlite(conn)
    if version == 1:
        cursor.execute(_SESSIONS_DDL)
    elif version == 2:
        if sqlite:
            cursor.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cursor.fetchall()]
            if "pinned" not in cols:
                cursor.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0")
        else:
            cursor.execute(_PG_ADD_PINNED)
    elif version == 3:
        for stmt in _INDEX_STATEMENTS:
            cursor.execute(stmt)
    else:
        raise ValueError(f"Unknown migration version {version}")


def _current_schema_version(conn, cursor) -> int:
    """Returns the highest applied version, or 0 if schema_version is missing/empty."""
    try:
        cursor.execute("SELECT MAX(version) AS v FROM schema_version")
        row = cursor.fetchone()
    except Exception:
        # Table does not exist yet (fresh DB or pre-migration DB).
        if not _is_sqlite(conn):
            try:
                conn.rollback()
            except Exception:
                logger.debug("Rollback after missing schema_version failed", exc_info=True)
        return 0
    if row is None:
        return 0
    try:
        val = row["v"]
    except (KeyError, IndexError, TypeError):
        val = row[0]
    return int(val) if val is not None else 0


def _record_version(conn, cursor, version: int, name: str):
    if _is_sqlite(conn):
        cursor.execute("INSERT OR IGNORE INTO schema_version (version, name) VALUES (?, ?)", (version, name))
    else:
        cursor.execute(
            "INSERT INTO schema_version (version, name) VALUES (%s, %s) ON CONFLICT (version) DO NOTHING",
            (version, name),
        )


def _apply_pg_migration_locked(conn, version: int, name: str):
    """Apply one migration on Postgres in its own transaction under a cross-process advisory lock.

    The pool hands out autocommit connections, so autocommit is switched off for the duration
    (and restored afterwards). The version is re-read under the lock: another process may have
    applied it while we waited.
    """
    prev_autocommit = getattr(conn, "autocommit", False)
    try:
        if prev_autocommit:
            conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_MIGRATION_LOCK_KEY,))
        cursor.execute(_SCHEMA_VERSION_DDL)
        if _current_schema_version(conn, cursor) < version:
            _apply_migration(conn, cursor, version)
            _record_version(conn, cursor, version, name)
            logger.info("Applied schema migration %s (%s)", version, name)
        conn.commit()  # also releases the advisory lock
    except Exception:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Rollback after failed migration failed", exc_info=True)
        raise
    finally:
        if prev_autocommit:
            try:
                conn.autocommit = True
            except Exception:
                logger.debug("Could not restore autocommit after migration", exc_info=True)


def _run_migrations() -> bool:
    """Applies pending migrations in order. Returns True if schema is current. Never raises."""
    global _migration_failed_at
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current = _current_schema_version(conn, cursor)
        if current >= LATEST_SCHEMA_VERSION:
            return True
        if _is_sqlite(conn):
            cursor.execute(_SCHEMA_VERSION_DDL)
            for version, name in MIGRATIONS:
                if version <= current:
                    continue
                _apply_migration(conn, cursor, version)
                _record_version(conn, cursor, version, name)
                _commit(conn)
                logger.info("Applied schema migration %s (%s)", version, name)
        else:
            if not getattr(conn, "autocommit", False):
                conn.rollback()  # end the transaction the version probe opened
            for version, name in MIGRATIONS:
                if version > current:
                    _apply_pg_migration_locked(conn, version, name)
        return True
    except Exception:
        logger.error("Database migration/initialization failed", exc_info=True)
        if conn is None or not _is_sqlite(conn):
            _migration_failed_at = time.monotonic()  # remote DB: back off; local SQLite: retry freely
        try:
            conn.rollback()  # conn may be None if acquiring it failed; swallowed below
        except Exception:
            logger.debug("Rollback after failed migration failed", exc_info=True)
        return False
    finally:
        release_db_connection(conn)


def init_db():
    """Brings the database schema up to date (versioned migrations). Runs once per process on success.

    Serialised with a lock; after a failed run against Postgres, retries wait MIGRATION_RETRY_BACKOFF.
    """
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    with _MIGRATION_LOCK:
        if _DB_INITIALIZED:
            return
        if _migration_failed_at is not None and time.monotonic() - _migration_failed_at < MIGRATION_RETRY_BACKOFF:
            return
        if _run_migrations():
            _reset_migration_backoff()
            _DB_INITIALIZED = True
