"""Versioned schema migrations and init_db."""
import logging

from .pool import get_db_connection, release_db_connection, _commit

logger = logging.getLogger("lumi.persistence")

def migrate_db():
    """Backward-compatible wrapper: applies any pending schema migrations (see init_db)."""
    return _run_migrations()

_DB_INITIALIZED = False

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


def _run_migrations() -> bool:
    """Applies pending migrations in order. Returns True if schema is current. Never raises."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        current = _current_schema_version(conn, cursor)
        if current >= LATEST_SCHEMA_VERSION:
            return True
        cursor.execute(_SCHEMA_VERSION_DDL)
        for version, name in MIGRATIONS:
            if version <= current:
                continue
            _apply_migration(conn, cursor, version)
            if _is_sqlite(conn):
                cursor.execute("INSERT OR IGNORE INTO schema_version (version, name) VALUES (?, ?)", (version, name))
            else:
                cursor.execute(
                    "INSERT INTO schema_version (version, name) VALUES (%s, %s) ON CONFLICT (version) DO NOTHING",
                    (version, name),
                )
            _commit(conn)
            logger.info("Applied schema migration %s (%s)", version, name)
        return True
    except Exception:
        logger.error("Database migration/initialization failed", exc_info=True)
        try:
            conn.rollback()  # conn may be None if acquiring it failed; swallowed below
        except Exception:
            logger.debug("Rollback after failed migration failed", exc_info=True)
        return False
    finally:
        release_db_connection(conn)


def init_db():
    """Brings the database schema up to date (versioned migrations). Runs once per process on success."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    if _run_migrations():
        _DB_INITIALIZED = True
