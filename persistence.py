"""Persistence facade.

The implementation lives in the ``lumi_db`` package (pool / schema / sessions). This module
re-exports every public and private name so ``import persistence`` / ``from persistence import x``
keep working, and so tests can monkeypatch them.

Monkeypatching: assigning an attribute on this module (``persistence.DB_PATH = ...``,
``mock.patch("persistence.get_db_connection")``, ``persistence._apply_migration = ...``) is
propagated to every ``lumi_db`` submodule that defines that name, so the override takes effect
inside the implementation. Mutable state (``DB_PATH``, ``DB_PERF_LOG``, ``_DB_INITIALIZED``) is
not copied here; reads are resolved live from the owning submodule.
"""
from lumi_db import pool as _pool, schema as _schema, sessions as _sessions
from lumi_db.pool import (  # noqa: F401
    logger, _perf_lock, _log_db_perf, _get_cached_db_url, _get_db_url, get_env, storage_warning,
    _commit, BoundedPgPool, DatabaseUnavailableError, POOL_WAIT_TIMEOUT, _get_pg_pool, _warm_pool_async, db_cursor, get_db_connection,
    release_db_connection, _execute, run_with_retry,
)
from lumi_db.schema import (  # noqa: F401
    migrate_db, _SCHEMA_VERSION_DDL, _SESSIONS_DDL, _PG_ADD_PINNED, _INDEX_STATEMENTS,
    MIGRATIONS, LATEST_SCHEMA_VERSION, _is_sqlite, _apply_migration, _current_schema_version,
    _run_migrations, init_db, MIGRATION_RETRY_BACKOFF, _reset_migration_backoff,
)
from lumi_db.sessions import (  # noqa: F401
    clear_user_projects_cache, update_cached_project, remove_cached_project,
    toggle_cached_project_pin, save_session, load_session, SessionLoadError, get_user_projects,
    count_user_sessions, delete_session, rename_session, toggle_pin_session, reconcile_session,
    is_guest_id,
)

from facade_util import install_facade

install_facade(__name__, (_pool, _schema, _sessions), live=("DB_PATH", "DB_PERF_LOG", "_DB_INITIALIZED"))
