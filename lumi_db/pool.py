"""Connection acquisition: URL discovery, bounded PG pool, SQLite fallback, perf log."""
import json
import logging
import time
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("lumi.persistence")

DB_PATH = Path(__file__).parent.parent / "lumi.db"
DB_PERF_LOG = Path(__file__).parent.parent / ".lumi_cache" / "db_perf_log.jsonl"
_perf_lock = threading.Lock()

def _log_db_perf(op: str, duration_ms: float, details: dict = None, error: str = None):
    try:
        DB_PERF_LOG.parent.mkdir(exist_ok=True)
        rec = {
            "timestamp": time.time(),
            "iso": datetime.now().isoformat(),
            "op": op,
            "duration_ms": round(duration_ms, 2),
            "thread": threading.current_thread().name,
            "is_main_thread": threading.current_thread() is threading.main_thread(),
            "details": details or {},
            "error": str(error) if error else None
        }
        with _perf_lock:
            with open(DB_PERF_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception:
        logger.debug("Failed to write DB perf log", exc_info=True)

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from psycopg2.pool import PoolError

def _lookup_db_url(is_testing: bool):
    if is_testing:
        return None

    import os
    import streamlit as st
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")

    def _find_url(sec):
        if not isinstance(sec, (dict, st.secrets.__class__)):
            return None
        if "DB_URL" in sec:
            return sec["DB_URL"]
        if "DATABASE_URL" in sec:
            return sec["DATABASE_URL"]
        for k in sec:
            try:
                val = sec[k]
                if isinstance(val, (dict, st.secrets.__class__)):
                    res = _find_url(val)
                    if res:
                        return res
            except Exception:
                logger.debug("Skipping unreadable secrets entry while searching for DB URL", exc_info=True)
        return None

    if not db_url:
        try:
            if hasattr(st, "secrets"):
                db_url = _find_url(st.secrets)
        except Exception:
            logger.debug("Could not read DB URL from st.secrets", exc_info=True)

    if not db_url:
        try:
            secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                with open(secrets_path, "rb") as f:
                    sec = tomllib.load(f)
                    db_url = _find_url(sec)
        except Exception:
            logger.warning("Could not parse .streamlit/secrets.toml for a DB URL", exc_info=True)

    if db_url and ("postgresql://" in db_url or "postgres://" in db_url):
        params_to_add = []
        if "connect_timeout" not in db_url:
            params_to_add.append("connect_timeout=5")
        if "keepalives" not in db_url:
            params_to_add.append("keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5")
        if params_to_add:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}{'&'.join(params_to_add)}"

    return db_url

_db_url_cache: dict = {}
_db_url_miss_at: dict = {}
_DB_URL_MISS_TTL = 30.0  # seconds a missing URL is remembered before re-reading secrets

def _get_cached_db_url(is_testing: bool):
    """Cache a found URL for good; remember a miss only briefly so secrets edits apply."""
    cached = _db_url_cache.get(is_testing)
    if cached:
        return cached
    missed_at = _db_url_miss_at.get(is_testing)
    if missed_at is not None and time.monotonic() - missed_at < _DB_URL_MISS_TTL:
        return None
    db_url = _lookup_db_url(is_testing)
    if db_url:
        _db_url_cache[is_testing] = db_url
        _db_url_miss_at.pop(is_testing, None)
    else:
        _db_url_miss_at[is_testing] = time.monotonic()
    return db_url

def _clear_db_url_cache():
    _db_url_cache.clear()
    _db_url_miss_at.clear()

_get_cached_db_url.cache_clear = _clear_db_url_cache

def _get_db_url():
    import streamlit as st
    import os
    
    is_testing = os.getenv("LUMI_TESTING") == "1"
    try:
        if hasattr(st, "session_state") and st.session_state.get("_is_testing", False):
            is_testing = True
    except Exception:
        pass

    return _get_cached_db_url(is_testing)

def get_env():
    """Return "production" or "development" (default).

    Reads env var LUMI_ENV, else the top-level secret ``env``.
    """
    import os
    val = os.getenv("LUMI_ENV")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get("env")
        except Exception:
            val = None
    return "production" if str(val or "").strip().lower() == "production" else "development"


def storage_warning():
    """Return a user-facing warning if production runs without a persistent DB, else None.

    SQLite on Streamlit Community Cloud is ephemeral (wiped on restart/redeploy).
    Never fires under LUMI_TESTING=1.
    """
    import os
    if os.getenv("LUMI_TESTING") == "1":
        return None
    if get_env() != "production":
        return None
    if _get_db_url():
        return None
    msg = ("LUMI_ENV=production but no DB_URL/DATABASE_URL is configured: falling back to "
           "ephemeral SQLite. Saved sessions and accounts will be lost on restart. "
           "Set DB_URL in the app secrets.")
    logger.error(msg)
    return msg

import streamlit as st
from contextlib import contextmanager

def _commit(conn):
    """Safely commits transaction based on database engine and driver state."""
    if conn is None:
        return
    if "sqlite" in type(conn).__module__:
        conn.commit()
    elif not getattr(conn, "autocommit", False):
        conn.commit()

POOL_WAIT_TIMEOUT = 3.0  # seconds a caller waits for a free pooled connection
_CHECKOUT_ATTEMPTS = 2  # fresh connections tried before getconn gives up
POOL_FAILURE_TTL = 20.0  # seconds a failed pool construction is remembered (fail fast during outages)

class DatabaseUnavailableError(RuntimeError):
    """The configured Postgres database cannot be reached (and silently using SQLite would lose data)."""

_fallback_warned = False
_fallback_lock = threading.Lock()

def _note_pool_failure(e):
    """Log a dev-mode SQLite fallback once per outage (reset when the pool works again)."""
    global _fallback_warned
    with _fallback_lock:
        first = not _fallback_warned
        _fallback_warned = True
    if first:
        logger.warning("PostgreSQL pool unavailable; falling back to local SQLite (development only): %s", e)

def _note_pool_success():
    global _fallback_warned
    if _fallback_warned:
        with _fallback_lock:
            _fallback_warned = False

class BoundedPgPool:
    """Thread-safe PostgreSQL connection pool with semaphore queueing and autocommit."""
    def __init__(self, raw_pool, maxconn=10, timeout=POOL_WAIT_TIMEOUT):
        self.raw_pool = raw_pool
        self.maxconn = maxconn
        self.timeout = timeout
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(maxconn)
        self._used = set()

    def _discard(self, conn):
        """Close a connection and drop it from the raw pool without ever raising."""
        try:
            self.raw_pool.putconn(conn, close=True)
            return
        except Exception:
            logger.debug("raw_pool.putconn(close=True) failed while discarding", exc_info=True)
        try:
            conn.close()
        except Exception:
            logger.debug("conn.close() failed while discarding", exc_info=True)

    def _checkout(self):
        """Get a healthy, autocommit connection from the raw pool or raise PoolError.

        Closed/broken connections are discarded and replaced (bounded retries).
        """
        last_err = None
        for _ in range(_CHECKOUT_ATTEMPTS):
            conn = self.raw_pool.getconn()
            if getattr(conn, "closed", 0) != 0:
                last_err = "pooled connection was closed"
                self._discard(conn)
                continue
            try:
                if hasattr(conn, "autocommit"):
                    status = getattr(conn, "get_transaction_status", lambda: 0)()
                    if status != 0:
                        conn.rollback()
                    conn.autocommit = True
            except Exception as e:
                logger.warning("Discarding pooled connection that could not be prepared: %s", e)
                last_err = str(e)
                self._discard(conn)
                continue
            return conn
        raise PoolError(f"Could not obtain a healthy pooled connection: {last_err}")

    def getconn(self, timeout=None):
        timeout = timeout or self.timeout
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise PoolError(f"Connection pool timeout after {timeout}s: all {self.maxconn} connections busy")

        ok = False
        try:
            conn = self._checkout()
            with self._lock:
                self._used.add(conn)
            ok = True
            return conn
        finally:
            if not ok:
                self._semaphore.release()

    def putconn(self, conn, close=False):
        was_used = False
        with self._lock:
            if conn in self._used:
                self._used.remove(conn)
                was_used = True
        try:
            if not close:
                if getattr(conn, "closed", 0) != 0:
                    close = True
                else:
                    try:
                        if getattr(conn, "get_transaction_status", lambda: 0)() != 0:
                            conn.rollback()
                    except Exception:
                        logger.warning("Rollback failed while returning connection; discarding it", exc_info=True)
                        close = True
            try:
                self.raw_pool.putconn(conn, close=close)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    logger.debug("conn.close() failed after putconn error", exc_info=True)
                raise
        finally:
            if was_used:
                self._semaphore.release()

    def closeall(self):
        if hasattr(self.raw_pool, "closeall"):
            self.raw_pool.closeall()

@st.cache_resource
def _build_pg_pool(db_url):
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor
    raw_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=db_url, cursor_factory=RealDictCursor)
    return BoundedPgPool(raw_pool, maxconn=10, timeout=POOL_WAIT_TIMEOUT)

_pool_failures: dict = {}  # db_url -> (monotonic time, error string)
_pool_failures_lock = threading.Lock()

def _get_pg_pool(db_url):
    """Cached pool; a failed construction is remembered for POOL_FAILURE_TTL so callers fail fast."""
    with _pool_failures_lock:
        failure = _pool_failures.get(db_url)
    if failure is not None:
        failed_at, msg = failure
        if time.monotonic() - failed_at < POOL_FAILURE_TTL:
            raise PoolError(f"PostgreSQL pool unavailable (recent failure): {msg}")
    try:
        pool = _build_pg_pool(db_url)
    except Exception as e:
        with _pool_failures_lock:
            _pool_failures[db_url] = (time.monotonic(), str(e))
        raise
    if failure is not None:
        with _pool_failures_lock:
            _pool_failures.pop(db_url, None)
    return pool

def _clear_pg_pool_cache():
    _build_pg_pool.clear()
    with _pool_failures_lock:
        _pool_failures.clear()

_get_pg_pool.clear = _clear_pg_pool_cache

_WARM_CONNECTIONS = 2  # connections opened by the warm-up thread (one at a time)

def _warm_pool_async():
    """Asynchronously opens a few pooled connections to cut cold-start TLS latency for users.

    Each connection is checked out and returned immediately, so the warmer never holds more
    than one slot and first user requests are not starved behind it (POOL_WAIT_TIMEOUT).
    """
    def _warm():
        try:
            url = _get_db_url()
            if url:
                pool = _get_pg_pool(url)
                for _ in range(min(_WARM_CONNECTIONS, getattr(pool, "maxconn", _WARM_CONNECTIONS))):
                    try:
                        c = pool.getconn(timeout=2.0)
                    except Exception:
                        break
                    pool.putconn(c)
        except Exception:
            logger.warning("Connection pool warm-up failed", exc_info=True)

    t = threading.Thread(target=_warm, daemon=True, name="LumiPoolWarmer")
    t.start()
    return t

# Pre-warm connection pool in background thread on module load
_warm_pool_async()

@contextmanager
def db_cursor(commit: bool = False):
    """Context manager for obtaining a database cursor with guaranteed connection release and error rollback."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        yield conn, cursor
        if commit:
            _commit(conn)
    except Exception:
        if commit:
            try:
                conn.rollback()
            except Exception:
                logger.warning("Rollback failed in db_cursor", exc_info=True)
        raise
    finally:
        release_db_connection(conn)

def get_db_connection():
    """Establishes and returns a connection to the database.
    
    If DB_URL is present in st.secrets or .streamlit/secrets.toml (and not in testing mode),
    it borrows a connection from the PostgreSQL connection pool. Otherwise it uses SQLite.
    If the pool fails, production raises DatabaseUnavailableError (no silent data loss);
    development falls back to SQLite with a one-time warning.
    """
    t0 = time.perf_counter()
    backend = "sqlite"
    err = None
    try:
        db_url = _get_db_url()
        if db_url:
            backend = "postgresql"
            try:
                pool = _get_pg_pool(db_url)
                conn = pool.getconn()
                _note_pool_success()
                return conn
            except Exception as e:
                err = str(e)
                if get_env() == "production":
                    backend = "postgresql_unavailable"
                    logger.error("PostgreSQL pool unavailable in production; refusing SQLite fallback: %s", e)
                    raise DatabaseUnavailableError(f"Database unavailable: {e}") from e
                backend = "sqlite_fallback"
                _note_pool_failure(e)
        
        # Fallback to local SQLite database
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn
    finally:
        t1 = time.perf_counter()
        _log_db_perf("get_db_connection", (t1 - t0) * 1000, {"backend": backend}, err)

def release_db_connection(conn, close: bool = False):
    """Releases a connection back to the pool or closes it depending on the driver.

    ``close=True`` discards a pooled connection instead of reusing it (e.g. the server dropped it).
    """
    if conn is None:
        return
    import sqlite3
    if isinstance(conn, sqlite3.Connection):
        conn.close()
    else:
        db_url = _get_db_url()
        if db_url:
            try:
                pool = _get_pg_pool(db_url)
                pool.putconn(conn, close=close)
            except Exception as e:
                logger.warning("Failed to release connection back to pool; closing it: %s", e)
                try:
                    conn.close()
                except Exception:
                    logger.warning("Failed to close connection", exc_info=True)
        else:
            try:
                conn.close()
            except Exception:
                logger.warning("Failed to close connection", exc_info=True)

def _execute(conn, cursor, sql: str, params: tuple = ()):
    """Helper to execute SQL queries using correct placeholders based on database engine."""
    if "sqlite" not in type(conn).__module__:
        # PostgreSQL uses %s instead of ?
        sql = sql.replace("?", "%s")
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)


# A connection the server dropped (restart, idle timeout, network blip) surfaces only when used.
_RETRYABLE_DB_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

def run_with_retry(op: str, fn):
    """Run ``fn(conn)`` on a fresh connection; retry once on a new one if the server dropped it.

    Only driver-level connection errors are retried. Results (including refusals such as a
    rejected upsert) are returned as-is, and any other exception propagates on the first try.
    ``fn`` must be safe to repeat (an idempotent write or a read).
    """
    for attempt in range(2):
        conn = get_db_connection()  # may raise DatabaseUnavailableError
        broken = False
        try:
            return fn(conn)
        except _RETRYABLE_DB_ERRORS as e:
            broken = True
            if attempt:
                raise
            logger.warning("%s: database connection dropped (%s); retrying on a fresh connection", op, e)
        finally:
            if broken:
                release_db_connection(conn, close=True)
            else:
                release_db_connection(conn)
