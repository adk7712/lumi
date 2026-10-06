import json
import time
import threading
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "lumi.db"
DB_PERF_LOG = Path(__file__).parent / ".lumi_cache" / "db_perf_log.jsonl"
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
        pass

import functools
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from psycopg2.pool import PoolError

@functools.lru_cache(maxsize=2)
def _get_cached_db_url(is_testing: bool):
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
                pass
        return None

    if not db_url:
        try:
            if hasattr(st, "secrets"):
                db_url = _find_url(st.secrets)
        except Exception:
            pass

    if not db_url:
        try:
            secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                with open(secrets_path, "rb") as f:
                    sec = tomllib.load(f)
                    db_url = _find_url(sec)
        except Exception:
            pass

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

class BoundedPgPool:
    """Thread-safe PostgreSQL connection pool with semaphore queueing and autocommit."""
    def __init__(self, raw_pool, maxconn=10, timeout=10.0):
        self.raw_pool = raw_pool
        self.maxconn = maxconn
        self.timeout = timeout
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(maxconn)
        self._used = set()

    def getconn(self, timeout=None):
        timeout = timeout or self.timeout
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise PoolError(f"Connection pool timeout after {timeout}s: all {self.maxconn} connections busy")
        
        try:
            conn = self.raw_pool.getconn()
            if getattr(conn, "closed", 0) != 0:
                try:
                    self.raw_pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self.raw_pool.getconn()
            if hasattr(conn, "autocommit"):
                conn.autocommit = True
            with self._lock:
                self._used.add(conn)
            return conn
        except Exception:
            self._semaphore.release()
            raise

    def putconn(self, conn, close=False):
        was_used = False
        with self._lock:
            if conn in self._used:
                self._used.remove(conn)
                was_used = True
        try:
            if not getattr(conn, "autocommit", False) and getattr(conn, "get_transaction_status", lambda: 0)() != 0:
                try:
                    conn.rollback()
                except Exception:
                    pass
            self.raw_pool.putconn(conn, close=close)
        finally:
            if was_used:
                self._semaphore.release()

    def closeall(self):
        if hasattr(self.raw_pool, "closeall"):
            self.raw_pool.closeall()

@st.cache_resource
def _get_pg_pool(db_url):
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor
    raw_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=db_url, cursor_factory=RealDictCursor)
    return BoundedPgPool(raw_pool, maxconn=10, timeout=10.0)

def _warm_pool_async():
    """Asynchronously establishes initial connections to eliminate cold-start TLS latency for users."""
    def _warm():
        try:
            url = _get_db_url()
            if url:
                pool = _get_pg_pool(url)
                conns = []
                try:
                    max_to_warm = getattr(pool, "maxconn", 10)
                    for _ in range(max_to_warm):
                        try:
                            c = pool.getconn(timeout=2.0)
                            conns.append(c)
                        except Exception:
                            break
                finally:
                    for c in conns:
                        pool.putconn(c)
        except Exception:
            pass

    t = threading.Thread(target=_warm, daemon=True, name="LumiPoolWarmer")
    t.start()

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
                pass
        raise
    finally:
        release_db_connection(conn)

def get_db_connection():
    """Establishes and returns a connection to the database.
    
    If DB_URL is present in st.secrets or .streamlit/secrets.toml (and not in testing mode),
    it borrows a connection from the PostgreSQL connection pool. Otherwise, it falls back to SQLite.
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
                return conn
            except Exception as e:
                err = str(e)
                print(f"Warning: Failed to get connection from PostgreSQL pool. Falling back to SQLite. Error: {e}")
                backend = "sqlite_fallback"
        
        # Fallback to local SQLite database
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn
    finally:
        t1 = time.perf_counter()
        _log_db_perf("get_db_connection", (t1 - t0) * 1000, {"backend": backend}, err)

def release_db_connection(conn):
    """Releases a connection back to the pool or closes it depending on the driver."""
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
                pool.putconn(conn)
            except Exception as e:
                print(f"Warning: Failed to release connection back to pool. Closing connection. Error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            try:
                conn.close()
            except Exception:
                pass

def _execute(conn, cursor, sql: str, params: tuple = ()):
    """Helper to execute SQL queries using correct placeholders based on database engine."""
    if "sqlite" not in type(conn).__module__:
        # PostgreSQL uses %s instead of ?
        sql = sql.replace("?", "%s")
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)

def migrate_db():
    """Ensures existing tables have necessary new columns added safely."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if "sqlite" in type(conn).__module__:
            cursor.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cursor.fetchall()]
            if "pinned" not in cols:
                cursor.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0")
        else:
            cursor.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS pinned INTEGER DEFAULT 0")
        _commit(conn)
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        release_db_connection(conn)

_DB_INITIALIZED = False

def clear_user_projects_cache(user_id: str = None):
    """Helper to clear per-rerun user projects cache upon workspace mutations."""
    import streamlit as st
    if hasattr(st, "session_state"):
        if user_id:
            st.session_state.pop(f"_cache_user_projects_{user_id}", None)
        else:
            keys_to_del = [k for k in st.session_state.keys() if str(k).startswith("_cache_user_projects_")]
            for k in keys_to_del:
                st.session_state.pop(k, None)

def update_cached_project(user_id: str, session_id: str, **updates):
    """Optimistically updates a project in the session_state cache in-place without triggering WAN refetch."""
    if not user_id or user_id.startswith("guest_"):
        return
    import streamlit as st
    if not hasattr(st, "session_state"):
        return
    cache_key = f"_cache_user_projects_{user_id}"
    if cache_key in st.session_state:
        projects = st.session_state[cache_key]
        found = False
        for p in projects:
            if p.get("session_id") == session_id:
                p.update(updates)
                found = True
                break
        if not found and "filename" in updates:
            new_proj = {
                "session_id": session_id,
                "filename": updates.get("filename", ""),
                "project_name": updates.get("project_name") or updates.get("filename", "Untitled Workspace"),
                "updated_at": updates.get("updated_at", datetime.now().isoformat()),
                "step_count": updates.get("step_count", 0),
                "pinned": updates.get("pinned", 0)
            }
            projects.insert(0, new_proj)
        try:
            projects.sort(key=lambda x: (x.get("pinned", 0), x.get("updated_at", "")), reverse=True)
        except Exception:
            pass

def remove_cached_project(user_id: str, session_id: str):
    """Optimistically removes a project from the session_state cache in-place."""
    if not user_id or user_id.startswith("guest_"):
        return
    import streamlit as st
    if not hasattr(st, "session_state"):
        return
    cache_key = f"_cache_user_projects_{user_id}"
    if cache_key in st.session_state:
        st.session_state[cache_key] = [
            p for p in st.session_state[cache_key] if p.get("session_id") != session_id
        ]

def toggle_cached_project_pin(user_id: str, session_id: str):
    """Optimistically toggles pinned status and re-sorts cache in-place."""
    if not user_id or user_id.startswith("guest_"):
        return
    import streamlit as st
    if not hasattr(st, "session_state"):
        return
    cache_key = f"_cache_user_projects_{user_id}"
    if cache_key in st.session_state:
        for p in st.session_state[cache_key]:
            if p.get("session_id") == session_id:
                p["pinned"] = 0 if p.get("pinned", 0) == 1 else 1
                p["updated_at"] = datetime.now().isoformat()
                break
        st.session_state[cache_key].sort(
            key=lambda x: (x.get("pinned", 0), x.get("updated_at", "")),
            reverse=True
        )

def init_db():
    """Initializes the database schema if tables do not exist."""
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if "sqlite" in type(conn).__module__:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            if cursor.fetchone():
                _DB_INITIALIZED = True
                return
            cursor.execute("""
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
            """)
            cursor.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cursor.fetchall()]
            if "pinned" not in cols:
                cursor.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0")
        else:
            cursor.execute("""
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
            """)
            cursor.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS pinned INTEGER DEFAULT 0")
        conn.commit()
        _DB_INITIALIZED = True
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        release_db_connection(conn)

def save_session(session_id: str, filename: str, recipe: list, rules: list, scanned_columns: list, user_id: str = None, project_name: str = None):
    """Saves or updates the session details in the database."""
    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Serialize fields to JSON
        recipe_json = json.dumps(recipe)
        rules_json = json.dumps(rules)
        scanned_columns_json = json.dumps(list(scanned_columns))
        step_count = len(recipe)
        now = datetime.now().isoformat()
        
        # Single atomic UPSERT - eliminates redundant SELECT round-trip over WAN
        _execute(conn, cursor, """
            INSERT INTO sessions (session_id, filename, cleaning_recipe, step_count, rules, scanned_columns, user_id, project_name, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, ?, 'Untitled Workspace'), ?)
            ON CONFLICT(session_id) DO UPDATE SET
                filename = excluded.filename,
                cleaning_recipe = excluded.cleaning_recipe,
                step_count = excluded.step_count,
                rules = excluded.rules,
                scanned_columns = excluded.scanned_columns,
                user_id = COALESCE(?, sessions.user_id),
                project_name = COALESCE(?, sessions.project_name),
                updated_at = excluded.updated_at
        """, (session_id, filename, recipe_json, step_count, rules_json, scanned_columns_json, user_id, project_name, filename, now, user_id, project_name))
        
        _commit(conn)
        update_cached_project(
            user_id,
            session_id,
            filename=filename,
            project_name=project_name or filename,
            step_count=step_count,
            updated_at=now
        )
    except Exception as e:
        err = str(e)
        print(f"Error saving session {session_id}: {e}")
    finally:
        release_db_connection(conn)
        _log_db_perf("save_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "step_count": len(recipe)}, err)

def load_session(session_id: str, current_user_email: str = None) -> dict:
    """Loads a session's details from the database."""
    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        _execute(conn, cursor, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            db_user_id = row["user_id"]
            # Security: If owned by a real registered user, restrict access to owner only.
            # Guest sessions (starting with "guest_") remain claimable during login transition.
            if db_user_id is not None and not db_user_id.startswith("guest_"):
                if db_user_id != current_user_email:
                    return None
            return {
                "session_id": row["session_id"],
                "filename": row["filename"],
                "cleaning_recipe": json.loads(row["cleaning_recipe"]),
                "step_count": row["step_count"],
                "rules": json.loads(row["rules"]) if row["rules"] else [],
                "scanned_columns": set(json.loads(row["scanned_columns"])) if row["scanned_columns"] else set(),
                "user_id": row["user_id"],
                "project_name": row["project_name"],
                "pinned": row["pinned"] if "pinned" in row.keys() else 0,
                "updated_at": row["updated_at"]
            }
        return None
    except Exception as e:
        err = str(e)
        print(f"Error loading session {session_id}: {e}")
        return None
    finally:
        release_db_connection(conn)
        _log_db_perf("load_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id}, err)

def get_user_projects(user_id: str) -> list:
    """Retrieves all sessions/projects belonging to a specific user ordered by pinned status and last update."""
    if not user_id or user_id.startswith("guest_"):
        return []
        
    import streamlit as st
    cache_key = f"_cache_user_projects_{user_id}"
    if hasattr(st, "session_state") and cache_key in st.session_state:
        return st.session_state[cache_key]

    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Pruned column list to avoid downloading large recipes and rules blobs over WAN
        _execute(conn, cursor, "SELECT session_id, filename, project_name, updated_at, step_count, pinned FROM sessions WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC", (user_id,))
        rows = cursor.fetchall()
        projects = []
        for row in rows:
            pinned_val = 0
            if isinstance(row, dict):
                pinned_val = row.get("pinned", 0)
            elif hasattr(row, "keys") and "pinned" in row.keys():
                pinned_val = row["pinned"]
            elif hasattr(row, "__getitem__"):
                try:
                    pinned_val = row["pinned"]
                except Exception:
                    pinned_val = 0
            projects.append({
                "session_id": row["session_id"],
                "filename": row["filename"],
                "project_name": row["project_name"],
                "updated_at": row["updated_at"],
                "step_count": row["step_count"],
                "pinned": pinned_val
            })
        if hasattr(st, "session_state"):
            st.session_state[cache_key] = projects
        return projects
    except Exception as e:
        err = str(e)
        print(f"Error getting user projects for {user_id}: {e}")
        return []
    finally:
        release_db_connection(conn)
        _log_db_perf("get_user_projects", (time.perf_counter() - t0) * 1000, {"user_id": user_id}, err)

def count_user_sessions(user_id: str) -> int:
    """Returns the total number of workspaces owned by a user."""
    if not user_id or user_id.startswith("guest_"):
        return 0
    return len(get_user_projects(user_id))

def delete_session(session_id: str, user_id: str) -> bool:
    """Deletes a workspace session if owned by user_id."""
    if not session_id or not user_id or user_id.startswith("guest_"):
        return False
    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        _execute(conn, cursor, "DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        _commit(conn)
        success = cursor.rowcount > 0
        if success:
            remove_cached_project(user_id, session_id)
        return success
    except Exception as e:
        err = str(e)
        print(f"Error deleting session {session_id}: {e}")
        return False
    finally:
        release_db_connection(conn)
        _log_db_perf("delete_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "user_id": user_id}, err)

def rename_session(session_id: str, new_name: str, user_id: str) -> bool:
    """Renames a workspace session if owned by user_id."""
    if not session_id or not new_name or not user_id or user_id.startswith("guest_"):
        return False
    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, "UPDATE sessions SET project_name = ?, updated_at = ? WHERE session_id = ? AND user_id = ?", (new_name, now, session_id, user_id))
        _commit(conn)
        success = cursor.rowcount > 0
        if success:
            update_cached_project(user_id, session_id, project_name=new_name, updated_at=now)
        return success
    except Exception as e:
        err = str(e)
        print(f"Error renaming session {session_id}: {e}")
        return False
    finally:
        release_db_connection(conn)
        _log_db_perf("rename_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "user_id": user_id}, err)

def toggle_pin_session(session_id: str, user_id: str) -> bool:
    """Toggles the pinned status of a workspace session if owned by user_id in a single atomic update."""
    if not session_id or not user_id or user_id.startswith("guest_"):
        return False
    t0 = time.perf_counter()
    err = None
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, """
            UPDATE sessions 
            SET pinned = CASE WHEN COALESCE(pinned, 0) = 1 THEN 0 ELSE 1 END,
                updated_at = ?
            WHERE session_id = ? AND user_id = ?
        """, (now, session_id, user_id))
        _commit(conn)
        success = cursor.rowcount > 0
        if success:
            toggle_cached_project_pin(user_id, session_id)
        return success
    except Exception as e:
        err = str(e)
        print(f"Error toggling pin status for session {session_id}: {e}")
        return False
    finally:
        release_db_connection(conn)
        _log_db_perf("toggle_pin_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "user_id": user_id}, err)

def reconcile_session(session_id: str, user_id: str):
    """Reconciles an anonymous session by assigning it to a logged-in user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, """
            UPDATE sessions 
            SET user_id = ?, updated_at = ?
            WHERE session_id = ?
        """, (user_id, now, session_id))
        _commit(conn)
    except Exception as e:
        print(f"Error reconciling session {session_id} to user {user_id}: {e}")
    finally:
        release_db_connection(conn)
