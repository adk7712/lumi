import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "lumi.db"

def get_db_connection():
    """Establishes and returns a connection to the database.
    
    If DB_URL is present in st.secrets or .streamlit/secrets.toml (and not in testing mode),
    it connects to PostgreSQL using psycopg2. Otherwise, it falls back to a local SQLite database.
    """
    import streamlit as st
    import os
    
    is_testing = os.getenv("LUMI_TESTING") == "1"
    try:
        if hasattr(st, "session_state") and st.session_state.get("_is_testing", False):
            is_testing = True
    except Exception:
        pass

    db_url = None

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

    if not is_testing:
        db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
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

    if db_url and not is_testing:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            return conn
        except Exception as e:
            print(f"Warning: Failed to connect to PostgreSQL (DB_URL configured). Falling back to SQLite. Error: {e}")
    
    # Fallback to local SQLite database
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _execute(conn, cursor, sql: str, params: tuple = ()):
    """Helper to execute SQL queries using correct placeholders based on database engine."""
    if "sqlite" not in type(conn).__module__:
        # PostgreSQL uses %s instead of ?
        sql = sql.replace("?", "%s")
    cursor.execute(sql, params)

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
        conn.commit()
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        conn.close()

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

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if "sqlite" in type(conn).__module__:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            if cursor.fetchone():
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
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()

def save_session(session_id: str, filename: str, recipe: list, rules: list, scanned_columns: list, user_id: str = None, project_name: str = None):
    """Saves or updates the session details in the database."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Serialize fields to JSON
        recipe_json = json.dumps(recipe)
        rules_json = json.dumps(rules)
        scanned_columns_json = json.dumps(list(scanned_columns))
        step_count = len(recipe)
        
        # Check if session exists to preserve project_name/user_id/created_at
        _execute(conn, cursor, "SELECT project_name, user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            db_project_name = row["project_name"]
            db_user_id = row["user_id"]
        else:
            db_project_name = None
            db_user_id = None
            
        final_project_name = project_name or db_project_name or filename or "Untitled Workspace"
            
        final_user_id = user_id or db_user_id
        now = datetime.now().isoformat()
        
        _execute(conn, cursor, """
            INSERT INTO sessions (session_id, filename, cleaning_recipe, step_count, rules, scanned_columns, user_id, project_name, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                filename = excluded.filename,
                cleaning_recipe = excluded.cleaning_recipe,
                step_count = excluded.step_count,
                rules = excluded.rules,
                scanned_columns = excluded.scanned_columns,
                user_id = COALESCE(excluded.user_id, sessions.user_id),
                project_name = COALESCE(excluded.project_name, sessions.project_name),
                updated_at = excluded.updated_at
        """, (session_id, filename, recipe_json, step_count, rules_json, scanned_columns_json, final_user_id, final_project_name, now))
        
        conn.commit()
        clear_user_projects_cache(final_user_id)
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")
    finally:
        conn.close()

def load_session(session_id: str, current_user_email: str = None) -> dict:
    """Loads a session's details from the database."""
    init_db()
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
        print(f"Error loading session {session_id}: {e}")
        return None
    finally:
        conn.close()

def get_user_projects(user_id: str) -> list:
    """Retrieves all sessions/projects belonging to a specific user ordered by pinned status and last update."""
    if not user_id or user_id.startswith("guest_"):
        return []
        
    import streamlit as st
    cache_key = f"_cache_user_projects_{user_id}"
    if hasattr(st, "session_state") and cache_key in st.session_state:
        return st.session_state[cache_key]

    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        _execute(conn, cursor, "SELECT * FROM sessions WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC", (user_id,))
        rows = cursor.fetchall()
        projects = []
        for row in rows:
            projects.append({
                "session_id": row["session_id"],
                "filename": row["filename"],
                "project_name": row["project_name"],
                "updated_at": row["updated_at"],
                "step_count": row["step_count"],
                "pinned": row["pinned"] if "pinned" in row.keys() else 0
            })
        if hasattr(st, "session_state"):
            st.session_state[cache_key] = projects
        return projects
    except Exception as e:
        print(f"Error getting user projects for {user_id}: {e}")
        return []
    finally:
        conn.close()

def count_user_sessions(user_id: str) -> int:
    """Returns the total number of workspaces owned by a user."""
    if not user_id or user_id.startswith("guest_"):
        return 0
    return len(get_user_projects(user_id))

def delete_session(session_id: str, user_id: str) -> bool:
    """Deletes a workspace session if owned by user_id."""
    if not session_id or not user_id or user_id.startswith("guest_"):
        return False
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        _execute(conn, cursor, "DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        conn.commit()
        clear_user_projects_cache(user_id)
        return True
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        return False
    finally:
        conn.close()

def rename_session(session_id: str, new_name: str, user_id: str) -> bool:
    """Renames a workspace session if owned by user_id."""
    if not session_id or not new_name or not user_id or user_id.startswith("guest_"):
        return False
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, "UPDATE sessions SET project_name = ?, updated_at = ? WHERE session_id = ? AND user_id = ?", (new_name, now, session_id, user_id))
        conn.commit()
        clear_user_projects_cache(user_id)
        return True
    except Exception as e:
        print(f"Error renaming session {session_id}: {e}")
        return False
    finally:
        conn.close()

def toggle_pin_session(session_id: str, user_id: str) -> bool:
    """Toggles the pinned status of a workspace session if owned by user_id."""
    if not session_id or not user_id or user_id.startswith("guest_"):
        return False
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        _execute(conn, cursor, "SELECT pinned FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        row = cursor.fetchone()
        if row:
            curr_pinned = row["pinned"] if "pinned" in row.keys() else 0
            new_pinned = 0 if curr_pinned else 1
            now = datetime.now().isoformat()
            _execute(conn, cursor, "UPDATE sessions SET pinned = ?, updated_at = ? WHERE session_id = ? AND user_id = ?", (new_pinned, now, session_id, user_id))
            conn.commit()
            clear_user_projects_cache(user_id)
            return True
        return False
    except Exception as e:
        print(f"Error toggling pin status for session {session_id}: {e}")
        return False
    finally:
        conn.close()

def reconcile_session(session_id: str, user_id: str):
    """Reconciles an anonymous session by assigning it to a logged-in user."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, """
            UPDATE sessions 
            SET user_id = ?, updated_at = ?
            WHERE session_id = ?
        """, (user_id, now, session_id))
        conn.commit()
    except Exception as e:
        print(f"Error reconciling session {session_id} to user {user_id}: {e}")
    finally:
        conn.close()
