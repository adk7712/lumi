"""Session repository (save/load/delete/rename/pin/reconcile) and per-rerun project cache."""
import json
import logging
import re
import threading
import time
from datetime import datetime

from .pool import get_db_connection, release_db_connection, run_with_retry, _execute, _commit, _log_db_perf

logger = logging.getLogger("lumi.persistence")

# Exact shape of ids minted by ui_utils for anonymous guests: f"guest_{uuid4().hex}@lumi.ai".
# Only these rows are claimable by a registered user; a real account whose email merely starts
# with "guest_" is a registered owner like any other.
_GUEST_ID_RE = re.compile(r"guest_[0-9a-f]{32}@lumi\.ai")
_GUEST_GLOB = "guest_" + "[0-9a-f]" * 32 + "@lumi.ai"


def is_guest_id(user_id) -> bool:
    """True only for an app-generated anonymous guest id."""
    return bool(user_id) and _GUEST_ID_RE.fullmatch(str(user_id)) is not None


def _is_registered(user_id) -> bool:
    """A signed-in account (mirrors ui_utils.is_authenticated_user: any "guest_" prefix is not)."""
    return bool(user_id) and not str(user_id).startswith("guest_")


def _guest_owner_sql(conn, col: str) -> str:
    """SQL predicate: ``col`` holds an app-generated guest id (GLOB on SQLite, regex on Postgres)."""
    if "sqlite" in type(conn).__module__:
        return f"{col} GLOB '{_GUEST_GLOB}'"
    return f"{col} ~ '^guest_[0-9a-f]{{32}}@lumi\\.ai$'"


def _iso(value):
    """updated_at as an ISO string (Postgres returns datetime, SQLite and the cache use strings)."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _sort_projects(projects: list):
    """Pinned first, then newest; never raises on mixed/missing updated_at values."""
    try:
        projects.sort(key=lambda x: (int(x.get("pinned") or 0), str(_iso(x.get("updated_at")) or "")), reverse=True)
    except Exception:
        logger.debug("Could not re-sort cached projects", exc_info=True)

def _owns_session_state() -> bool:
    """True when this thread may use st.session_state: the script thread or the main thread.

    Background save workers (lumi_state.session_io) have no ScriptRunContext; there
    st.session_state is not the user's session, so cache helpers must be no-ops.
    """
    if threading.current_thread() is threading.main_thread():
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False

def clear_user_projects_cache(user_id: str = None):
    """Helper to clear per-rerun user projects cache upon workspace mutations."""
    import streamlit as st
    if hasattr(st, "session_state") and _owns_session_state():
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
    if not hasattr(st, "session_state") or not _owns_session_state():
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
        _sort_projects(projects)

def remove_cached_project(user_id: str, session_id: str):
    """Optimistically removes a project from the session_state cache in-place."""
    if not user_id or user_id.startswith("guest_"):
        return
    import streamlit as st
    if not hasattr(st, "session_state") or not _owns_session_state():
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
    if not hasattr(st, "session_state") or not _owns_session_state():
        return
    cache_key = f"_cache_user_projects_{user_id}"
    if cache_key in st.session_state:
        for p in st.session_state[cache_key]:
            if p.get("session_id") == session_id:
                p["pinned"] = 0 if p.get("pinned", 0) == 1 else 1
                p["updated_at"] = datetime.now().isoformat()
                break
        _sort_projects(st.session_state[cache_key])

class SessionLoadError(RuntimeError):
    """``load_session`` could not reach or read the database (not the same as "not found")."""


def save_session(session_id: str, filename: str, recipe: list, rules: list, scanned_columns: list, user_id: str = None, project_name: str = None):
    """Saves or updates the session details in the database.

    Returns True if the row was written, False if refused (the row belongs to someone else), or
    None if a database or serialization error occurred. Callers rotate the session id only on
    False: an outage must not split one workspace into two rows.

    On the background save thread it never touches ``st.session_state``; there the caller
    (``save_db_session``) updates the sidebar cache on the script thread instead.
    """
    t0 = time.perf_counter()
    err = None
    saved = False
    try:
        # Serialize fields to JSON
        recipe_json = json.dumps(recipe)
        rules_json = json.dumps(rules)
        scanned_columns_json = json.dumps(list(scanned_columns))
        step_count = len(recipe)
        now = datetime.now().isoformat()
        caller_registered = 1 if _is_registered(user_id) else 0

        def _upsert(conn):
            cursor = conn.cursor()
            # Single atomic UPSERT. DO UPDATE only applies when the row is unowned, owned by the
            # caller, or owned by an app-generated guest and the caller is a registered user
            # (the guest -> account claim). Another guest or an anonymous caller cannot overwrite
            # a guest's row.
            _execute(conn, cursor, f"""
                INSERT INTO sessions (session_id, filename, cleaning_recipe, step_count, rules, scanned_columns, user_id, project_name, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, ?, 'Untitled Workspace'), ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    filename = excluded.filename,
                    cleaning_recipe = excluded.cleaning_recipe,
                    step_count = excluded.step_count,
                    rules = excluded.rules,
                    scanned_columns = excluded.scanned_columns,
                    user_id = COALESCE(excluded.user_id, sessions.user_id),
                    project_name = COALESCE(?, sessions.project_name),
                    updated_at = excluded.updated_at
                WHERE sessions.user_id IS NULL
                   OR sessions.user_id = excluded.user_id
                   OR ({_guest_owner_sql(conn, "sessions.user_id")} AND ? = 1)
            """, (session_id, filename, recipe_json, step_count, rules_json, scanned_columns_json, user_id,
                  project_name, filename, now, project_name, caller_registered))
            ok = cursor.rowcount > 0
            _commit(conn)
            return ok

        saved = run_with_retry("save_session", _upsert)
        if not saved:
            logger.warning("Refused to overwrite session %s: owned by a different user", session_id)
            return False
        # No-op on the background save thread (see _owns_session_state); save_db_session updates
        # the cache on the script thread instead.
        update_cached_project(user_id, session_id, filename=filename, step_count=step_count, updated_at=now,
                              **({"project_name": project_name} if project_name else {}))
        return True
    except Exception as e:
        err = str(e)
        logger.error("Error saving session %s: %s", session_id, e, exc_info=True)
        return None
    finally:
        _log_db_perf("save_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "step_count": len(recipe), "saved": saved}, err)

def _may_read(db_user_id, current_user_email) -> bool:
    """Owner check for load_session.

    Unowned rows (anonymous saves): readable by whoever holds the id, which only travels in the
    HMAC-signed session cookie/URL. Guest rows: the same guest, or a registered user claiming it
    after sign-in (app.py reconciles right after). Registered rows: the owner only.
    """
    if db_user_id is None or db_user_id == current_user_email:
        return True
    return is_guest_id(db_user_id) and _is_registered(current_user_email)

def load_session(session_id: str, current_user_email: str = None) -> dict:
    """Loads a session's details from the database.

    Returns None if the session does not exist or the caller may not read it (see ``_may_read``).
    Raises ``SessionLoadError`` on a database error, so callers keep the resume cookie.
    """
    t0 = time.perf_counter()
    err = None

    def _select(conn):
        cursor = conn.cursor()
        _execute(conn, cursor, "SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row or not _may_read(row["user_id"], current_user_email):
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
            "updated_at": _iso(row["updated_at"])
        }

    try:
        return run_with_retry("load_session", _select)
    except Exception as e:
        err = str(e)
        logger.error("Error loading session %s: %s", session_id, e, exc_info=True)
        raise SessionLoadError(str(e)) from e
    finally:
        _log_db_perf("load_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id}, err)

def get_user_projects(user_id: str) -> list:
    """Retrieves all sessions/projects belonging to a specific user ordered by pinned status and last update."""
    if not user_id or user_id.startswith("guest_"):
        return []
        
    import streamlit as st
    cache_key = f"_cache_user_projects_{user_id}"
    owns_state = hasattr(st, "session_state") and _owns_session_state()
    if owns_state and cache_key in st.session_state:
        return st.session_state[cache_key]

    t0 = time.perf_counter()
    err = None
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
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
                "updated_at": _iso(row["updated_at"]),
                "step_count": row["step_count"],
                "pinned": pinned_val
            })
        if owns_state:
            st.session_state[cache_key] = projects
        return projects
    except Exception as e:
        err = str(e)
        logger.error("Error getting user projects for %s: %s", user_id, e, exc_info=True)
        return []
    finally:
        release_db_connection(conn)
        _log_db_perf("get_user_projects", (time.perf_counter() - t0) * 1000, {"user_id": user_id}, err)

def count_user_sessions(user_id: str) -> int:
    """Returns the number of workspaces owned by a user, counted in the database.

    Used to enforce the workspace limit, so it never trusts the per-rerun project cache (which
    can miss a workspace whose background save has not been reflected yet).
    """
    if not user_id or user_id.startswith("guest_"):
        return 0

    def _count(conn):
        cursor = conn.cursor()
        _execute(conn, cursor, "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row is None:
            return 0
        try:
            return int(row["n"])
        except (KeyError, IndexError, TypeError):
            return int(row[0])

    try:
        return run_with_retry("count_user_sessions", _count)
    except Exception as e:
        logger.error("Error counting sessions for %s: %s", user_id, e, exc_info=True)
        import streamlit as st
        cached = st.session_state.get(f"_cache_user_projects_{user_id}") if hasattr(st, "session_state") and _owns_session_state() else None
        return len(cached or [])

def delete_session(session_id: str, user_id: str) -> bool:
    """Deletes a workspace session if owned by user_id."""
    if not session_id or not user_id or user_id.startswith("guest_"):
        return False
    t0 = time.perf_counter()
    err = None
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
        cursor = conn.cursor()
        _execute(conn, cursor, "DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        _commit(conn)
        success = cursor.rowcount > 0
        if success:
            remove_cached_project(user_id, session_id)
        return success
    except Exception as e:
        err = str(e)
        logger.error("Error deleting session %s: %s", session_id, e, exc_info=True)
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
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
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
        logger.error("Error renaming session %s: %s", session_id, e, exc_info=True)
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
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
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
        logger.error("Error toggling pin status for session %s: %s", session_id, e, exc_info=True)
        return False
    finally:
        release_db_connection(conn)
        _log_db_perf("toggle_pin_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "user_id": user_id}, err)

def reconcile_session(session_id: str, user_id: str) -> bool:
    """Reconciles an anonymous session by assigning it to a logged-in user.

    Unowned sessions can be claimed by anyone; app-generated guest sessions only by a registered
    user; sessions owned by another registered user are left untouched. Returns True if the
    session now belongs to user_id.
    """
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, f"""
            UPDATE sessions 
            SET user_id = ?, updated_at = ?
            WHERE session_id = ?
              AND (user_id IS NULL OR user_id = ? OR ({_guest_owner_sql(conn, "user_id")} AND ? = 1))
        """, (user_id, now, session_id, user_id, 1 if _is_registered(user_id) else 0))
        updated = cursor.rowcount > 0
        _commit(conn)
        if not updated:
            logger.info("Session %s not reconciled to %s (missing or owned by another user)", session_id, user_id)
        return updated
    except Exception as e:
        logger.error("Error reconciling session %s to user %s: %s", session_id, user_id, e, exc_info=True)
        return False
    finally:
        release_db_connection(conn)
