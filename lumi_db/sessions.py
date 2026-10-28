"""Session repository (save/load/delete/rename/pin/reconcile) and per-rerun project cache."""
import json
import logging
import time
from datetime import datetime

from .pool import get_db_connection, release_db_connection, _execute, _commit, _log_db_perf

logger = logging.getLogger("lumi.persistence")

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
            logger.debug("Could not re-sort cached projects", exc_info=True)

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

class SessionLoadError(RuntimeError):
    """``load_session`` could not reach or read the database (not the same as "not found")."""


def save_session(session_id: str, filename: str, recipe: list, rules: list, scanned_columns: list, user_id: str = None, project_name: str = None):
    """Saves or updates the session details in the database.

    Returns True if the row was written, False if refused (the row is owned by a different
    registered user), or None if a database or serialization error occurred. Callers rotate the
    session id only on False: an outage must not split one workspace into two rows.
    """
    t0 = time.perf_counter()
    err = None
    saved = False
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
        cursor = conn.cursor()
        
        # Serialize fields to JSON
        recipe_json = json.dumps(recipe)
        rules_json = json.dumps(rules)
        scanned_columns_json = json.dumps(list(scanned_columns))
        step_count = len(recipe)
        now = datetime.now().isoformat()
        
        # Single atomic UPSERT. The WHERE clause on DO UPDATE refuses to overwrite a row owned by a
        # different registered user; unowned and guest-owned rows may still be claimed.
        # (substr instead of LIKE avoids '_' wildcard and '%' escaping issues across drivers.)
        _execute(conn, cursor, """
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
               OR substr(sessions.user_id, 1, 6) = 'guest_'
               OR sessions.user_id = excluded.user_id
        """, (session_id, filename, recipe_json, step_count, rules_json, scanned_columns_json, user_id, project_name, filename, now, project_name))
        
        saved = cursor.rowcount > 0
        _commit(conn)
        if not saved:
            logger.warning("Refused to overwrite session %s: owned by a different user", session_id)
            return False
        update_cached_project(
            user_id,
            session_id,
            filename=filename,
            project_name=project_name or filename,
            step_count=step_count,
            updated_at=now
        )
        return True
    except Exception as e:
        err = str(e)
        logger.error("Error saving session %s: %s", session_id, e, exc_info=True)
        return None
    finally:
        release_db_connection(conn)
        _log_db_perf("save_session", (time.perf_counter() - t0) * 1000, {"session_id": session_id, "step_count": len(recipe), "saved": saved}, err)

def load_session(session_id: str, current_user_email: str = None) -> dict:
    """Loads a session's details from the database.

    Returns None if the session does not exist or belongs to another registered user.
    Raises ``SessionLoadError`` on a database error, so callers keep the resume cookie.
    """
    t0 = time.perf_counter()
    err = None
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
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
        logger.error("Error loading session %s: %s", session_id, e, exc_info=True)
        raise SessionLoadError(str(e)) from e
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
                "updated_at": row["updated_at"],
                "step_count": row["step_count"],
                "pinned": pinned_val
            })
        if hasattr(st, "session_state"):
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

    Only unowned or guest-owned sessions can be claimed; sessions owned by another registered
    user are left untouched. Returns True if the session now belongs to user_id.
    """
    conn = None
    try:
        conn = get_db_connection()  # may raise DatabaseUnavailableError; handled below
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        _execute(conn, cursor, """
            UPDATE sessions 
            SET user_id = ?, updated_at = ?
            WHERE session_id = ?
              AND (user_id IS NULL OR substr(user_id, 1, 6) = 'guest_' OR user_id = ?)
        """, (user_id, now, session_id, user_id))
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
