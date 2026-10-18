"""Local cache + DB session save/restore, cookies, background save executor, workspace switching."""
import streamlit as st
from pathlib import Path
import json
import hashlib
from engine import apply_recipe
from scout import generate_proposals
from concurrent.futures import ThreadPoolExecutor
import copy
import logging

logger = logging.getLogger("lumi.state")

from lumi_state.core import (
    MAX_SAMPLE_ROWS, LARGE_FILE_THRESHOLD_BYTES, CachedFileWrapper, load_data, calculate_health,
    build_original_entry, initialize_state,
)

_db_executor = ThreadPoolExecutor(max_workers=4)

def _update_project_cache():
    """Best-effort refresh of the cached project step count (never raises)."""
    try:
        from ui_utils import get_logged_in_user
        from persistence import update_cached_project
        user_id = get_logged_in_user()
        session_id = st.session_state.get("session_id")
        if user_id and session_id:
            update_cached_project(user_id, session_id, step_count=len(st.session_state.cleaning_recipe))
    except Exception:
        logger.warning("Could not update cached project step count", exc_info=True)



CACHE_DIR = Path(".lumi_cache")

def calculate_file_hash(file_buffer) -> str:
    """Generates a unique hash for a file by reading its full content.
    
    Previously only hashed the first 8KB, which caused hash collisions between
    files that share the same name, size, and header but differ in later rows.
    """
    hasher = hashlib.md5()
    name = getattr(file_buffer, 'name', '')
    hasher.update(name.encode('utf-8'))
    
    try:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        # Read in chunks to avoid loading huge files into memory at once.
        while True:
            chunk = file_buffer.read(65536)
            if not chunk:
                break
            if isinstance(chunk, str):
                hasher.update(chunk.encode('utf-8'))
            else:
                hasher.update(chunk)
        file_buffer.seek(pos)
    except Exception:
        logger.warning("Could not fully read buffer for hashing", exc_info=True)
        
    return hasher.hexdigest()

def _warn_save_failure_once(msg: str):
    """Surface a non-blocking warning once per session (avoids toast spam)."""
    try:
        if st.session_state.get("_save_warning_shown"):
            return
        st.session_state["_save_warning_shown"] = True
        st.toast(f"Warning: {msg} Your recent changes may not be saved.")
    except Exception:
        logger.debug("Could not display save warning", exc_info=True)


def save_session_state():
    """Saves the current recipe and rules to the local cache directory."""
    file_hash = st.session_state.get('last_file_hash')
    if not file_hash:
        return
        
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path = CACHE_DIR / f"{file_hash}.json"
        
        # Serialize scanned_columns as list
        scanned_cols = list(st.session_state.get('scanned_columns', set()))
        
        data = {
            'cleaning_recipe': st.session_state.get('cleaning_recipe', []),
            'rules': st.session_state.get('rules', []),
            'scanned_columns': scanned_cols
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception:
        logger.error("Failed to save local session cache", exc_info=True)
        _warn_save_failure_once("Could not save progress to local cache.")


def cache_uploaded_file(session_id: str, file_buffer):
    """Saves raw uploaded file to .lumi_cache/{session_id}.dat for logged-in users."""
    from ui_utils import is_authenticated_user
    if not is_authenticated_user() or st.session_state.get("_is_testing", False) or not session_id:
        return  # Skip guests and test suite to prevent disk leaks
        
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{session_id}.dat"
    try:
        pos = file_buffer.tell() if hasattr(file_buffer, 'tell') else 0
        file_buffer.seek(0)
        cache_path.write_bytes(file_buffer.read())
        file_buffer.seek(pos)
    except Exception:
        logger.error("Error caching raw file for session %s", session_id, exc_info=True)

def load_cached_file(session_id: str, filename: str):
    """Loads raw file buffer wrapped as CachedFileWrapper."""
    cache_path = CACHE_DIR / f"{session_id}.dat"
    if not cache_path.exists():
        return None
    try:
        return CachedFileWrapper(cache_path.read_bytes(), filename)
    except Exception:
        logger.warning("Could not read cached file for session %s", session_id, exc_info=True)
        return None

def delete_cached_file(session_id: str):
    """Deletes cached .dat file for a session."""
    cache_path = CACHE_DIR / f"{session_id}.dat"
    if cache_path.exists():
        try:
            cache_path.unlink()
        except Exception:
            logger.warning("Could not delete cached file %s", cache_path, exc_info=True)

def process_uploaded_file(file_buffer, file_hash: str, restore_session_id: str = None, _skip_db_save: bool = False):
    """Processes a newly uploaded file and initializes the session state.
    
    Args:
        _skip_db_save: When True, suppresses the automatic save_db_session() call.
            Used during session restoration to avoid overwriting the DB with empty state
            before the caller has a chance to restore rules/recipe/scanned_columns.
    """
    # Ensure file cursor is at the start before reading
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    
    file_size = getattr(file_buffer, 'size', 0)
    is_large = file_size > LARGE_FILE_THRESHOLD_BYTES
    if is_large:
        st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.")
    raw_df = load_data(file_buffer, nrows=MAX_SAMPLE_ROWS if is_large else None)
    st.session_state.original_full_data = raw_df
    if not is_large and len(raw_df) > MAX_SAMPLE_ROWS:
        st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    else:
        st.session_state.raw_data = raw_df

    st.session_state.last_file_hash = file_hash
    st.session_state.filename = getattr(file_buffer, 'name', 'dataset.csv')
    from ui_utils import queue_event
    queue_event("file_uploaded", {"filename": getattr(file_buffer, 'name', 'dataset.csv'), "size": file_size})
    
    if restore_session_id:
        session_id = restore_session_id
    else:
        import uuid
        session_id = str(uuid.uuid4())
        
    st.session_state.session_id = session_id

    # Store it in a browser cookie via CookieController (not a URL param)
    _set_session_cookie(session_id)

    st.session_state.active_features = []
    st.session_state.scanned_columns = set()
    st.session_state.cleaning_recipe = []
    st.session_state.rules = []

    base_df = st.session_state.raw_data
    bh = calculate_health(base_df)
    st.session_state.intermediate_states = [build_original_entry(base_df)]
    st.session_state.current_df = base_df.copy()
    st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
    
    if not _skip_db_save:
        save_db_session()
        
    cache_uploaded_file(session_id, file_buffer)
    st.toast("Dataset Analyzed")

def load_session_state(file_hash: str, file_buffer):
    """Loads and restores the cleaning recipe and rules from the local cache."""
    process_uploaded_file(file_buffer, file_hash, _skip_db_save=True)
    
    cache_path = CACHE_DIR / f"{file_hash}.json"
    if not cache_path.exists():
        save_db_session()
        return
        
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        st.session_state.cleaning_recipe = data.get('cleaning_recipe', [])
        st.session_state.rules = data.get('rules', [])
        st.session_state.scanned_columns = set(data.get('scanned_columns', []))
        
        # Apply the full recipe to restore intermediate states and current_df
        recipe = st.session_state.cleaning_recipe
        st.session_state.current_df, _ = apply_recipe(st.session_state.raw_data.copy(), recipe)
        
        # Re-build intermediate_states metadata list
        intermediate_states = [st.session_state.intermediate_states[0]]
        temp_df = st.session_state.raw_data.copy()
        
        for step in recipe:
            temp_df, _ = apply_recipe(temp_df, [step])
            th = calculate_health(temp_df)
            step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
            intermediate_states.append((step_desc, th, len(temp_df)))
            
        st.session_state.intermediate_states = intermediate_states
        
        # Re-generate proposals based on scanned columns
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
        
        # Persist the restored state to the DB so it survives page reloads
        save_db_session()
        
        st.toast("Session Restored successfully")
    except Exception as e:
        st.error(f"Error restoring session: {str(e)}")

_rotated_sessions: dict = {}


def _on_bg_done(fut, old_sid):
    try:
        res = fut.result()
    except Exception:
        logger.error("Background save crashed", exc_info=True)
        return
    if res is False:
        logger.error("Background DB save failed for session %s", old_sid)
    elif res and res != old_sid:
        _rotated_sessions[old_sid] = res


def _set_session_cookie(session_id):
    """Write the signed ``lumi_session`` cookie for ``session_id``."""
    try:
        from streamlit_cookies_controller import CookieController
        from ui_utils import sign_cookie_value, SESSION_COOKIE
        CookieController().set(SESSION_COOKIE, sign_cookie_value(session_id, purpose="session"))
    except Exception as e:
        logger.warning("Error setting session cookie: %s", e)


def _apply_rotated_session():
    new = _rotated_sessions.pop(st.session_state.get("session_id"), None)
    if new:
        st.session_state.session_id = new
        _set_session_cookie(new)


def save_db_session():
    """Saves the current session state to the database asynchronously via background thread."""
    _apply_rotated_session()
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
    
    user_id = None
    try:
        from ui_utils import get_logged_in_user
        user_id = get_logged_in_user()
    except Exception:
        logger.warning("Could not resolve logged-in user for DB save", exc_info=True)
    
    # Snapshot all data synchronously from session_state
    filename = st.session_state.get("filename", "untitled.csv")
    recipe = copy.deepcopy(st.session_state.get("cleaning_recipe", []))
    rules = copy.deepcopy(st.session_state.get("rules", []))
    scanned_columns = set(st.session_state.get("scanned_columns", set()))
    project_name = st.session_state.get("project_name") or st.session_state.get("_pending_workspace_name")
    
    # Handle pending workspace name synchronously
    if st.session_state.get("_pending_workspace_name"):
        st.session_state.project_name = st.session_state._pending_workspace_name
        st.session_state._pending_workspace_name = None
    
    def _bg_save(sid=session_id):
        from persistence import save_session
        try:
            ok = save_session(
                session_id=sid, filename=filename, recipe=recipe, rules=rules,
                scanned_columns=scanned_columns, user_id=user_id, project_name=project_name
            )
            if ok is False:
                import uuid
                logger.warning("save_session refused for session %s (owned by another user); rotating id", sid)
                new_sid = str(uuid.uuid4())
                ok = save_session(
                    session_id=new_sid, filename=filename, recipe=recipe, rules=rules,
                    scanned_columns=scanned_columns, user_id=user_id, project_name=project_name
                )
                if ok is not False:
                    return new_sid
                logger.error("Retry save under fresh session id %s also failed", new_sid)
                return False
            return sid
        except Exception:
            logger.error("Database save failed for session %s", sid, exc_info=True)
            return False

    def _finish(new_sid):
        if new_sid is False:
            _warn_save_failure_once("Could not save to the database.")
        elif new_sid and new_sid != session_id:
            st.session_state.session_id = new_sid
            _set_session_cookie(new_sid)

    # During testing, run synchronously to avoid race conditions with assertions
    import os
    is_testing = os.getenv("LUMI_TESTING") == "1" or st.session_state.get("_is_testing", False)
    if is_testing:
        _finish(_bg_save())
    else:
        # Background thread cannot touch st.session_state safely; rotate id synchronously on the
        # main thread only when the result is already known, otherwise keep best-effort semantics.
        fut = _db_executor.submit(_bg_save)
        fut.add_done_callback(lambda f: _on_bg_done(f, session_id))

def load_db_session(session_id: str, file_buffer, preloaded_session: dict = None) -> bool:
    """Loads and restores the cleaning recipe and rules from the database."""
    from persistence import load_session
    from ui_utils import get_logged_in_user
    db_session = preloaded_session or load_session(session_id, get_logged_in_user())
    if not db_session:
        return False
    
    # Ensure file cursor is at the start before reading
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    
    # Process the file buffer to load data, using the retrieved session_id.
    # _skip_db_save=True prevents overwriting the DB with empty state before
    # we have a chance to restore the actual recipe/rules/scanned_columns.
    process_uploaded_file(file_buffer, session_id, restore_session_id=session_id, _skip_db_save=True)
    
    # Restore metadata from DB
    st.session_state.cleaning_recipe = db_session.get('cleaning_recipe', [])
    st.session_state.rules = db_session.get('rules', [])
    st.session_state.scanned_columns = db_session.get('scanned_columns', set())
    st.session_state.session_id = session_id
    st.session_state.filename = db_session.get('filename', file_buffer.name)
    
    # Apply the full recipe to restore intermediate states and current_df
    recipe = st.session_state.cleaning_recipe
    st.session_state.current_df, _ = apply_recipe(st.session_state.raw_data.copy(), recipe)
    
    # Re-build intermediate_states metadata list
    intermediate_states = [st.session_state.intermediate_states[0]]
    temp_df = st.session_state.raw_data.copy()
    
    for step in recipe:
        temp_df, _ = apply_recipe(temp_df, [step])
        th = calculate_health(temp_df)
        step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
        intermediate_states.append((step_desc, th, len(temp_df)))
        
    st.session_state.intermediate_states = intermediate_states
    
    # Re-generate proposals based on scanned columns
    st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
    
    st.toast("Session Restored successfully")
    return True


def switch_workspace(target_session_id: str) -> bool:
    """Switches the active workspace to target_session_id for an authenticated user."""
    from ui_utils import get_logged_in_user, is_authenticated_user
    if not is_authenticated_user():
        return False
        
    user_email = get_logged_in_user()
    
    # 1. Auto-save active workspace before switching
    save_session_state()
    save_db_session()
    
    # 2. Verify target workspace ownership
    from persistence import load_session
    db_session = load_session(target_session_id, user_email)
    if not db_session:
        st.error("Workspace not found or access denied.")
        return False
        
    filename = db_session.get("filename", "dataset.csv")
    buf = load_cached_file(target_session_id, filename)
    
    if buf is None:
        # Fallback: Cache missing on disk -> clear state & redirect to landing resume prompt
        initialize_state(from_reset=True)
        st.session_state.resume_session_id = target_session_id
        st.rerun()
        return False
        
    # 3. Clean state reset
    initialize_state(from_reset=True)
    
    # 4. Delegate to existing load_db_session with preloaded_session (bypasses redundant WAN roundtrip)
    success = load_db_session(target_session_id, buf, preloaded_session=db_session)
    if success:
        st.session_state.project_name = db_session.get("project_name")
        st.rerun()
    return success

