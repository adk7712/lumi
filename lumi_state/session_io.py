"""Local cache + DB session save/restore, cookies, background save executor, workspace switching."""
import streamlit as st
from pathlib import Path
import json
import hashlib
from engine import apply_recipe
from scout import generate_proposals
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import time
import logging

logger = logging.getLogger("lumi.state")

from lumi_state.core import (
    MAX_SAMPLE_ROWS, LARGE_FILE_THRESHOLD_BYTES, CachedFileWrapper, load_data, calculate_health,
    build_original_entry, initialize_state, downcast_dtypes,
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


def resume_owner():
    """Owner scope for the local resume cache, or None when there is no identity yet.

    Signed-in users are scoped by account id. Guests are scoped by their session id:
    the current ``session_id`` if a dataset is loaded, else the one recovered from the
    HMAC-signed ``lumi_session`` cookie (unsigned/forged cookies yield nothing), so only
    the same browser sees its own restore prompt.
    """
    try:
        from ui_utils import get_logged_in_user, is_authenticated_user, read_session_cookie
        if is_authenticated_user():
            return f"user:{get_logged_in_user()}"
        sid = st.session_state.get("session_id") or read_session_cookie()
    except Exception:
        logger.warning("Could not resolve resume-cache owner", exc_info=True)
        return None
    return f"guest:{sid}" if sid else None


def resume_cache_path(file_hash: str, owner: str = None):
    """Owner-scoped cache file ``.lumi_cache/<sha256(owner|file_hash)>.json`` (None without an owner).

    Best-effort only: on Streamlit Community Cloud the local disk is ephemeral, so this
    cache disappears on redeploy/restart. Signed-in users' durable path is the DB-backed
    workspace list; for guests, resume after closing the tab is best-effort.
    """
    owner = owner if owner is not None else resume_owner()
    if not owner or not file_hash:
        return None
    key = hashlib.sha256(f"lumi-resume-v2|{owner}|{file_hash}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _purge_legacy_cache(file_hash: str):
    """Delete the pre-V1 unscoped ``.lumi_cache/{file_hash}.json`` (it is never offered to anyone)."""
    if not file_hash:
        return
    legacy = CACHE_DIR / f"{file_hash}.json"
    try:
        if legacy.exists():
            legacy.unlink()
    except Exception:
        logger.debug("Could not delete legacy cache %s", legacy, exc_info=True)


def has_resume_cache(file_hash: str, owner: str = None) -> bool:
    """True only when *this owner* has saved progress for this file."""
    _purge_legacy_cache(file_hash)
    path = resume_cache_path(file_hash, owner)
    return bool(path and path.exists())


def discard_resume_cache(file_hash: str, owner: str = None):
    """Delete this owner's saved progress for this file (no-op if absent)."""
    _purge_legacy_cache(file_hash)
    path = resume_cache_path(file_hash, owner)
    if path and path.exists():
        try:
            path.unlink()
        except Exception:
            logger.warning("Could not delete resume cache %s", path, exc_info=True)


def save_session_state():
    """Saves the current recipe and rules to the owner-scoped local cache."""
    file_hash = st.session_state.get('last_file_hash')
    if not file_hash:
        return
        
    try:
        cache_path = resume_cache_path(file_hash)
        if cache_path is None:
            return  # no owner identity yet -> never write an unscoped cache
        CACHE_DIR.mkdir(exist_ok=True)
        
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

def _capture_full_source(file_buffer):
    """Raw bytes + name of the upload, kept only for sampled files (None if unreadable)."""
    try:
        if hasattr(file_buffer, 'seek'):
            file_buffer.seek(0)
        raw = file_buffer.read()
        if hasattr(file_buffer, 'seek'):
            file_buffer.seek(0)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return {"name": getattr(file_buffer, 'name', 'dataset.csv'), "bytes": bytes(raw)}
    except Exception:
        logger.warning("Could not retain uploaded bytes for full-file export", exc_info=True)
        return None


def load_full_data():
    """Re-reads the entire uploaded file (all rows, same dtype downcasting as load_data).

    Deliberately bypasses the ``st.cache_data`` wrapper on ``load_data`` so the big frame is not
    retained in the global cache. Raises ``ValueError`` when the source bytes are unavailable.
    """
    import io
    import pandas as pd
    src = st.session_state.get("full_source")
    if not src or not src.get("bytes"):
        raise ValueError("The original upload is no longer available; re-upload the file to export all rows.")
    from lumi_state.core import read_tabular, UnsafeFileError
    try:
        df, _notices = read_tabular(src["bytes"], src.get("name") or "")
    except UnsafeFileError as e:
        raise ValueError(str(e)) from e  # same failure path as missing bytes: caller shows st.error
    return downcast_dtypes(df)


def process_uploaded_file(file_buffer, file_hash: str, restore_session_id: str = None, _skip_db_save: bool = False):
    """Processes a newly uploaded file and initializes the session state.
    
    Args:
        _skip_db_save: When True, suppresses the automatic save_db_session() call.
            Used during session restoration to avoid overwriting the DB with empty state
            before the caller has a chance to restore rules/recipe/scanned_columns.

    Returns False (and leaves ``raw_data`` None, reason in ``st.session_state.load_error``)
    when the file cannot be loaded; True otherwise.
    """
    # Ensure file cursor is at the start before reading
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    
    file_size = getattr(file_buffer, 'size', 0)
    is_large = file_size > LARGE_FILE_THRESHOLD_BYTES
    if is_large:
        st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.")
    raw_df = load_data(file_buffer, nrows=MAX_SAMPLE_ROWS if is_large else None)
    load_error = raw_df.attrs.pop("lumi_load_error", None)
    if load_error is None and raw_df.columns.empty:
        load_error = "Could not read this file: it is empty (no column headers or rows)."
    elif load_error is None and raw_df.empty:
        load_error = "Could not use this file: it has column headers but no data rows."
    if load_error:
        # Stay on the landing page: a 0-row workspace would render blank after the rerun.
        # The landing page shows this until the next upload attempt.
        st.session_state.raw_data = None
        st.session_state.load_error = load_error
        st.session_state._load_error_hash = file_hash
        return False
    st.session_state.pop("load_error", None)
    st.session_state.pop("_load_error_hash", None)
    load_notices = list(raw_df.attrs.pop("lumi_notices", []) or [])
    st.session_state.load_notices = load_notices
    for _n in load_notices:
        st.toast(_n)
    # The full frame is never kept in session_state; sampled uploads keep only the raw file bytes
    # (smaller than a frame) so the full file can be re-read on demand (see load_full_data).
    st.session_state.original_full_data = None
    st.session_state.full_export = None
    st.session_state.sample_export = None
    st.session_state.full_source = None
    if not is_large and len(raw_df) > MAX_SAMPLE_ROWS:
        st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        st.session_state.sample_info = {"shown": MAX_SAMPLE_ROWS, "total": len(raw_df), "method": "random"}
    elif is_large:
        # Total is unknown: counting rows would mean reading the whole >50MB file.
        st.session_state.raw_data = raw_df
        st.session_state.sample_info = {"shown": len(raw_df), "total": None, "method": "head"}
    else:
        st.session_state.raw_data = raw_df
        st.session_state.sample_info = None
    if st.session_state.sample_info:
        st.session_state.full_source = _capture_full_source(file_buffer)

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
    return True

# Widget keys whose value belongs to one workspace's dataset (column picks, preview size, mode).
# Streamlit keeps a keyed widget's value across reruns, so without clearing them a new/switched/
# reset workspace inherits the previous one's "Rows to Show", mode radio and column selections.
WORKSPACE_VIEW_KEYS = (
    "p_mode", "preview_mode_select", "preview_row_count", "full_include_xlsx",
    "desc_cols", "desc_dtypes", "outlier_box_cols", "corr_range_val",
)


def reset_workspace_view_state():
    """Forget per-workspace view widget values (and the last step's warnings)."""
    for key in WORKSPACE_VIEW_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("_step_warnings", None)


def _regenerate_restored_proposals():
    """Rebuild Recommended Rules after a session is restored (resume or workspace switch).

    process_uploaded_file() leaves the proposals of a fresh upload; the restored recipe, rules
    and scanned_columns must filter them, or the remaining ones are missing/stale.
    """
    from lumi_state.steps import regenerate_proposals
    regenerate_proposals()


def load_session_state(file_hash: str, file_buffer):
    """Loads and restores the cleaning recipe and rules from this owner's local cache."""
    # Resolve the owner before process_uploaded_file() assigns a session id.
    owner = resume_owner()
    cache_path = resume_cache_path(file_hash, owner)
    _purge_legacy_cache(file_hash)
    # Guests keep their session id across a restore so the cookie (and therefore the
    # scope of this cache) stays the same after the next reload.
    restore_sid = owner.split(":", 1)[1] if owner and owner.startswith("guest:") else None
    if not process_uploaded_file(file_buffer, file_hash, restore_session_id=restore_sid, _skip_db_save=True):
        return

    if cache_path is None or not cache_path.exists():
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
        
        # Re-generate proposals on the restored (post-recipe) data, minus ones already applied/dismissed
        _regenerate_restored_proposals()
        
        # Persist the restored state to the DB so it survives page reloads
        save_db_session()
        
        st.toast("Session Restored successfully")
    except Exception as e:
        st.error(f"Error restoring session: {str(e)}")

_rotated_sessions: dict = {}
# Failed background saves, keyed by the session id that was saved. Background threads must not
# touch st.session_state / st.toast, so they only record here; the next main-thread run consumes it.
_failed_saves: dict = {}
_bg_lock = threading.Lock()
_BG_MAP_CAP = 256


def _bounded_put(mapping: dict, key, value):
    """Insert under ``_bg_lock`` (held by caller), evicting oldest entries beyond the cap."""
    mapping.pop(key, None)
    mapping[key] = value
    while len(mapping) > _BG_MAP_CAP:
        mapping.pop(next(iter(mapping)), None)


def _on_bg_done(fut, old_sid):
    try:
        res = fut.result()
    except Exception:
        logger.error("Background save crashed", exc_info=True)
        res = False
    _record_bg_result(res, old_sid)


def _record_bg_result(res, old_sid):
    with _bg_lock:
        if res is False or res is None:  # None: a DB error that was not retried away
            logger.error("Background DB save failed for session %s", old_sid)
            _bounded_put(_failed_saves, old_sid, True)
        else:
            _failed_saves.pop(old_sid, None)
            if res and res != old_sid:
                _bounded_put(_rotated_sessions, old_sid, res)


# Saves for one session run one at a time, newest snapshot only. Parallel workers used to
# commit snapshots out of order (last writer wins), so a reload could restore an older recipe.
_pending_saves: dict = {}   # sid -> latest job not yet started
_running_saves: set = set()  # sids with a drain loop in progress
_save_cond = threading.Condition(_bg_lock)


def _submit_save(sid, job):
    """Queue ``job(target_sid) -> result`` for ``sid``; a newer job replaces an unstarted one."""
    with _save_cond:
        _pending_saves[sid] = job
        if sid in _running_saves:
            return
        _running_saves.add(sid)
    _db_executor.submit(_drain_saves, sid)


# Pauses before re-running a save job that hit a DB error (None / exception). A refusal (False)
# is final and never retried; a newer queued snapshot also replaces the retry.
_SAVE_RETRY_DELAYS = (0.5, 1.5)


def _run_save_job(sid, job, target):
    """``job(target)`` with retries on errors; returns the result, an error becomes False."""
    for attempt in range(len(_SAVE_RETRY_DELAYS) + 1):
        try:
            res = job(target)
        except Exception:
            logger.error("Background save crashed", exc_info=True)
            res = None
        if res is not None:
            return res
        if attempt == len(_SAVE_RETRY_DELAYS):
            break
        with _save_cond:
            if sid in _pending_saves:
                break  # a newer snapshot is queued; it is the one worth saving
        logger.warning("Background save for session %s failed; retrying", sid)
        time.sleep(_SAVE_RETRY_DELAYS[attempt])
    return False


def _drain_saves(sid):
    target = sid
    while True:
        with _save_cond:
            job = _pending_saves.pop(sid, None)
            if job is None:
                _running_saves.discard(sid)
                _save_cond.notify_all()
                return
        res = _run_save_job(sid, job, target)
        if res and res is not True:
            target = res  # a rotated id: later snapshots of this session go to the new row
        _record_bg_result(res, sid)


def cancel_pending_saves(sid, timeout: float = 5.0) -> bool:
    """Drop queued saves for ``sid`` and wait for a running one, so a delete is not undone."""
    with _save_cond:
        _pending_saves.pop(sid, None)
        return _save_cond.wait_for(lambda: sid not in _running_saves, timeout=timeout)


def _set_session_cookie(session_id):
    """Write the signed ``lumi_session`` cookie for ``session_id``."""
    try:
        from ui_utils import sign_cookie_value, set_cookie, SESSION_COOKIE
        set_cookie(SESSION_COOKIE, sign_cookie_value(session_id, purpose="session"))
    except Exception as e:
        logger.warning("Error setting session cookie: %s", e)


def _apply_rotated_session():
    """Main-thread hook (call each run): surface a failed background save once, apply id rotation."""
    sid = st.session_state.get("session_id")
    with _bg_lock:
        failed = _failed_saves.pop(sid, None)
        new = _rotated_sessions.pop(sid, None)
    if failed:
        _warn_save_failure_once("Could not save to the database.")
    if new:
        st.session_state.session_id = new
        _set_session_cookie(new)
    if failed or new:
        _invalidate_project_cache()  # the optimistic sidebar entry no longer matches the DB


def _invalidate_project_cache():
    """Main thread only: drop the signed-in user's cached workspace list so it is re-read."""
    try:
        from ui_utils import get_logged_in_user
        from persistence import clear_user_projects_cache
        user_id = get_logged_in_user()
        if user_id:
            clear_user_projects_cache(user_id)
    except Exception:
        logger.warning("Could not invalidate cached workspace list", exc_info=True)


def _cache_saved_project(user_id, session_id, filename, project_name, step_count):
    """Main thread only: reflect a save in the sidebar cache (inserting a new workspace).

    The save itself runs on a background thread where st.session_state is not this user's
    session, so the cache is updated here from the same snapshot instead.
    """
    try:
        from datetime import datetime
        from persistence import update_cached_project
        updates = {"filename": filename, "step_count": step_count, "updated_at": datetime.now().isoformat()}
        if project_name:
            updates["project_name"] = project_name
        update_cached_project(user_id, session_id, **updates)
    except Exception:
        logger.warning("Could not update cached workspace list", exc_info=True)


def _is_sync_save() -> bool:
    """Saves run synchronously under test (avoids races with assertions)."""
    import os
    return os.getenv("LUMI_TESTING") == "1" or st.session_state.get("_is_testing", False)


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
        """sid / rotated id on success; None for a DB error (retryable); False after a refusal."""
        from persistence import save_session
        refused = False
        try:
            ok = save_session(
                session_id=sid, filename=filename, recipe=recipe, rules=rules,
                scanned_columns=scanned_columns, user_id=user_id, project_name=project_name
            )
            if ok is None:
                return None  # DB/serialization error: keep the id; the worker retries, then reports
            if ok is False:
                refused = True
                import uuid
                logger.warning("save_session refused for session %s (owned by another user); rotating id", sid)
                new_sid = str(uuid.uuid4())
                ok = save_session(
                    session_id=new_sid, filename=filename, recipe=recipe, rules=rules,
                    scanned_columns=scanned_columns, user_id=user_id, project_name=project_name
                )
                if ok:
                    return new_sid
                logger.error("Retry save under fresh session id %s also failed", new_sid)
                return False
            return sid
        except Exception:
            logger.error("Database save failed for session %s", sid, exc_info=True)
            return False if refused else None

    def _finish(new_sid):
        if not new_sid:
            _warn_save_failure_once("Could not save to the database.")
        elif new_sid and new_sid != session_id:
            st.session_state.session_id = new_sid
            _set_session_cookie(new_sid)

    # During testing, run synchronously to avoid race conditions with assertions
    if _is_sync_save():
        res = _bg_save()
        _finish(res)
        if res:
            _cache_saved_project(user_id, res, filename, project_name, len(recipe))
    else:
        # Background thread cannot touch st.session_state safely; rotate id synchronously on the
        # main thread only when the result is already known, otherwise keep best-effort semantics.
        _submit_save(session_id, lambda target: _bg_save(sid=target))
        # Optimistic sidebar entry; a refused/failed save invalidates it via _apply_rotated_session.
        _cache_saved_project(user_id, session_id, filename, project_name, len(recipe))

def load_db_session(session_id: str, file_buffer, preloaded_session: dict = None) -> bool:
    """Loads and restores the cleaning recipe and rules from the database."""
    from persistence import load_session, SessionLoadError
    from ui_utils import get_logged_in_user
    try:
        db_session = preloaded_session or load_session(session_id, get_logged_in_user())
    except SessionLoadError:
        return False
    if not db_session:
        return False
    
    # Ensure file cursor is at the start before reading
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    
    # Process the file buffer to load data, using the retrieved session_id.
    # _skip_db_save=True prevents overwriting the DB with empty state before
    # we have a chance to restore the actual recipe/rules/scanned_columns.
    if not process_uploaded_file(file_buffer, session_id, restore_session_id=session_id, _skip_db_save=True):
        return False
    
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
    
    # Re-generate proposals on the restored (post-recipe) data, minus ones already applied/dismissed
    _regenerate_restored_proposals()

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
    from persistence import load_session, SessionLoadError
    try:
        db_session = load_session(target_session_id, user_email)
    except SessionLoadError:
        st.error("Could not reach the database. Try again in a moment.")
        return False
    if not db_session:
        st.error("Workspace not found or access denied.")
        return False
        
    filename = db_session.get("filename", "dataset.csv")
    buf = load_cached_file(target_session_id, filename)
    
    if buf is None:
        # Fallback: Cache missing on disk -> clear state & redirect to landing resume prompt
        initialize_state(from_reset=True)
        reset_workspace_view_state()
        st.session_state.resume_session_id = target_session_id
        st.rerun()
        return False
        
    # 3. Clean state reset
    initialize_state(from_reset=True)
    reset_workspace_view_state()
    
    # 4. Delegate to existing load_db_session with preloaded_session (bypasses redundant WAN roundtrip)
    success = load_db_session(target_session_id, buf, preloaded_session=db_session)
    if success:
        st.session_state.project_name = db_session.get("project_name")
        st.rerun()
    elif st.session_state.get("load_error"):
        st.rerun()  # the cached file no longer loads: the landing page shows why
    return success

