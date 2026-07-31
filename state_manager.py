import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
import io
from engine import apply_recipe
from scout import generate_proposals

# Define constants
MAX_SAMPLE_ROWS = 10000
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024 # 50MB


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts float and integer columns to more memory-efficient types."""
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


class CachedFileWrapper:
    """Emulates Streamlit's UploadedFile interface for cached dataset bytes."""
    def __init__(self, raw_bytes: bytes, filename: str):
        from io import BytesIO
        self._bio = BytesIO(raw_bytes)
        self.name = filename
        self.size = len(raw_bytes)
        self.type = "text/csv" if filename.lower().endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def read(self, *args, **kwargs):
        return self._bio.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._bio.seek(*args, **kwargs)

    def tell(self):
        return self._bio.tell()

def _hash_file_or_buffer(obj):
    if isinstance(obj, str):
        return obj
    return f"{getattr(obj, 'name', '')}_{id(obj)}"

@st.cache_data(hash_funcs={
    CachedFileWrapper: _hash_file_or_buffer,
    io.BytesIO: _hash_file_or_buffer,
    io.StringIO: _hash_file_or_buffer,
})
def load_data(file_path_or_buffer, nrows=None):
    """Loads data from a file path or buffer, supporting CSV and Excel."""
    try:
        # Streamlit's UploadedFile object has a 'type' attribute
        if hasattr(file_path_or_buffer, 'type'):
            file_type = file_path_or_buffer.type
            name = getattr(file_path_or_buffer, 'name', '').lower()
            if file_type == "text/csv" or name.endswith('.csv'):
                df = pd.read_csv(file_path_or_buffer, nrows=nrows)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or name.endswith('.xlsx'):
                df = pd.read_excel(file_path_or_buffer, nrows=nrows)
            else:
                df = pd.DataFrame()
        # For local file paths
        elif isinstance(file_path_or_buffer, str):
            suffix = Path(file_path_or_buffer).suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(file_path_or_buffer, nrows=nrows)
            elif suffix == '.xlsx':
                df = pd.read_excel(file_path_or_buffer, nrows=nrows)
            else:
                df = pd.DataFrame()
        else:
            # Fallback for buffers without a clear type
            st.warning("Could not determine file type, attempting to read as CSV. May fail for other formats.")
            df = pd.read_csv(file_path_or_buffer, nrows=nrows)

        return downcast_dtypes(df)

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error loading data: {type(e).__name__} - {e}. Please check the file format and content.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
        return pd.DataFrame()

def add_rule(rule_dict: dict, at_end: bool = True):
    """Applies color/enabled status to a rule and adds it to st.session_state.rules."""
    from ui_utils import get_safe_hue, queue_event
    rule = rule_dict.copy()
    if 'enabled' not in rule:
        rule['enabled'] = True
    if 'color' not in rule:
        hue = get_safe_hue(len(st.session_state.rules))
        rule['color'] = f"hsla({hue}, 70%, 50%, 0.4)"
        
    if at_end:
        st.session_state.rules.append(rule)
    else:
        st.session_state.rules.insert(0, rule)
    queue_event("issue_flagged", {"type": rule.get("type"), "col": rule.get("col")})
    save_session_state()
    save_db_session()

def calculate_health(df: pd.DataFrame) -> int:
    """Calculates overall dataset health percentage: (1 - proportion of null cells) * 100"""
    if df.size == 0:
        return 0
    null_cells = df.isnull().sum().sum()
    return int((1 - (null_cells / df.size)) * 100)

def initialize_state(from_reset=False):
    """Initializes all required session state variables."""
    # 1. Load initial data first to get the columns list for default dropdown selections
    if 'raw_data' not in st.session_state or from_reset:
        st.session_state.raw_data = None
        st.session_state.original_full_data = None
        st.session_state.intermediate_states = []
        st.session_state.current_df = None
        st.session_state.proposals = []
        st.session_state.scanned_columns = set()

    df = st.session_state.raw_data
    all_cols = df.columns.tolist() if df is not None else []
    first_col = all_cols[0] if all_cols else ""

    # 2. Define default values for the session state
    defaults = {
        'active_features': [],
        'rules': [],
        'cleaning_recipe': [],
        'intermediate_states': st.session_state.intermediate_states,
        'current_df': st.session_state.current_df,
        'proposals': st.session_state.proposals,
        'scanned_columns': st.session_state.scanned_columns,
        'last_file_hash': None,
        'session_id': None,
        'filename': None,
        'raw_data': st.session_state.raw_data,
        'original_full_data': st.session_state.original_full_data,

        # Transient UI Widget state initializers (prevents AppTest KeyErrors)
        'find_input': "",
        'replace_input': "",
        'replace_target_col': "All",
        'replace_use_regex': False,
        'rename_new_name_input': "",
        'rename_target_col': first_col,
        'norm_target_col': "All",
        'norm_method_select': "lowercase",
        'cast_target_col': first_col,
        'cast_dtype_select': "string",
        'drop_target_col': first_col,
        'strip_target_col': "All",
        'rule_target_col': first_col,
        'rule_type_select': "Null Check",
        'trans_type_select': "Cast Data Type",
        'rel_feature_a': first_col,
        'rel_feature_b': first_col,
        'rel_op': ">",
        'rel_target_type_radio': "Another Feature",
        'rel_val_input': "",
        'info_note_input': "",
        'show_reorder_success': False,
        'datetime_extract_col': first_col,
        'datetime_component_select': "year",
        'datetime_new_col_name': "",
        'show_uploader': False,
        '_pending_workspace_name': None,
        'project_name': None,
    }

    # Force reset or initialize for the first time
    for key, value in defaults.items():
        if from_reset or key not in st.session_state:
            st.session_state[key] = value

def add_step(step):
    """Adds a cleaning step to the recipe, updates the cached state, and shows a toast."""
    st.session_state.cleaning_recipe.append(step)

    # Calculate delta state from current_df and cache metadata only (no full df copy)
    new_df, messages = apply_recipe(st.session_state.current_df, [step])
    for msg in messages:
        st.toast(msg)

    th = calculate_health(new_df)
    step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
    st.session_state.intermediate_states.append((step_desc, th, len(new_df)))
    st.session_state.current_df = new_df
    
    from ui_utils import queue_event
    queue_event("cleaning_step_added", {"action": step.get("action"), "column": step.get("column")})

    st.toast(f"Step Added: {step['action']}")
    save_session_state()
    save_db_session()


def get_state_at_step(n: int) -> pd.DataFrame:
    """Reconstructs the dataframe at step N by replaying the first N recipe steps from raw_data."""
    original = st.session_state.raw_data
    if n <= 0:
        return original.copy()
    recipe_so_far = st.session_state.cleaning_recipe[:n]
    df, _ = apply_recipe(original.copy(), recipe_so_far)
    return df


def get_column_dependencies(target: str) -> list[str]:
    """Returns the descriptions of any active validation rules that depend on the specified column."""
    dependent_rules = []
    for r in st.session_state.rules:
        if r.get('col') == target or r.get('col_a') == target or r.get('col_b') == target:
            dependent_rules.append(r['desc'])
        elif r.get('type') == "Custom Expression" and target in r.get('query', ''):
            dependent_rules.append(r['desc'])
    return dependent_rules


def sync_column_rename(target: str, new_name: str):
    """Synchronizes active rules and selected diagnostic features when a column is renamed."""
    # Sync validation rules
    for rule in st.session_state.rules:
        if rule.get('col') == target:
            rule['col'] = new_name
            rule['desc'] = rule['desc'].replace(target, new_name)
        if rule.get('col_a') == target:
            rule['col_a'] = new_name
            rule['desc'] = rule['desc'].replace(target, new_name)
        if rule.get('col_b') == target:
            rule['col_b'] = new_name
            rule['desc'] = rule['desc'].replace(target, new_name)
        if rule.get('type') == "Custom Expression":
            import re
            pattern = r'\b' + re.escape(target) + r'\b'
            if 'query' in rule and re.search(pattern, rule['query']):
                rule['query'] = re.sub(pattern, new_name, rule['query'])
            if 'desc' in rule and re.search(pattern, rule['desc']):
                rule['desc'] = re.sub(pattern, new_name, rule['desc'])
            
    # Sync active features in diagnostics tab
    if target in st.session_state.active_features:
        idx = st.session_state.active_features.index(target)
        st.session_state.active_features[idx] = new_name


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
        pass
        
    return hasher.hexdigest()

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
        pass


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
    except Exception as e:
        print(f"Error caching raw file for session {session_id}: {e}")

def load_cached_file(session_id: str, filename: str):
    """Loads raw file buffer wrapped as CachedFileWrapper."""
    cache_path = CACHE_DIR / f"{session_id}.dat"
    if not cache_path.exists():
        return None
    try:
        return CachedFileWrapper(cache_path.read_bytes(), filename)
    except Exception:
        return None

def delete_cached_file(session_id: str):
    """Deletes cached .dat file for a session."""
    cache_path = CACHE_DIR / f"{session_id}.dat"
    if cache_path.exists():
        try: cache_path.unlink()
        except Exception: pass

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
    try:
        from streamlit_cookies_controller import CookieController
        controller = CookieController()
        controller.set("lumi_session", session_id)
    except Exception as e:
        print(f"Error setting cookie: {e}")

    st.session_state.active_features = []
    st.session_state.scanned_columns = set()
    st.session_state.cleaning_recipe = []
    st.session_state.rules = []

    base_df = st.session_state.raw_data
    bh = calculate_health(base_df)
    st.session_state.intermediate_states = [("Original Data", bh, len(base_df))]
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

def save_db_session():
    """Saves the current session state to the database."""
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
        
    user_id = None
    try:
        from ui_utils import get_logged_in_user
        user_id = get_logged_in_user()
    except Exception:
        pass
        
    filename = st.session_state.get("filename", "untitled.csv")
    recipe = st.session_state.get("cleaning_recipe", [])
    rules = st.session_state.get("rules", [])
    scanned_columns = st.session_state.get("scanned_columns", set())
    project_name = st.session_state.get("project_name") or st.session_state.get("_pending_workspace_name")
    
    from persistence import save_session
    save_session(
        session_id=session_id,
        filename=filename,
        recipe=recipe,
        rules=rules,
        scanned_columns=scanned_columns,
        user_id=user_id,
        project_name=project_name
    )
    if st.session_state.get("_pending_workspace_name"):
        st.session_state.project_name = st.session_state._pending_workspace_name
        st.session_state._pending_workspace_name = None

def load_db_session(session_id: str, file_buffer) -> bool:
    """Loads and restores the cleaning recipe and rules from the SQLite database."""
    from persistence import load_session
    from ui_utils import get_logged_in_user
    db_session = load_session(session_id, get_logged_in_user())
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
    
    # Persist the fully restored state to the DB so it survives subsequent reloads
    save_db_session()
    
    st.toast("Session Restored successfully")
    return True

def regenerate_proposals():
    """Helper to recalculate proposals when active rules/steps are removed."""
    if 'raw_data' in st.session_state and st.session_state.raw_data is not None:
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)

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
    
    # 4. Delegate to existing load_db_session (handles recipe, rules, proposals & state sync)
    success = load_db_session(target_session_id, buf)
    if success:
        st.session_state.project_name = db_session.get("project_name")
        st.rerun()
    return success
