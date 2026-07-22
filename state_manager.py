import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
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


@st.cache_data
def load_data(file_path_or_buffer, nrows=None):
    """Loads data from a file path or buffer, supporting CSV and Excel."""
    try:
        # Streamlit's UploadedFile object has a 'type' attribute
        if hasattr(file_path_or_buffer, 'type'):
            file_type = file_path_or_buffer.type
            if file_type == "text/csv":
                df = pd.read_csv(file_path_or_buffer, nrows=nrows)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
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

def add_rule(rule_dict: dict, at_end: bool = False):
    """Applies color/enabled status to a rule and adds it to st.session_state.rules."""
    from ui_utils import get_safe_hue
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
        'trans_type_select': "Find and Replace",
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
            
    # Sync active features in diagnostics tab
    if target in st.session_state.active_features:
        idx = st.session_state.active_features.index(target)
        st.session_state.active_features[idx] = new_name


CACHE_DIR = Path(".lumi_cache")

def calculate_file_hash(file_buffer) -> str:
    """Generates a unique hash for a file using its name, size, and first 8KB of content."""
    hasher = hashlib.md5()
    name = getattr(file_buffer, 'name', '')
    size = getattr(file_buffer, 'size', 0)
    hasher.update(name.encode('utf-8'))
    hasher.update(str(size).encode('utf-8'))
    
    try:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        chunk = file_buffer.read(8192)
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

def process_uploaded_file(file_buffer, file_hash: str, restore_session_id: str = None):
    """Processes a newly uploaded file and initializes the session state."""
    is_large = file_buffer.size > LARGE_FILE_THRESHOLD_BYTES
    if is_large:
        st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.")
    raw_df = load_data(file_buffer, nrows=MAX_SAMPLE_ROWS if is_large else None)
    st.session_state.original_full_data = raw_df
    if not is_large and len(raw_df) > MAX_SAMPLE_ROWS:
        st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    else:
        st.session_state.raw_data = raw_df

    st.session_state.last_file_hash = file_hash
    st.session_state.filename = file_buffer.name
    
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
    
    save_db_session()
    st.toast("Dataset Analyzed")

def load_session_state(file_hash: str, file_buffer):
    """Loads and restores the cleaning recipe and rules from the local cache."""
    process_uploaded_file(file_buffer, file_hash)
    
    cache_path = CACHE_DIR / f"{file_hash}.json"
    if not cache_path.exists():
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
        
        st.toast("Session Restored successfully")
    except Exception as e:
        st.error(f"Error restoring session: {str(e)}")

def save_db_session():
    """Saves the current session state to the SQLite database."""
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
        
    user_id = None
    try:
        if st.experimental_user and st.experimental_user.get("email"):
            user_id = st.experimental_user.get("email")
    except Exception:
        pass
        
    filename = st.session_state.get("filename", "untitled.csv")
    recipe = st.session_state.get("cleaning_recipe", [])
    rules = st.session_state.get("rules", [])
    scanned_columns = st.session_state.get("scanned_columns", set())
    
    from persistence import save_session
    save_session(
        session_id=session_id,
        filename=filename,
        recipe=recipe,
        rules=rules,
        scanned_columns=scanned_columns,
        user_id=user_id
    )

def load_db_session(session_id: str, file_buffer) -> bool:
    """Loads and restores the cleaning recipe and rules from the SQLite database."""
    from persistence import load_session
    db_session = load_session(session_id)
    if not db_session:
        return False
        
    # Process the file buffer to load data, using the retrieved session_id
    process_uploaded_file(file_buffer, session_id, restore_session_id=session_id)
    
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
