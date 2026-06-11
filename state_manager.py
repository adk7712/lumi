import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from engine import apply_recipe
from scout import generate_proposals

# Define constants
MAX_SAMPLE_ROWS = 10000

@st.cache_data
def load_data(file_path_or_buffer, nrows=None):
    """Loads data from a file path or buffer, supporting CSV and Excel."""
    try:
        # Streamlit's UploadedFile object has a 'type' attribute
        if hasattr(file_path_or_buffer, 'type'):
            file_type = file_path_or_buffer.type
            if file_type == "text/csv":
                return pd.read_csv(file_path_or_buffer, nrows=nrows)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(file_path_or_buffer, nrows=nrows)
        # For local file paths
        elif isinstance(file_path_or_buffer, str):
            suffix = Path(file_path_or_buffer).suffix.lower()
            if suffix == '.csv':
                return pd.read_csv(file_path_or_buffer, nrows=nrows)
            elif suffix == '.xlsx':
                return pd.read_excel(file_path_or_buffer, nrows=nrows)

        # Fallback for buffers without a clear type
        st.warning("Could not determine file type, attempting to read as CSV. May fail for other formats.")
        return pd.read_csv(file_path_or_buffer, nrows=nrows)

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error loading data: {type(e).__name__} - {e}. Please check the file format and content.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
        return pd.DataFrame()

def initialize_state(from_reset=False):
    """Initializes all required session state variables."""
    # 1. Load initial data first to get the columns list for default dropdown selections
    if 'raw_data' not in st.session_state or from_reset:
        raw_df = load_data("mock_data/train.csv")
        st.session_state.original_full_data = raw_df
        if len(raw_df) > MAX_SAMPLE_ROWS:
            st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        else:
            st.session_state.raw_data = raw_df

        base_df = st.session_state.raw_data
        bh = int((1 - (base_df.isnull().sum().sum() / base_df.size)) * 100) if base_df.size > 0 else 0
        st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, set())
        st.session_state.scanned_columns = set()

    df = st.session_state.raw_data
    all_cols = df.columns.tolist()
    first_col = all_cols[0] if all_cols else ""

    # 2. Define default values for the session state
    defaults = {
        'active_features': [],
        'rules': [],
        'cleaning_recipe': [],
        'intermediate_states': st.session_state.intermediate_states,
        'proposals': st.session_state.proposals,
        'scanned_columns': st.session_state.scanned_columns,
        'last_file_hash': None,
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
    }

    # Force reset or initialize for the first time
    for key, value in defaults.items():
        if from_reset or key not in st.session_state:
            st.session_state[key] = value

def add_step(step):
    """Adds a cleaning step to the recipe, updates the cached state, and shows a toast."""
    st.session_state.cleaning_recipe.append(step)

    # Calculate delta state and cache it
    last_df = st.session_state.intermediate_states[-1][3]
    new_df, messages = apply_recipe(last_df, [step])
    for msg in messages:
        st.toast(msg)

    th = int((1 - (new_df.isnull().sum().sum() / new_df.size)) * 100) if new_df.size > 0 else 0
    step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
    st.session_state.intermediate_states.append((step_desc, th, len(new_df), new_df))

    st.toast(f"Step Added: {step['action']}")


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
