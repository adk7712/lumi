"""Data loading, dtype downcasting and session-state initialisation."""
import streamlit as st
import pandas as pd
from pathlib import Path
import io
import logging

logger = logging.getLogger("lumi.state")

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

def calculate_health(df: pd.DataFrame) -> int:
    """Calculates overall dataset health percentage: (1 - proportion of null cells) * 100"""
    if df.size == 0:
        return 0
    null_cells = df.isnull().sum().sum()
    return int((1 - (null_cells / df.size)) * 100)


def build_original_entry(df) -> tuple:
    """Canonical 'Original Data' entry for intermediate_states (index 0)."""
    return ("Original Data", calculate_health(df), len(df))


def reset_intermediate_states():
    """Re-seed intermediate_states with just the 'Original Data' entry (empty if no data)."""
    base = st.session_state.get("raw_data")
    st.session_state.intermediate_states = [build_original_entry(base)] if base is not None else []


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
        'redo_stack': [],
        'recommendations_expanded': True,
        'rule_violations_cache': {},
        'rule_violations_cache_key': None,
    }

    # Force reset or initialize for the first time
    for key, value in defaults.items():
        if from_reset or key not in st.session_state:
            st.session_state[key] = value
