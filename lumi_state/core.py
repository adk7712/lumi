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
    """Content-based cache key (id() values are reused after GC, so they can alias a stale entry).

    Hashes name + bytes; the buffer position is always restored.
    """
    if isinstance(obj, str):
        return obj
    import hashlib
    h = hashlib.sha1()
    h.update(str(getattr(obj, 'name', '')).encode('utf-8', 'replace'))
    h.update(b'\0')
    try:
        data = None
        getvalue = getattr(obj, 'getvalue', None)
        if callable(getvalue):  # Streamlit UploadedFile / BytesIO / StringIO: no position change
            data = getvalue()
        if data is None:
            pos = obj.tell()
            try:
                obj.seek(0)
                data = obj.read()
            finally:
                obj.seek(pos)
        h.update(data.encode('utf-8') if isinstance(data, str) else data)
    except Exception:
        logger.warning("Could not hash buffer contents for the load cache; using a unique key", exc_info=True)
        import uuid
        h.update(uuid.uuid4().bytes)  # unhashable content: never share a cache entry
    return h.hexdigest()

_SNIFF_BYTES = 64 * 1024
_SNIFF_LINES = 60
_CANDIDATE_DELIMS = (";", "\t", "|")
_DELIM_NAMES = {";": "semicolon", "\t": "tab", "|": "pipe"}


def _source_bytes(src) -> bytes:
    """Raw bytes of a path, bytes object or file-like (buffer position is restored)."""
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if isinstance(src, str):
        with open(src, "rb") as fh:
            return fh.read()
    getvalue = getattr(src, "getvalue", None)
    if callable(getvalue):
        data = getvalue()
    else:
        pos = src.tell()
        try:
            src.seek(0)
            data = src.read()
        finally:
            src.seek(pos)
    return data.encode("utf-8") if isinstance(data, str) else data


def _decodes_cleanly(data: bytes, encoding: str, chunk: int = 1 << 20) -> bool:
    """Strict, streaming decode check (no full-size str copy is kept)."""
    import codecs
    dec = codecs.getincrementaldecoder(encoding)(errors="strict")
    try:
        for i in range(0, len(data), chunk):
            dec.decode(data[i:i + chunk])
        dec.decode(b"", final=True)
        return True
    except UnicodeDecodeError:
        return False


def _detect_encoding(data: bytes):
    """Returns (pandas_encoding, notice_or_None). Order: BOMs, utf-8, cp1252, latin-1."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", "Read as UTF-16 encoding."
    if _decodes_cleanly(data, "utf-8"):
        return "utf-8", None
    if _decodes_cleanly(data, "cp1252"):
        return "cp1252", "Read as Windows-1252 encoding (the file is not valid UTF-8)."
    return "latin-1", "Read as Latin-1 (ISO-8859-1) encoding (the file is not valid UTF-8)."


def _field_counts(lines, delim):
    import csv
    return [len(row) for row in csv.reader(lines, delimiter=delim, quotechar='"')]


def _sniff_delimiter(data: bytes, encoding: str) -> str:
    """Comma unless the first ~64 KB clearly looks ``;``/tab/``|`` delimited.

    A delimiter counts as "consistent" when >=80% of the sampled lines have the same field count (>1)
    and the header line has that count too. Comma wins whenever it is consistent; we only move away
    from ',' when it is not and a candidate is. Quoting is respected, so comma files whose values contain semicolons stay comma separated.
    """
    from collections import Counter
    prefix = data[:_SNIFF_BYTES]
    text = prefix.decode(encoding, errors="replace")
    lines = text.splitlines()
    if len(data) > _SNIFF_BYTES and lines:
        lines = lines[:-1]  # last line is truncated
    lines = [ln for ln in lines if ln.strip()][:_SNIFF_LINES]
    if not lines:
        return ","

    def consistent(delim):
        """(field count, share of lines with it) if the header matches the dominant count, else None."""
        counts = _field_counts(lines, delim)
        mode, n = Counter(counts).most_common(1)[0]
        share = n / len(counts)
        if mode > 1 and counts[0] == mode and share >= 0.8:
            return share
        return None

    if consistent(",") is not None:
        return ","
    best, best_key = ",", None
    for prio, d in enumerate(_CANDIDATE_DELIMS):
        share = consistent(d)
        if share is not None and (best_key is None or (share, -prio) > best_key):
            best, best_key = d, (share, -prio)
    return best


_DECIMAL_COMMA_RE = r"^\s*[-+]?\d+(,\d+)?\s*$"


def _convert_decimal_commas(df: pd.DataFrame):
    """For ';' files: text columns whose every non-null value is ``123`` or ``12,5`` (with at least
    one comma) become numeric. Anything ambiguous (thousand separators, mixed text) stays text."""
    converted = []
    for col in df.columns:
        ser = df[col]
        if not (pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser)):
            continue
        vals = ser.dropna().astype(str)
        if vals.empty or not vals.str.contains(",", regex=False).any():
            continue
        if not vals.str.match(_DECIMAL_COMMA_RE).all():
            continue
        df[col] = pd.to_numeric(ser.astype(str).str.strip().str.replace(",", ".", regex=False).where(ser.notna()))
        converted.append(str(col))
    return df, converted


def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """All column names become str; duplicates get pandas-style ``.1``/``.2`` suffixes."""
    names = [str(c) for c in df.columns]
    taken, seen, counters, out = set(names), set(), {}, []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
            continue
        k = counters.get(n, 0)
        while True:
            k += 1
            cand = f"{n}.{k}"
            if cand not in taken:
                break
        counters[n] = k
        taken.add(cand)
        seen.add(cand)
        out.append(cand)
    df.columns = out
    return df


def read_tabular(source, name: str = "", nrows=None, kind=None):
    """Shared CSV/XLSX reader used by ``load_data`` and ``load_full_data`` (they must parse identically).

    ``source``: path, bytes or file-like. ``kind``: "csv" | "xlsx" (default: from ``name`` suffix, csv
    unless it ends in .xlsx). Returns ``(df, notices)``; ``df`` has str column names, no dtype downcast.
    """
    notices = []
    if kind is None:
        kind = "xlsx" if str(name).lower().endswith(".xlsx") else "csv"
    data = _source_bytes(source)
    if kind == "xlsx":
        df = pd.read_excel(io.BytesIO(data), nrows=nrows)
        return _stringify_columns(df), notices

    encoding, enc_notice = _detect_encoding(data)
    if enc_notice:
        notices.append(enc_notice)
    sep = _sniff_delimiter(data, encoding)
    # skipinitialspace: `a, "b"` -> quotes are honoured and padding after a delimiter is dropped
    # (so unquoted values like `a, x` lose their leading space; trailing spaces are kept).
    df = pd.read_csv(io.BytesIO(data), nrows=nrows, encoding=encoding, sep=sep, skipinitialspace=True)
    if sep != ",":
        notices.append(f"Detected {_DELIM_NAMES[sep]}-delimited file.")
    if sep == ";":
        df, converted = _convert_decimal_commas(df)
        if converted:
            shown = ", ".join(converted[:5]) + ("..." if len(converted) > 5 else "")
            notices.append(f"Read decimal commas as numbers in: {shown}.")
    return _stringify_columns(df), notices


def _kind_for_buffer(obj):
    """'csv' / 'xlsx' / None / 'fallback' with the original load_data type+name precedence."""
    if isinstance(obj, str):
        suffix = Path(obj).suffix.lower()
        return "csv" if suffix == ".csv" else "xlsx" if suffix == ".xlsx" else None
    if hasattr(obj, "type"):
        name = getattr(obj, "name", "").lower()
        if obj.type == "text/csv" or name.endswith(".csv"):
            return "csv"
        if obj.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or name.endswith(".xlsx"):
            return "xlsx"
        return None
    return "fallback"


@st.cache_data(hash_funcs={
    CachedFileWrapper: _hash_file_or_buffer,
    io.BytesIO: _hash_file_or_buffer,
    io.StringIO: _hash_file_or_buffer,
})
def load_data(file_path_or_buffer, nrows=None):
    """Loads data from a file path or buffer (CSV or Excel).

    User-facing notices (encoding / delimiter detection) are carried in ``df.attrs["lumi_notices"]``.
    """
    try:
        kind = _kind_for_buffer(file_path_or_buffer)
        if kind is None:
            return pd.DataFrame()
        if kind == "fallback":
            st.warning("Could not determine file type, attempting to read as CSV. May fail for other formats.")
            kind = "csv"
        name = file_path_or_buffer if isinstance(file_path_or_buffer, str) else getattr(file_path_or_buffer, "name", "")
        df, notices = read_tabular(file_path_or_buffer, name, nrows=nrows, kind=kind)
        df = downcast_dtypes(df)
        if notices:
            df.attrs["lumi_notices"] = list(notices)
        return df

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error loading data: {type(e).__name__} - {e}. Please check the file format and content.")
        return pd.DataFrame()
    except UnicodeDecodeError:
        st.error("Could not read this file: its text encoding is not supported. Please re-save it as UTF-8 CSV or as an XLSX file.")
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
        st.session_state.full_export = None
        st.session_state.full_source = None
        st.session_state.sample_export = None
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
        'load_notices': [],
        'session_id': None,
        'filename': None,
        'raw_data': st.session_state.raw_data,
        'original_full_data': st.session_state.original_full_data,
        'full_export': st.session_state.get('full_export'),
        'full_source': st.session_state.get('full_source'),
        'sample_export': st.session_state.get('sample_export'),

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
