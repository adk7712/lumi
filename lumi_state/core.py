"""Data loading, dtype downcasting and session-state initialisation."""
import streamlit as st
import pandas as pd
from pathlib import Path
import io
import logging
import re
import warnings
import zipfile

import numpy as np

logger = logging.getLogger("lumi.state")

# Define constants
MAX_SAMPLE_ROWS = 10000
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024 # 50MB
# XLSX is a zip: pandas inflates every sheet fully into memory, so cap what a file may expand to.
MAX_XLSX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024
MAX_XLSX_COMPRESSION_RATIO = 100
_XLSX_RATIO_MIN_BYTES = 20 * 1024 * 1024  # the ratio test only applies once the content is sizeable


class UnsafeFileError(ValueError):
    """Raised for an upload that is rejected before parsing (e.g. a zip-bomb style XLSX)."""


def check_xlsx_safe(source) -> None:
    """Reject an XLSX whose zip entries would inflate past the cap, before pandas parses it.

    ``source``: path or bytes. A non-zip payload is left for the parser to report.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    except (zipfile.BadZipFile, OSError):
        return
    with zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        packed = sum(i.compress_size for i in infos) or 1
    if total > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise UnsafeFileError(
            f"This Excel file expands to {total // (1024 * 1024)} MB of data, above the "
            f"{MAX_XLSX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit. Please export it as CSV or split it."
        )
    if total > _XLSX_RATIO_MIN_BYTES and total / packed > MAX_XLSX_COMPRESSION_RATIO:
        raise UnsafeFileError(
            "This Excel file is compressed far more than a normal spreadsheet and was not opened. "
            "Please re-save it from Excel or export it as CSV."
        )


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts integer columns to more memory-efficient types.

    Floats stay float64: float32 shows noise such as 214.34 -> 214.33999633789062.
    Mirrored by ``codegen._DOWNCAST_HELPER`` (engine == export).
    """
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
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


# Raw text that a numeric parse would change: a leading zero (zip codes, ids such as 00501) or
# an integer too long for float64 to hold exactly (it becomes float as soon as a value is missing).
_LEADING_ZERO_RE = r"^[+-]?0\d"
_LONG_INT_RE = r"^[+-]?\d{16,}$"
_TEXT_HINT_BYTES_RE = re.compile(rb'(?:^|[,;\t|"\s])[+-]?0\d|\d{16}', re.MULTILINE)


def _keep_text_columns(df: pd.DataFrame, reread) -> pd.DataFrame:
    """Numeric columns whose raw text a number would change go back to their text values.

    ``reread(positions)`` returns those columns (same order) read as str with default NA
    handling, so ``n/a`` is still missing. Other numeric columns stay numeric.
    """
    cand = [i for i in range(df.shape[1])
            if pd.api.types.is_numeric_dtype(df.iloc[:, i]) and not pd.api.types.is_bool_dtype(df.iloc[:, i])]
    if not cand:
        return df
    raw = reread(cand)
    for j, i in enumerate(cand):
        text = raw.iloc[:, j]
        vals = text.dropna().astype(str).str.strip()
        if vals.empty:
            continue
        lead = vals.str.match(_LEADING_ZERO_RE).any()
        long_int = pd.api.types.is_float_dtype(df.iloc[:, i]) and vals.str.match(_LONG_INT_RE).any()
        if lead or long_int:
            df.isetitem(i, text.astype(object).where(text.notna(), np.nan).to_numpy())
    return df


_MARKUP_HEADER_RE = re.compile(r"^\s*(\{|\[\s*[\{\[\"]|<[!?/a-zA-Z])")


_STRONG_MARKUP_RE = re.compile(r"^\s*(<[!?]|<(html|head|body|div|table|xml|svg|rss)\b|<(\w+)[^>]*>.*</\3>)", re.I)


def _check_not_markup(df: pd.DataFrame) -> None:
    """JSON / HTML / XML parsed as a CSV comes back as one junk column; refuse it.

    A real table whose first header merely starts with ``<`` or ``{`` (e.g. ``<a>,b``) is kept.
    """
    if not df.shape[1]:
        return
    first = str(df.columns[0])
    if _MARKUP_HEADER_RE.match(first) and (df.shape[1] == 1 or _STRONG_MARKUP_RE.match(first)):
        raise UnsafeFileError(
            "This file looks like JSON or HTML/XML, not a table. Please upload a CSV or XLSX file.")


_MARKUP_TEXT_RE = re.compile(r"^\s*<[!?/]?[a-zA-Z][^>]*>")
_CSV_FIELD_LIMIT = 256 * 1024 * 1024  # bytes per field for the row-width scan (csv's default is 128 KB)


def _check_text_not_markup(data: bytes, encoding: str, sep: str = ",") -> None:
    """Refuse HTML/XML/JSON before the row-width check (tags and commas read as ragged rows).

    A first line with the delimiter outside its tags (``<a>,b``) is a table, unless it is clearly
    markup (doctype, ``<html>``, a closed element).
    """
    head = data[:4096].decode(encoding, errors="ignore").lstrip("\ufeff")
    first = next((ln for ln in head.splitlines() if ln.strip()), "").strip()
    is_json = first in ("[", "{") or first.startswith(("[{", '{"', "[["))
    is_markup = bool(_MARKUP_TEXT_RE.match(first)) and (
        sep not in re.sub(r"<[^>]*>", "", first) or bool(_STRONG_MARKUP_RE.match(first)))
    if is_json or is_markup:
        raise UnsafeFileError(
            "This file looks like JSON or HTML/XML, not a table. Please upload a CSV or XLSX file.")


def _check_row_widths(data: bytes, encoding: str, sep: str, nrows) -> None:
    """Refuse rows with more non-empty fields than the header (pandas would silently drop them).

    Extra fields that are all empty (a trailing delimiter) are fine. Rows are counted like a
    spreadsheet: the header is row 1.
    """
    import csv
    old_limit = csv.field_size_limit(_CSV_FIELD_LIMIT)
    try:
        _scan_row_widths(csv, data, encoding, sep, nrows)
    finally:
        csv.field_size_limit(old_limit)


def _scan_row_widths(csv, data: bytes, encoding: str, sep: str, nrows) -> None:
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(data), encoding=encoding, newline=""),
                        delimiter=sep, quotechar='"', skipinitialspace=True)
    width = None
    seen = 0
    for row in reader:
        if not row:
            continue
        if width is None:
            width = len(row)
            continue
        if len(row) > width and any(f.strip() for f in row[width:]):
            raise UnsafeFileError(f"row {reader.line_num} has {len(row)} fields but the header has {width}. "
                                  "Please fix the delimiter or quoting, then re-upload.")
        seen += 1
        if nrows is not None and seen >= nrows:
            return


def _drop_trailing_empty_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """A trailing delimiter in the header makes an all-empty ``Unnamed: N`` column; drop those."""
    n = df.shape[1]
    while n > 0 and str(df.columns[n - 1]).startswith("Unnamed:") and df.iloc[:, n - 1].isna().all():
        n -= 1
    return df if n == df.shape[1] else df.iloc[:, :n].copy()


def read_tabular(source, name: str = "", nrows=None, kind=None):
    """Shared CSV/XLSX reader used by ``load_data`` and ``load_full_data`` (they must parse identically).

    ``source``: path, bytes or file-like. ``kind``: "csv" | "xlsx" (default: from ``name`` suffix, csv
    unless it ends in .xlsx). Returns ``(df, notices)``; ``df`` has str column names, no dtype downcast.
    Columns whose text a numeric parse would change (leading zeros, long ids) stay text.
    """
    notices = []
    if kind is None:
        kind = "xlsx" if str(name).lower().endswith(".xlsx") else "csv"
    if kind == "xlsx":
        if isinstance(source, str):
            # Path: let zipfile/pandas read from disk instead of holding another full byte copy.
            check_xlsx_safe(source)
            src = lambda: source  # noqa: E731
        else:
            data = _source_bytes(source)
            check_xlsx_safe(data)
            src = lambda: io.BytesIO(data)  # noqa: E731
        try:
            df = pd.read_excel(src(), nrows=nrows)
        except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
            if isinstance(e, UnsafeFileError):
                raise
            raise UnsafeFileError("this is not a valid Excel workbook (.xlsx).") from e
        df = _keep_text_columns(df, lambda cols: pd.read_excel(src(), nrows=nrows, usecols=cols, dtype=str))
        return _stringify_columns(df), notices

    data = _source_bytes(source)
    if data.startswith(b"PK\x03\x04"):
        if str(name).lower().endswith(".xlsx"):
            raise UnsafeFileError("this is not a valid Excel workbook (.xlsx).")
        raise UnsafeFileError("this is a zip/Excel container, not a CSV. Name it .xlsx if it is a workbook.")
    encoding, enc_notice = _detect_encoding(data)
    if enc_notice:
        notices.append(enc_notice)
    sep = _sniff_delimiter(data, encoding)
    # skipinitialspace: `a, "b"` -> quotes are honoured and padding after a delimiter is dropped
    # (so unquoted values like `a, x` lose their leading space; trailing spaces are kept).
    # index_col=False: a trailing delimiter on data rows must not turn the first column into the index.
    if nrows != 0:
        _check_text_not_markup(data, encoding, sep)
        _check_row_widths(data, encoding, sep, nrows)
    csv_kw = dict(nrows=nrows, encoding=encoding, sep=sep, skipinitialspace=True, index_col=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning)  # the empty trailing field is dropped
        df = pd.read_csv(io.BytesIO(data), **csv_kw)
        if _TEXT_HINT_BYTES_RE.search(data):  # cheap pre-check: most files need no second pass
            df = _keep_text_columns(
                df, lambda cols: pd.read_csv(io.BytesIO(data), usecols=cols, dtype=str, **csv_kw))
    _check_not_markup(df)
    df = _drop_trailing_empty_unnamed(df)
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
}, max_entries=8, ttl=1800)
def load_data(file_path_or_buffer, nrows=None):
    """Loads data from a file path or buffer (CSV or Excel).

    User-facing notices (encoding / delimiter detection) are carried in ``df.attrs["lumi_notices"]``.
    A failed load returns an empty frame with the reason in ``df.attrs["lumi_load_error"]``.
    """
    try:
        kind = _kind_for_buffer(file_path_or_buffer)
        if kind is None:
            return _load_failure("Unsupported file type. Please upload a CSV or XLSX file.")
        if kind == "fallback":
            st.warning("Could not determine file type, attempting to read as CSV. May fail for other formats.")
            kind = "csv"
        name = file_path_or_buffer if isinstance(file_path_or_buffer, str) else getattr(file_path_or_buffer, "name", "")
        if kind == "csv" and str(name).lower().endswith(".xlsx"):
            kind = "xlsx"  # fallback buffers: trust the .xlsx name
        if Path(str(name)).suffix.lower() in (".xls", ".xlsm", ".xlsb", ".ods", ".json", ".parquet"):
            return _load_failure("Unsupported file type. Please upload a CSV or XLSX file.")
        df, notices = read_tabular(file_path_or_buffer, name, nrows=nrows, kind=kind)
        if df.shape[1] == 0:
            return _load_failure("Could not read this file: it has no columns (only delimiters or blank headers).")
        df = downcast_dtypes(df)
        if notices:
            df.attrs["lumi_notices"] = list(notices)
        return df

    except pd.errors.EmptyDataError:
        return _load_failure("Could not read this file: it is empty (no column headers or rows).")
    except (FileNotFoundError, pd.errors.ParserError) as e:
        return _load_failure(f"Error loading data: {type(e).__name__} - {e}. Please check the file format and content.")
    except UnsafeFileError as e:
        return _load_failure(f"Could not open this file: {e}")
    except UnicodeDecodeError:
        return _load_failure("Could not read this file: its text encoding is not supported. Please re-save it as UTF-8 CSV or as an XLSX file.")
    except Exception as e:
        return _load_failure(f"An unexpected error occurred: {type(e).__name__} - {e}")


def _load_failure(message: str) -> pd.DataFrame:
    """Empty frame carrying the reason in ``attrs["lumi_load_error"]``.

    Not shown here: ``load_data`` is cached, and an ``st.error`` would only flash before the
    caller's rerun. ``process_uploaded_file`` stores it for the landing page instead.
    """
    logger.info("Upload could not be loaded: %s", message)
    df = pd.DataFrame()
    df.attrs["lumi_load_error"] = message
    return df

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
        # preserved across a reset: the sidebar sets the new workspace's name just before resetting
        '_pending_workspace_name': st.session_state.get('_pending_workspace_name'),
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
