import pandas as pd
import numpy as np
import re
from typing import TypedDict, Optional, Any, Union, List, Tuple
from rule_utils import evaluate_rule
from query_guard import unsafe_regex_reason, safe_regex_replace, regex_deadline, RegexTimeoutError

KNN_DEFAULT_NEIGHBORS = 5
# Above this many rows KNN/Iterative imputers are fitted on a random sample (they are O(n^2)),
# then applied to the full frame.
KNN_MAX_ROWS = 20000
FUZZY_MATCH_THRESHOLD = 85
FUZZY_TOTAL_BUDGET = 3000  # max unique values fuzzy-deduped across all columns of one 'All' step


def _normalize_one_name(orig, method: str) -> str:
    orig = str(orig)  # headers can be ints/floats/None
    if method == 'snake_case':
        val = re.sub(r'[ \-.]', '_', orig.strip())
        val = re.sub(r'[^a-zA-Z0-9_]', '', val)
        val = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', val)
        val = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', val)
        val = re.sub(r'_+', '_', val).lower().strip('_')
        if val[:1].isdigit():
            val = 'col_' + val
    elif method == 'lowercase':
        val = orig.lower()
    elif method == 'uppercase':
        val = orig.upper()
    elif method == 'remove_spaces':
        val = orig.replace(' ', '')
    else:
        val = orig
    if not val:
        val = f"column_{orig}"
    return val


def _dedupe_names(names: list) -> list:
    """Make names unique: later repeats get _2, _3, ... suffixes that never collide with any name in the list."""
    reserved = set(names)
    used = set()
    out = []
    for name in names:
        if name in used:
            n = 2
            while f"{name}_{n}" in reserved or f"{name}_{n}" in used:
                n += 1
            name = f"{name}_{n}"
        used.add(name)
        out.append(name)
    return out


def normalized_column_names(columns: list, method: str) -> list:
    """New names for *columns* (same order, always unique); duplicate results get _2, _3 suffixes."""
    return _dedupe_names([_normalize_one_name(c, method) for c in columns])


def predict_column_renames(columns: list, method: str, only_changed: bool = False) -> dict:
    """Predicts how column names will be changed by the normalization logic."""
    new_names = {}
    for orig, val in zip(columns, normalized_column_names(columns, method)):
        if orig in new_names:
            continue
        if not only_changed or val != orig:
            new_names[orig] = val
    return new_names

class RuleDef(TypedDict, total=False):
    type: str
    desc: str
    enabled: bool
    color: str
    col: str
    min: float
    max: float
    col_a: str
    op: str
    col_b: str
    target_type: str
    value: Any
    query: str
    resolved: bool

class CleaningStep(TypedDict, total=False):
    action: str
    column: str
    value: Any
    min: float
    max: float
    dtype: str
    rule: RuleDef
    find: str
    replace: str
    regex: bool
    new_column: str
    component: str

def _handle_drop_column(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        return df.drop(columns=[col]), []
    return df, [f"Warning: Column '{col}' not found for drop_column action."]

def _handle_drop_nulls(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        return df.dropna(subset=[col]), []
    return df, [f"Warning: Column '{col}' not found for drop_nulls action."]

def _handle_fill_null(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col not in df.columns:
        return df, [f"Warning: Column '{col}' not found for fill_null action."]
    
    val = step.get('value')
    fill_value = None
    if val in ["mean", "median"]:
        try:
            num_s = pd.to_numeric(df[col], errors='coerce')
            if pd.api.types.is_float_dtype(num_s):
                num_s = num_s.replace([np.inf, -np.inf], np.nan)  # +/-inf must not poison the mean/median
            if num_s.isnull().all():
                return df, [f"Warning: Cannot calculate {val} for non-numeric column '{col}'."]
            fill_value = num_s.mean() if val == "mean" else num_s.median()
            if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                fill_value = int(round(fill_value))  # integer columns can't hold a fractional fill
        except Exception as e:
            return df, [f"Warning: Cannot calculate {val} for column '{col}': {e}"]
    elif val == "mode":
        mode_result = df[col].mode()
        if not mode_result.empty:
            fill_value = mode_result.iloc[0]
    elif val in ["knn", "iterative"]:
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import KNNImputer, IterativeImputer
            imputer = KNNImputer(n_neighbors=KNN_DEFAULT_NEIGHBORS) if val == "knn" else IterativeImputer(random_state=42)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in numeric_cols:
                return df, [f"Error: {val.upper()} imputation requires a numeric column. '{col}' is {df[col].dtype}."]
            if df[col].isna().all():
                return df, [f"Warning: {val.upper()} imputation skipped on '{col}': the column is entirely empty."]

            # All-empty and +/-inf columns make the imputers fail; leave them out of the feature set.
            def _usable_feature(c):
                f = df[c].astype('float64')
                return bool(f.notna().any()) and not bool(np.isinf(f).any())
            numeric_cols = [c for c in numeric_cols if c == col or _usable_feature(c)]

            # Select target column and up to 10 other numeric columns (sorted by correlation) to avoid performance bottlenecks
            other_numeric = [c for c in numeric_cols if c != col]
            if len(other_numeric) > 10:
                corrs = df[numeric_cols].corrwith(df[col]).abs().fillna(0)
                sorted_others = sorted(other_numeric, key=lambda x: corrs.get(x, 0), reverse=True)
                selected_cols = [col] + sorted_others[:10]
            else:
                selected_cols = numeric_cols
            
            df_numeric = df[selected_cols].copy()
            if len(df_numeric) > KNN_MAX_ROWS:
                imputer.fit(df_numeric.sample(n=KNN_MAX_ROWS, random_state=42))
                df_imputed = pd.DataFrame(imputer.transform(df_numeric), columns=selected_cols, index=df.index)
            else:
                df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=selected_cols, index=df.index)
            df[col] = df_imputed[col]
            return df, []
        except Exception as e:
            return df, [f"Error applying {val.upper()} imputation on {col}: {str(e)}"]
    else:
        fill_value = val
    
    if fill_value is not None:
        df[col] = df[col].fillna(fill_value)
        return df, []
    return df, [f"Warning: Could not determine fill value for column '{col}' with strategy '{val}'."]

def _cap_series(s: pd.Series, lo, hi) -> pd.Series:
    """Clip *s* to [lo, hi] without pandas' silent int->float upcast.

    dtype policy: integer column -> stays integer (bounds are rounded inward: lower bound up,
    upper bound down, then clamped to the dtype's range so e.g. int8 is preserved). Only when
    rounding leaves no integer inside [lo, hi] is the column converted to float64.
    Float columns stay float. Must mirror codegen._DOWNCAST_HELPER's _cap_series.
    """
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        import math
        lo_i = None if lo is None or not math.isfinite(float(lo)) else math.ceil(lo)
        hi_i = None if hi is None or not math.isfinite(float(hi)) else math.floor(hi)
        if lo_i is None or hi_i is None or lo_i <= hi_i:
            info = np.iinfo(s.dtype.numpy_dtype if hasattr(s.dtype, 'numpy_dtype') else s.dtype)
            lo_i = None if lo_i is None else max(int(lo_i), int(info.min))
            hi_i = None if hi_i is None else min(int(hi_i), int(info.max))
            return s.clip(lower=lo_i, upper=hi_i)
        s = s.astype('float64')
    return s.clip(lower=lo, upper=hi)

def _handle_cap_range(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        df[col] = _cap_series(df[col], step['min'], step['max'])
        return df, []
    return df, [f"Warning: Column '{col}' not found for cap_range action."]

def _numeric_to_datetime(s: pd.Series, col) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Convert a numeric series to datetime without the 'ints are nanoseconds' trap.

    Returns (datetime_series, None) when all non-null values look like years
    (1000-2999 -> '%Y') or YYYYMMDD dates (8 digits -> '%Y%m%d'); otherwise
    (None, warning). Must mirror codegen._DT_HELPER.
    """
    nn = s.dropna()
    if len(nn) == 0:
        return pd.to_datetime(s, errors='coerce'), None
    f = nn.astype('float64')
    if not (f % 1 == 0).all():
        return None, (f"Warning: '{col}' has non-integer numeric values; not converted to datetime "
                      f"(numbers would be read as nanoseconds since 1970). Column left unchanged.")
    if f.between(1000, 2999).all():
        fmt = '%Y'
    elif f.between(10000101, 29991231).all():
        fmt = '%Y%m%d'
    else:
        return None, (f"Warning: '{col}' is numeric but its values are neither years (1000-2999) nor "
                      f"YYYYMMDD dates; not converted to datetime. Column left unchanged.")
    strs = s.map(lambda v: None if pd.isna(v) else str(int(v)))
    return pd.to_datetime(strs, format=fmt, errors='coerce'), None

def _text_to_datetime(s):
    """Parse text dates; returns ``(series, n_unparsed, n_two_digit)`` (export embeds this source).

    Two-digit years ('Aug-62', '15/01/05', '3-Mar-98') are read with a 50-year pivot
    (00-49 -> 20xx, 50-99 -> 19xx); ``n_two_digit`` counts those reinterpreted values.
    pandas infers one format from the first value and turns every other format into NaT, so
    when that loses values, parse each value on its own (``format='mixed'``) instead.
    ``n_unparsed`` counts non-blank values that still became NaT.
    """
    import re
    months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
    my_re = re.compile(r'^(\s*)([A-Za-z]{3,9})(\.?[\s\-/,]+)(\d{2})(\s*)$')
    dmy_re = re.compile(r'^(\s*\d{1,2}[\s\-/.](?:\d{1,2}|[A-Za-z]{3,9})[\s\-/.])(\d{2})(\s*(?:\d{1,2}:\d{2}.*)?)$')
    n_two = 0

    def _full_year(yy):
        return ('20' if int(yy) < 50 else '19') + yy

    def _expand(v):
        nonlocal n_two
        if not isinstance(v, str):
            return v
        m = my_re.match(v)
        if m and m.group(2)[:3].lower() in months:
            n_two += 1
            return m.group(1) + m.group(2) + m.group(3) + _full_year(m.group(4)) + m.group(5)
        m = dmy_re.match(v)
        if m:
            n_two += 1
            return m.group(1) + _full_year(m.group(2)) + m.group(3)
        return v

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        s = s.map(_expand)

    def _unparsed(out):
        blank = s.map(lambda v: isinstance(v, str) and not v.strip()).astype(bool)
        return int((s.notna() & ~blank & out.isna()).sum())
    out = pd.to_datetime(s, errors='coerce')
    lost = _unparsed(out)
    if lost:
        mixed = pd.to_datetime(s, errors='coerce', format='mixed')
        mixed_lost = _unparsed(mixed)
        if mixed_lost < lost:
            out, lost = mixed, mixed_lost
    return out, lost, n_two


def _unparsed_dates_msg(col, n):
    return f"Warning: {n} value(s) in '{col}' could not be read as dates and became empty (NaT)."


def _two_digit_years_msg(col, n):
    return (f"Warning: {n} value(s) in '{col}' had two-digit years; read as 20xx for 00-49 "
            f"and 19xx for 50-99.")


def _is_plain_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

def _handle_cast_type(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col not in df.columns:
        return df, [f"Warning: Column '{col}' not found for cast_type action."]
    
    try:
        target_dtype = step['dtype']
        if target_dtype in ['int64', 'int32', 'int']:
            target_dtype = "Int64"
        
        if step['dtype'] == "datetime64[ns]":
            if pd.api.types.is_bool_dtype(df[col]):
                return df, [f"Warning: '{col}' is boolean; not converted to datetime. Column left unchanged."]
            if _is_plain_numeric(df[col]):
                converted, warn = _numeric_to_datetime(df[col], col)
                if converted is None:
                    return df, [warn]
                df[col] = converted
            else:
                df[col], lost, n_two = _text_to_datetime(df[col])
                msgs = []
                if n_two:
                    msgs.append(_two_digit_years_msg(col, n_two))
                if lost:
                    msgs.append(_unparsed_dates_msg(col, lost))
                if msgs:
                    return df, msgs
        elif target_dtype in ['string', 'object']:
            df[col] = df[col].astype(target_dtype)
        else:
            src = df[col]
            if _is_text_dtype(src):
                # '25,649' -> 25649, but only when every non-blank value is a clean number
                # (optionally with proper thousands separators) and at least one has a comma.
                is_str = src.map(lambda v: isinstance(v, str) and v.strip() != '').astype(bool)
                txt = src[is_str].str.strip()
                if (txt.str.contains(',', regex=False).any()
                        and txt.str.match(r'^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+(\.\d+)?$').all()):
                    src = src.copy()
                    src[is_str] = txt.str.replace(',', '', regex=False)
            numeric_series = pd.to_numeric(src, errors='coerce')
            blank = df[col].map(lambda v: isinstance(v, str) and v.strip() == '').astype(bool)
            n_lost = int((df[col].notna() & ~blank & numeric_series.isna()).sum())
            df[col] = numeric_series.astype(target_dtype)
            if n_lost:
                return df, [f"Warning: {n_lost} value(s) in '{col}' could not be converted and became empty."]
        return df, []
    except Exception as e:
        return df, [f"Error: Could not cast '{col}' to {step['dtype']}: {str(e)}"]

def _handle_drop_violated(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    rule = step.get('rule')
    if not rule or rule.get('type') == "Informational":
        return df, []
    try:
        violation_mask = evaluate_rule(df, rule)
        return df[~violation_mask], []
    except Exception as e:
        return df, [f"Error: Could not apply drop_violated rule ({rule.get('desc', 'N/A')}): {str(e)}"]

def _replace_operands(s, find, repl):
    """``(find, repl, warning)`` for a plain (non-regex) replace on series ``s`` (export embeds this source).

    The UI sends the find/replace values as text; on a numeric column a find value that parses
    as a number is compared numerically (so '5' matches 5 and 2.0). The replacement must then
    parse the same way (and be whole for an integer column), otherwise ``warning`` is set.
    """
    import math
    if not (isinstance(find, str) and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
        return find, repl, None
    as_int = pd.api.types.is_integer_dtype(s)

    def _num(x):
        v = float(x.strip())
        if not math.isfinite(v):
            raise ValueError(x)
        if v.is_integer() and as_int:
            return int(v)
        if as_int:
            raise ValueError(x)
        return v

    try:
        f = float(find.strip())
        if not math.isfinite(f):
            return find, repl, None
        f = int(f) if as_int and f.is_integer() else f
    except ValueError:
        return find, repl, None
    try:
        r = _num(repl) if isinstance(repl, str) else repl
    except ValueError:
        return find, repl, f"replacement value {repl!r} is not a valid number for this numeric column"
    return f, r, None


def _handle_replace(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    f, r_val = step.get('find'), step.get('replace')
    use_regex = step.get('regex', False)
    if use_regex:
        why = unsafe_regex_reason(f)
        if why:
            return df, [f"Warning: Replace on '{col}' skipped: unsafe regular expression ({why})."]
    
    deadline = regex_deadline() if use_regex else None  # one budget for the whole step
    no_match = f"Warning: no values matched '{f}' in '{col}'; nothing was replaced."

    def _replace(c, numeric_operands=True):
        """(new_series, matched) for one column."""
        if use_regex:
            new = safe_regex_replace(df[c], f, r_val, deadline=deadline)
            return new, not new.equals(df[c])
        ff, rr = f, r_val
        if numeric_operands:
            ff, rr, bad = _replace_operands(df[c], f, r_val)
            if bad:
                raise ValueError(bad)
        matched = int(df[c].eq(ff).sum()) > 0
        return df[c].replace(ff, rr, regex=False), matched

    if col == "All":
        warnings = []
        any_match = False
        for c in df.select_dtypes(include=['object', 'string']).columns:
            orig_dtype = df[c].dtype
            try:
                new, matched = _replace(c, numeric_operands=False)
                df[c] = new.astype(orig_dtype)
                any_match = any_match or matched
            except RegexTimeoutError as e:
                warnings.append(f"Warning: Replace on '{c}' skipped: {e}; the column is unchanged.")
        if not any_match and not warnings:
            warnings.append(no_match)
        return df, warnings
    elif col in df.columns:
        orig_dtype = df[col].dtype
        try:
            new, matched = _replace(col)
            df[col] = new.astype(orig_dtype)
        except RegexTimeoutError as e:
            return df, [f"Warning: Replace on '{col}' skipped: {e}; the column is unchanged."]
        except ValueError as e:
            if use_regex:
                raise
            return df, [f"Warning: Replace on '{col}' skipped: {e}; the column is unchanged."]
        return df, ([] if matched else [no_match])
    return df, [f"Warning: Column '{col}' not found for replace action."]

def _is_text_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

def _apply_to_str(s: pd.Series, fn) -> pd.Series:
    """Apply fn(<Series of the str cells>) to the actual str cells only.

    Nulls and non-str values (ints, bools, ... inside object columns) are left untouched,
    never stringified. Must mirror codegen's exported _apply_to_str.
    """
    is_str = s.map(lambda v: isinstance(v, str)).astype(bool)
    if not is_str.any():
        return s
    out = s.copy()
    out.loc[is_str] = fn(s.loc[is_str]).to_numpy()
    return out

def _handle_strip_whitespace(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col == "All":
        for c in df.select_dtypes(include=['object', 'string']).columns:
            df[c] = _apply_to_str(df[c], lambda x: x.str.strip())
        return df, []
    elif col in df.columns:
        if not _is_text_dtype(df[col]):
            return df, [f"Warning: strip_whitespace skipped on non-text column '{col}' ({df[col].dtype})."]
        df[col] = _apply_to_str(df[col], lambda x: x.str.strip())
        return df, []
    return df, [f"Warning: Column '{col}' not found for strip_whitespace action."]

def _fuzzy_dedupe_mapping(s, threshold):
    """``{variant: canonical}`` for near-duplicate text values in ``s`` (export embeds this source).

    The most frequent spelling is canonical (ties broken alphabetically). Values whose digit
    runs differ are never merged, so IDs such as 'Product 10' and 'Product 11' stay distinct.
    Two multi-word values are merged only when every word of each fuzzily matches a word of
    the other, so 'No Hair' and 'Silver Hair' stay distinct while 'Jon Smith'/'John Smith' merge.
    """
    import re
    from thefuzz import fuzz, process
    counts = s[s.map(lambda v: isinstance(v, str))].value_counts()
    vals = [v for v, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]  # most frequent first

    def _words(v):
        return re.findall(r"[a-z0-9]+", v.lower())

    def _same_words(a, b):
        wa, wb = _words(a), _words(b)
        if len(wa) < 2 or len(wb) < 2:
            return True
        return (all(any(fuzz.ratio(x, y) >= 75 for y in wb) for x in wa)
                and all(any(fuzz.ratio(x, y) >= 75 for y in wa) for x in wb))

    mapping, handled = {}, set()
    for v in vals:
        if v in handled:
            continue
        handled.add(v)
        digits = re.findall(r"\d+", v)
        for match, score in process.extract(v, vals, limit=10):
            if (score > threshold and match not in handled and re.findall(r"\d+", match) == digits
                    and _same_words(v, match)):
                mapping[match] = v
                handled.add(match)
    return mapping


def _title_case(x):
    """Title-case a Series of str; English contraction suffixes stay lower ("don't" -> "Don't",
    "they're" -> "They're") while names keep their capital ("o'brien" -> "O'Brien").

    Must mirror codegen._TITLE_OP.
    """
    return x.str.title().str.replace(r"(?<=[^\W\d_])(['\u2019])(T|S|D|M|Ll|Re|Ve)\b", lambda m: m.group(1) + m.group(2).lower(), regex=True)


def _handle_normalize_text(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    method = step.get('value', 'lowercase')
    messages = []
    budget = {'left': FUZZY_TOTAL_BUDGET}  # unique values still allowed across all columns ("All")

    def normalize_series(s, orig_dtype, col_name):
        # Only real str cells are transformed; NaN/None/pd.NA and non-str values are preserved.
        if method == "lowercase": return _apply_to_str(s, lambda x: x.str.lower())
        elif method == "uppercase": return _apply_to_str(s, lambda x: x.str.upper())
        elif method == "titlecase": return _apply_to_str(s, _title_case)
        elif method == "remove_punctuation":
            import string
            return _apply_to_str(s, lambda x: x.str.replace('[' + re.escape(string.punctuation) + ']', '', regex=True))
        elif method == "fuzzy_dedupe":
            unique_vals = [v for v in s.dropna().unique() if isinstance(v, str)]
            if len(unique_vals) > 1000:
                messages.append(f"Fuzzy deduplication skipped on '{col_name}' because it has too many unique values ({len(unique_vals)} > 1000) to prevent performance hangs.")
                return s
            if col == "All":
                if len(unique_vals) > budget['left']:
                    messages.append(f"Fuzzy deduplication skipped on '{col_name}' because the total unique-value budget across columns ({FUZZY_TOTAL_BUDGET}) is used up.")
                    return s
                budget['left'] -= len(unique_vals)
            return s.replace(_fuzzy_dedupe_mapping(s, FUZZY_MATCH_THRESHOLD)).astype(orig_dtype)
        return s

    if col == "All":
        for c in df.select_dtypes(include=['object', 'string']).columns:
            df[c] = normalize_series(df[c], df[c].dtype, c)
        return df, messages
    elif col in df.columns:
        if not _is_text_dtype(df[col]):
            # Text ops on numeric/datetime/bool columns corrupt values (e.g. remove_punctuation turns 1.5 into 15).
            return df, [f"Warning: normalize_text skipped on non-text column '{col}' ({df[col].dtype})."]
        df[col] = normalize_series(df[col], df[col].dtype, col)
        return df, messages
    return df, [f"Warning: Column '{col}' not found for normalize_text action."]

def _handle_log_transform(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_neg = int((df[col] < 0).sum())
            df[col] = np.log1p(df[col].clip(lower=0))
            if n_neg:
                return df, [f"Warning: {n_neg} negative value(s) in '{col}' were set to 0 before the log transform."]
            return df, []
        return df, [f"Error: Log transformation requires a numeric column. '{col}' is {df[col].dtype}."]
    return df, [f"Warning: Column '{col}' not found for log_transform action."]

def _handle_rename_column(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    new_name = step.get('value')
    if col in df.columns:
        if new_name in df.columns and new_name != col:
            return df, [f"Warning: Cannot rename '{col}' to '{new_name}': a column with that name already exists."]
        return df.rename(columns={col: new_name}), []
    return df, [f"Warning: Column '{col}' not found for rename_column action."]

def _handle_reorder_columns(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    new_order = step.get('value')
    if not isinstance(new_order, list):
        return df, ["Error: Reorder columns action requires a list of column names."]
    existing_order = [c for c in new_order if c in df.columns]
    missing_cols = [c for c in df.columns if c not in existing_order]
    final_order = existing_order + missing_cols
    return df[final_order], []

def _handle_extract_datetime(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    new_col = step.get('new_column')
    component = step.get('component')
    
    if col not in df.columns:
        return df, [f"Warning: Column '{col}' not found for extract_datetime action."]
    if not new_col:
        return df, ["Error: Target column name not specified for extract_datetime."]
        
    messages = []
    try:
        # Coerce to datetime if not already datetime
        if _is_plain_numeric(df[col]):
            dt_series, warn = _numeric_to_datetime(df[col], col)
            if dt_series is None:
                return df, [warn]
        elif not pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_series, lost, n_two = _text_to_datetime(df[col])
            if n_two:
                messages.append(_two_digit_years_msg(col, n_two))
            if lost:
                messages.append(_unparsed_dates_msg(col, lost))
        else:
            dt_series = df[col]
            
        if component == "year":
            df[new_col] = dt_series.dt.year
        elif component == "month":
            df[new_col] = dt_series.dt.month
        elif component == "day":
            df[new_col] = dt_series.dt.day
        elif component == "day_of_week":
            df[new_col] = dt_series.dt.day_name()
        elif component == "hour":
            df[new_col] = dt_series.dt.hour
        else:
            return df, [f"Error: Unknown datetime component '{component}'."]
            
        return df, messages
    except Exception as e:
        return df, [f"Error: Could not extract datetime component '{component}' from '{col}': {str(e)}"]

def _handle_drop_duplicates(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    initial_rows = len(df)
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    return df, [f"Removed {dropped} duplicate rows."]

def _handle_drop_empty_columns(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    if len(df) == 0:
        # isnull().all() is vacuously True on zero rows; dropping every column would destroy the schema.
        return df, ["Warning: drop_empty_columns skipped: the dataset has no rows."]
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        return df, [f"Dropped empty columns: {', '.join(empty_cols)}."]
    return df, ["No empty columns found to drop."]

def _handle_drop_empty_rows(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    initial_rows = len(df)
    if df.shape[1] == 0:
        return df, ["Removed 0 empty rows."]
    df = df.dropna(how='all')
    dropped = initial_rows - len(df)
    return df, [f"Removed {dropped} empty rows."]

def _handle_normalize_column_names(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    method = step.get('value', 'snake_case')
    if method not in ('snake_case', 'lowercase', 'uppercase', 'remove_spaces'):
        # Same as the export: an unknown method changes nothing.
        return df, [f"Warning: Unknown column-name normalization method '{method}': no change."]
    df = df.copy()
    df.columns = normalized_column_names(df.columns.tolist(), method)
    return df, []

TRANSFORM_REGISTRY = {
    "drop_column": _handle_drop_column,
    "drop_nulls": _handle_drop_nulls,
    "fill_null": _handle_fill_null,
    "cap_range": _handle_cap_range,
    "cast_type": _handle_cast_type,
    "drop_violated": _handle_drop_violated,
    "replace": _handle_replace,
    "strip_whitespace": _handle_strip_whitespace,
    "normalize_text": _handle_normalize_text,
    "log_transform": _handle_log_transform,
    "rename_column": _handle_rename_column,
    "reorder_columns": _handle_reorder_columns,
    "extract_datetime": _handle_extract_datetime,
    "drop_duplicates": _handle_drop_duplicates,
    "drop_empty_columns": _handle_drop_empty_columns,
    "drop_empty_rows": _handle_drop_empty_rows,
    "normalize_column_names": _handle_normalize_column_names,
}

def apply_recipe(df: pd.DataFrame, recipe: List[CleaningStep]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies a sequence of cleaning steps to a DataFrame using a dispatcher pattern.
    """
    df_clean = df.copy()
    messages = []
    for step in recipe:
        action = step.get('action')
        handler = TRANSFORM_REGISTRY.get(action)
        if handler:
            try:
                df_clean, step_messages = handler(df_clean, step)
                messages.extend(step_messages)
            except Exception as e:
                messages.append(f"An unexpected error occurred applying {action}: {type(e).__name__} - {str(e)}")
        else:
            messages.append(f"Warning: Unknown action '{action}' encountered in recipe.")
    return df_clean, messages
