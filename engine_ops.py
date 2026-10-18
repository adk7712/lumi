import pandas as pd
import numpy as np
import re
from typing import TypedDict, Optional, Any, Union, List, Tuple
from rule_utils import evaluate_rule

KNN_DEFAULT_NEIGHBORS = 5
FUZZY_MATCH_THRESHOLD = 85


def predict_column_renames(columns: list, method: str, only_changed: bool = False) -> dict:
    """Predicts how column names will be changed by the normalization logic."""
    new_names = {}
    for orig in columns:
        if method == 'snake_case':
            val = re.sub(r'[^a-zA-Z0-9_]', '', orig.strip().replace(' ', '_').replace('-', '_'))
            val = re.sub(r'_+', '_', val).lower()
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
            if num_s.isnull().all():
                return df, [f"Warning: Cannot calculate {val} for non-numeric column '{col}'."]
            fill_value = num_s.mean() if val == "mean" else num_s.median()
        except Exception as e:
            return df, [f"Warning: Cannot calculate {val} for column '{col}': {e}"]
    elif val == "mode":
        mode_result = df[col].mode()
        if not mode_result.empty:
            fill_value = mode_result[0]
    elif val in ["knn", "iterative"]:
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import KNNImputer, IterativeImputer
            imputer = KNNImputer(n_neighbors=KNN_DEFAULT_NEIGHBORS) if val == "knn" else IterativeImputer(random_state=42)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in numeric_cols:
                return df, [f"Error: {val.upper()} imputation requires a numeric column. '{col}' is {df[col].dtype}."]
            
            # Select target column and up to 10 other numeric columns (sorted by correlation) to avoid performance bottlenecks
            other_numeric = [c for c in numeric_cols if c != col]
            if len(other_numeric) > 10:
                corrs = df[numeric_cols].corrwith(df[col]).abs().fillna(0)
                sorted_others = sorted(other_numeric, key=lambda x: corrs.get(x, 0), reverse=True)
                selected_cols = [col] + sorted_others[:10]
            else:
                selected_cols = numeric_cols
            
            df_numeric = df[selected_cols].copy()
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

    dtype policy: integer column + both bounds integral -> stays integer (bounds
    are clamped to the dtype's range so e.g. int8 is preserved). Integer column
    with a fractional bound -> explicitly converted to float64 (never rounded).
    Float columns stay float. Must mirror codegen._CAP_HELPER.
    """
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        integral = all(b is None or float(b) == int(b) for b in (lo, hi))
        if integral:
            info = np.iinfo(s.dtype.numpy_dtype if hasattr(s.dtype, 'numpy_dtype') else s.dtype)
            lo = None if lo is None else max(int(lo), int(info.min))
            hi = None if hi is None else min(int(hi), int(info.max))
            return s.clip(lower=lo, upper=hi)
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
            if _is_plain_numeric(df[col]):
                converted, warn = _numeric_to_datetime(df[col], col)
                if converted is None:
                    return df, [warn]
                df[col] = converted
            else:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        elif target_dtype in ['string', 'object']:
            df[col] = df[col].astype(target_dtype)
        else:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            df[col] = numeric_series.astype(target_dtype)
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

def _handle_replace(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    f, r_val = step.get('find'), step.get('replace')
    use_regex = step.get('regex', False)
    
    if col == "All":
        for c in df.select_dtypes(include=['object', 'string']).columns:
            orig_dtype = df[c].dtype
            df[c] = df[c].replace(f, r_val, regex=use_regex).astype(orig_dtype)
        return df, []
    elif col in df.columns:
        orig_dtype = df[col].dtype
        df[col] = df[col].replace(f, r_val, regex=use_regex).astype(orig_dtype)
        return df, []
    return df, [f"Warning: Column '{col}' not found for replace action."]

def _handle_strip_whitespace(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col == "All":
        for c in df.select_dtypes(include=['object', 'string']).columns:
            orig_dtype = df[c].dtype
            mask = df[c].notnull()
            df.loc[mask, c] = df.loc[mask, c].astype(str).str.strip().astype(orig_dtype)
        return df, []
    elif col in df.columns:
        orig_dtype = df[col].dtype
        mask = df[col].notnull()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip().astype(orig_dtype)
        return df, []
    return df, [f"Warning: Column '{col}' not found for strip_whitespace action."]

def _handle_normalize_text(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    method = step.get('value', 'lowercase')
    messages = []
    
    def _apply_str(s, fn):
        # Transform non-null values only so NaN/None/pd.NA are preserved (not turned into "nan").
        return fn(s.astype(str)).where(s.notnull(), s)

    def normalize_series(s, orig_dtype, col_name):
        if method == "lowercase": res = _apply_str(s, lambda x: x.str.lower())
        elif method == "uppercase": res = _apply_str(s, lambda x: x.str.upper())
        elif method == "titlecase": res = _apply_str(s, lambda x: x.str.title())
        elif method == "remove_punctuation":
            import string
            res = _apply_str(s, lambda x: x.str.replace(f'[{string.punctuation}]', '', regex=True))
        elif method == "fuzzy_dedupe":
            unique_vals = s.dropna().unique()
            if len(unique_vals) > 1000:
                messages.append(f"Fuzzy deduplication skipped on '{col_name}' because it has too many unique values ({len(unique_vals)} > 1000) to prevent performance hangs.")
                return s
            from thefuzz import process
            mapping = {}
            handled = set()
            for v in unique_vals:
                if v in handled: continue
                matches = process.extract(v, unique_vals, limit=10)
                for match, score in matches:
                    if score > FUZZY_MATCH_THRESHOLD: # Threshold for fuzzy matching
                        mapping[match] = v
                        handled.add(match)
            res = s.replace(mapping)
        else:
            res = s
        return res.astype(orig_dtype)

    if col == "All":
        for c in df.select_dtypes(include=['object', 'string']).columns:
            df[c] = normalize_series(df[c], df[c].dtype, c)
        return df, messages
    elif col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            # Text ops on numeric/datetime columns corrupt values (e.g. remove_punctuation turns 1.5 into 15).
            return df, [f"Warning: normalize_text skipped on non-text column '{col}' ({df[col].dtype})."]
        df[col] = normalize_series(df[col], df[col].dtype, col)
        return df, messages
    return df, [f"Warning: Column '{col}' not found for normalize_text action."]

def _handle_log_transform(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = np.log1p(df[col].clip(lower=0))
            return df, []
        return df, [f"Error: Log transformation requires a numeric column. '{col}' is {df[col].dtype}."]
    return df, [f"Warning: Column '{col}' not found for log_transform action."]

def _handle_rename_column(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    new_name = step.get('value')
    if col in df.columns:
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
        
    try:
        # Coerce to datetime if not already datetime
        if _is_plain_numeric(df[col]):
            dt_series, warn = _numeric_to_datetime(df[col], col)
            if dt_series is None:
                return df, [warn]
        elif not pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_series = pd.to_datetime(df[col], errors='coerce')
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
            
        return df, []
    except Exception as e:
        return df, [f"Error: Could not extract datetime component '{component}' from '{col}': {str(e)}"]

def _handle_drop_duplicates(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    initial_rows = len(df)
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    return df, [f"Removed {dropped} duplicate rows."]

def _handle_drop_empty_columns(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        return df, [f"Dropped empty columns: {', '.join(empty_cols)}."]
    return df, ["No empty columns found to drop."]

def _handle_drop_empty_rows(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    initial_rows = len(df)
    df = df.dropna(how='all')
    dropped = initial_rows - len(df)
    return df, [f"Removed {dropped} empty rows."]

def _handle_normalize_column_names(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    method = step.get('value', 'snake_case')
    new_names = predict_column_renames(df.columns.tolist(), method)
    return df.rename(columns=new_names), []

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
