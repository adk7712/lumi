import pandas as pd
import numpy as np
from typing import TypedDict, Optional, Any, Union, List, Tuple
from rule_utils import evaluate_rule

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
    if val == "mean":
        fill_value = df[col].mean()
    elif val == "median":
        fill_value = df[col].median()
    elif val == "mode":
        mode_result = df[col].mode()
        if not mode_result.empty:
            fill_value = mode_result[0]
    elif val in ["knn", "iterative"]:
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import KNNImputer, IterativeImputer
            imputer = KNNImputer(n_neighbors=5) if val == "knn" else IterativeImputer(random_state=42)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in numeric_cols:
                return df, [f"Error: {val.upper()} imputation requires a numeric column. '{col}' is {df[col].dtype}."]
            
            df_numeric = df[numeric_cols].copy()
            df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols, index=df.index)
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

def _handle_cap_range(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col in df.columns:
        df.loc[df[col] < step['min'], col] = step['min']
        df.loc[df[col] > step['max'], col] = step['max']
        return df, []
    return df, [f"Warning: Column '{col}' not found for cap_range action."]

def _handle_cast_type(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col not in df.columns:
        return df, [f"Warning: Column '{col}' not found for cast_type action."]
    
    try:
        target_dtype = step['dtype']
        if target_dtype in ['int64', 'int32', 'int']:
            target_dtype = "Int64"
        
        if step['dtype'] == "datetime64[ns]":
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
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].replace(f, r_val, regex=use_regex)
        return df, []
    elif col in df.columns:
        df[col] = df[col].replace(f, r_val, regex=use_regex)
        return df, []
    return df, [f"Warning: Column '{col}' not found for replace action."]

def _handle_strip_whitespace(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    if col == "All":
        for c in df.select_dtypes(include=['object']).columns:
            mask = df[c].notnull()
            df.loc[mask, c] = df.loc[mask, c].astype(str).str.strip()
        return df, []
    elif col in df.columns:
        mask = df[col].notnull()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
        return df, []
    return df, [f"Warning: Column '{col}' not found for strip_whitespace action."]

def _handle_normalize_text(df: pd.DataFrame, step: CleaningStep) -> Tuple[pd.DataFrame, List[str]]:
    col = step.get('column')
    method = step.get('value', 'lowercase')
    
    def normalize_series(s):
        if method == "lowercase": return s.astype(str).str.lower()
        if method == "uppercase": return s.astype(str).str.upper()
        if method == "titlecase": return s.astype(str).str.title()
        if method == "remove_punctuation": 
            import string
            return s.astype(str).str.replace(f'[{string.punctuation}]', '', regex=True)
        if method == "fuzzy_dedupe":
            from thefuzz import process
            unique_vals = s.dropna().unique()
            mapping = {}
            handled = set()
            for v in unique_vals:
                if v in handled: continue
                matches = process.extract(v, unique_vals, limit=10)
                for match, score in matches:
                    if score > 85: # Threshold for fuzzy matching
                        mapping[match] = v
                        handled.add(match)
            return s.replace(mapping)
        return s

    if col == "All":
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = normalize_series(df[c])
        return df, []
    elif col in df.columns:
        df[col] = normalize_series(df[col])
        return df, []
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
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
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
}
