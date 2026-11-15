import math
import re
import warnings

import pandas as pd
import numpy as np

# Configuration for heuristic thresholds used by the data quality scout.
# These values determine when certain data quality issues are flagged.
SCOUT_THRESHOLDS = {
    # Percentage of null values above which a column is considered for dropping.
    "NULL_DROP_PCT": 90.0,
    # Maximum number of unique values for a column to be considered constant.
    "CONSTANT_UNIQUE_MAX": 1,
    # Multiplier for the Interquartile Range (IQR) to detect outliers.
    "OUTLIER_IQR_SCALE": 1.5,
    # Absolute value of skewness considered "high".
    "HIGH_SKEWNESS_ABS": 2.0,
    # Minimum and maximum percentage of numeric values expected in a mixed-type column
    # for it to be flagged as potentially needing a type cast.
    "MIXED_TYPE_NUM_PCT_MIN": 0.60,
    "MIXED_TYPE_NUM_PCT_MAX": 0.99,
    # Percentage of unique values above which cardinality is considered "high",
    # suggesting it might be an ID or name column.
    "HIGH_CARDINALITY_PCT": 80.0,
}

def _fmt_pct(p: float) -> str:
    """One-decimal percentage; nonzero shares that would round to 0.0 show as '<0.1'."""
    if p > 0 and round(p, 1) == 0:
        return "<0.1"
    return f"{p:.1f}"


SAMPLE_NOTE = " (based on a 10,000-row sample)"


def _check_null_values(df: pd.DataFrame, col: str) -> list[dict]:
    """Identify and flag columns with a significant percentage of null values."""
    proposals = []
    null_count = df[col].isnull().sum()
    if null_count > 0:
        null_pct = (null_count / len(df)) * 100
        if null_pct > SCOUT_THRESHOLDS["NULL_DROP_PCT"]:
            proposals.append({
                "type": "Redundant Column", "column": col, "reason": f"{_fmt_pct(null_pct)}% empty (Near-complete nullity)",
                "rule_data": {"action": "drop_column", "column": col}
            })
        else:
            proposals.append({
                "type": "Null Check", "column": col, "reason": f"{_fmt_pct(null_pct)}% missing values",
                "rule_data": {"type": "Null Check", "col": col, "desc": f"{col} is NOT NULL"}
            })
    return proposals

def _check_constant_column(df: pd.DataFrame, col: str) -> list[dict]:
    """Identify columns with no variance (all values are the same)."""
    proposals = []
    # A single row is trivially "constant"; and NaN counts as a distinct value, so a column of
    # 'Y' and blanks is not zero-variance.
    if len(df) < 2:
        return proposals
    if df[col].nunique(dropna=False) <= SCOUT_THRESHOLDS["CONSTANT_UNIQUE_MAX"]:
        proposals.append({
            "type": "Constant Value", "column": col, "reason": "Zero variance (all values are identical)",
            "rule_data": {"action": "drop_column", "column": col}
        })
    return proposals

_THOUSANDS_RE = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_PLAIN_NUM_RE = re.compile(r"-?\d+(\.\d+)?")


def _mostly_datetime(as_text: pd.Series) -> bool:
    """>=95% of values parse as datetimes (mixed formats). Needs digits so month names/words don't count."""
    cand = as_text[as_text.str.contains(r"\d", regex=True) & (as_text.str.len() >= 6)]
    if len(cand) < 0.95 * len(as_text):
        return False

    def frac(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(x, errors="coerce", format="mixed").notna().sum()

    head = as_text.head(50)
    if frac(head) < 0.9 * len(head):  # cheap early exit before parsing the whole sample
        return False
    return frac(as_text) >= 0.95 * len(as_text)


_NUM_ID_NAME_HINT = re.compile(r"(^|[^a-z])(id|ids|zip|zipcode|postal|postcode|code|fips|uuid|sku)($|[^a-z])")


def _numeric_looks_like_id(series: pd.Series, col, is_int: bool) -> bool:
    """Numeric ID-like column: name hints at an identifier, or all values are unique integers."""
    if _NUM_ID_NAME_HINT.search(str(col).lower()):
        return True
    vals = series.dropna()
    if not (is_int and len(vals) > 1 and vals.nunique() == len(vals)):
        return False
    # Unique AND dense (like a row counter 1..n): a salary or population column is unique too,
    # but spread out, and its outliers still deserve a Range Check.
    span = int(vals.max()) - int(vals.min()) + 1
    return span <= 2 * len(vals)


def _check_numeric_diagnostics(df: pd.DataFrame, col: str) -> list[dict]:
    """Analyze numeric columns for statistical properties like outliers and skewness."""
    proposals = []
    # bool (incl. nullable "boolean") counts as numeric to pandas but IQR/skew are meaningless
    # and numpy refuses boolean subtraction.
    if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
        # Outlier Detection (IQR)
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - SCOUT_THRESHOLDS["OUTLIER_IQR_SCALE"] * IQR
        upper_bound = Q3 + SCOUT_THRESHOLDS["OUTLIER_IQR_SCALE"] * IQR
        
        col_vals = df[col].dropna()
        is_int = pd.api.types.is_integer_dtype(df[col])  # float columns (e.g. ints with NaN) are left as floats
        if is_int:
            # Integer data: bounds are whole numbers (rounded outward) and never below 0 when all values are >= 0.
            if col_vals.min() >= 0:
                lower_bound = max(lower_bound, 0)
            lower_bound, upper_bound = math.floor(lower_bound), math.ceil(upper_bound)

        # Use a mask for efficiency and clarity
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers = outlier_mask.sum()

        # IQR == 0 (binary flags, zero-inflated counts) collapses the bounds to one value and
        # would flag every other value as an outlier; Drop/Cap would then destroy real data.
        # ID-like columns (unique integers, id/zip/code names) have no meaningful outliers.
        if (outliers > 0 and IQR > 0 and df[col].nunique() > 2
                and not _numeric_looks_like_id(df[col], col, is_int)):
            proposals.append({
                "type": "Range Check", "column": col, "reason": f"{outliers} statistical outliers detected",
                "rule_data": {"type": "Range Check", "col": col, "min": float(lower_bound), "max": float(upper_bound), "desc": f"{col} within statistical bounds"}
            })

        # Skewness Warning
        skew = df[col].skew()
        if pd.notna(skew) and abs(skew) > SCOUT_THRESHOLDS["HIGH_SKEWNESS_ABS"]:
            proposals.append({
                "type": "Distribution Warning", "column": col, "reason": f"High skewness ({skew:.2f}) detected",
                "rule_data": {
                    "type": "Informational",
                    "desc": f"Distribution Warning for '{col}': skewness is {skew:.2f}"
                }
            })
    return proposals

_ID_NAME_HINT = re.compile(r"(^|[^a-z])(id|ids|zip|zipcode|postal|postcode|code|phone|ssn|sku|uuid)($|[^a-z])|(id|zip|code)$")


def _looks_like_identifier(col, numeric_strs: pd.Series) -> bool:
    """True for ID-like text columns where a float cast would destroy data (ZIPs, fixed-width codes)."""
    if _ID_NAME_HINT.search(str(col).lower()):
        return True
    stripped = numeric_strs.astype(str).str.strip()
    if stripped.str.match(r"^[+-]?0\d").any():  # leading zeros are significant
        return True
    digits = stripped[stripped.str.fullmatch(r"\d+")]
    # all-digit values of one fixed length (>= 5 digits) are codes, not quantities
    return len(digits) == len(stripped) and stripped.str.len().nunique() == 1 and stripped.str.len().iloc[0] >= 5


def _check_string_diagnostics(df: pd.DataFrame, col: str) -> list[dict]:
    """Analyze text-based columns for mixed types, high cardinality, and whitespace."""
    proposals = []
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        non_nulls = df[col].dropna()
        if len(non_nulls) > 0:
            # Mixed Type Detection
            parsed = pd.to_numeric(non_nulls, errors='coerce')
            num_pct = parsed.notnull().mean()
            as_text = non_nulls.astype(str)
            if (num_pct < 1 and as_text.str.contains(",", regex=False).any()
                    and as_text.str.fullmatch(_THOUSANDS_RE.pattern + "|" + _PLAIN_NUM_RE.pattern).all()
                    and not _looks_like_identifier(col, as_text.str.replace(",", "", regex=False))):
                proposals.append({
                    "type": "Type Cast", "column": col,
                    "reason": "Numbers stored as text with thousands separators (e.g. 25,649); every value can be cast to numeric",
                    "rule_data": {"action": "cast_type", "column": col, "dtype": "float64"}
                })
            elif (SCOUT_THRESHOLDS["MIXED_TYPE_NUM_PCT_MIN"] <= num_pct < SCOUT_THRESHOLDS["MIXED_TYPE_NUM_PCT_MAX"]
                    and not _looks_like_identifier(col, non_nulls[parsed.notnull()])):
                n_lost = int(parsed.isnull().sum())
                proposals.append({
                    "type": "Type Cast", "column": col,
                    "reason": (f"Mixed types ({num_pct:.1%} numeric values hidden in text); "
                               f"casting would turn {n_lost} non-numeric cell{'s' if n_lost != 1 else ''} into NaN"),
                    "rule_data": {"action": "cast_type", "column": col, "dtype": "float64"}
                })
            
            elif (num_pct < 0.5 and _mostly_datetime(as_text)):
                proposals.append({
                    "type": "Type Cast", "column": col,
                    "reason": "Text values parse as dates/times (>=95%); cast to datetime",
                    "rule_data": {"action": "cast_type", "column": col, "dtype": "datetime64[ns]"}
                })

            # High Cardinality Detection
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct > SCOUT_THRESHOLDS["HIGH_CARDINALITY_PCT"]:
                proposals.append({
                    "type": "High Cardinality", "column": col, "reason": f"{df[col].nunique()} unique values ({unique_pct:.1f}% unique). Likely an ID or Name.",
                    "rule_data": {
                        "type": "Informational",
                        "desc": f"High Cardinality warning for '{col}': {df[col].nunique()} unique values"
                    }
                })

            # Whitespace Detection
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                txt = df[col].dropna().astype(str)
                has_whitespace = bool((txt != txt.str.strip()).any())  # linear; a regex scan is quadratic
                if has_whitespace:
                    proposals.append({
                        "type": "Formatting Issue", "column": col, "reason": "Leading or trailing whitespace detected",
                        "rule_data": {"action": "strip_whitespace", "column": col}
                    })
    return proposals

DATASET_KEY = "__dataset__"


def _check_dataset_level(df: pd.DataFrame) -> list[dict]:
    """Dataset-wide issues: exact duplicate rows, all-empty rows, non-normalized headers.

    Proposals use the reserved column "__dataset__" so dismissal keys look like
    "__dataset__:duplicates". Run on the full frame (cheap vectorised checks).
    """
    from engine_ops import predict_column_renames, _normalize_one_name
    proposals = []

    dup = int(df.duplicated().sum())
    if dup > 0:
        proposals.append({
            "type": "duplicates", "column": DATASET_KEY, "title": "Duplicate Rows",
            "reason": f"{dup} exact duplicate row{'s' if dup != 1 else ''} detected",
            "rule_data": {"action": "drop_duplicates"},
        })

    empty = int(df.isnull().all(axis=1).sum())
    if empty > 0:
        proposals.append({
            "type": "empty_rows", "column": DATASET_KEY, "title": "Empty Rows",
            "reason": f"{empty} entirely empty row{'s' if empty != 1 else ''} detected",
            "rule_data": {"action": "drop_empty_rows"},
        })

    cols = [str(c) for c in df.columns]
    dup_names = sorted({c for c in cols if cols.count(c) > 1})
    if dup_names:
        preview = ", ".join(f"'{c}'" for c in dup_names[:3]) + ("..." if len(dup_names) > 3 else "")
        proposals.append({
            "type": "duplicate_columns", "column": DATASET_KEY, "title": "Duplicate Column Names",
            "reason": (f"{len(dup_names)} column name{'s' if len(dup_names) != 1 else ''} used more than once ({preview}); "
                       "per-column checks skip them. Normalizing names renames repeats with _2, _3 suffixes"),
            "rule_data": {"action": "normalize_column_names", "value": "snake_case"},
        })
    renames = predict_column_renames(cols, 'snake_case', only_changed=True)
    # Names that would collide after normalization (before the _2/_3 de-duplication suffixes)
    base = [_normalize_one_name(c, 'snake_case') for c in cols]
    collisions = {n for n in base if base.count(n) > 1}
    if renames:
        preview = ", ".join(f"'{c}'" for c in list(renames)[:3]) + ("..." if len(renames) > 3 else "")
        reason = f"{len(renames)} column name{'s' if len(renames) != 1 else ''} not snake_case ({preview})"
        if collisions:
            reason += f"; {len(collisions)} name(s) would collide after normalization"
        proposals.append({
            "type": "column_names", "column": DATASET_KEY, "title": "Column Names",
            "reason": reason,
            "rule_data": {"action": "normalize_column_names", "value": "snake_case"},
        })
    return proposals


def generate_proposals(df: pd.DataFrame, scanned_columns: set) -> list[dict]:
    """
    Automatically detects potential data quality issues in a DataFrame and proposes cleaning rules or transformations.

    This function iterates through each column of the input DataFrame, skipping columns that have
    already been processed (`scanned_columns`). It applies various heuristics to identify issues
    such as high null percentages, constant values, numerical outliers, high skewness, mixed
    data types, and high cardinality. For each detected issue, it generates a proposal
    (a dictionary) that describes the issue and suggests a corresponding cleaning action or rule.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze for data quality issues.
        scanned_columns (set): A set of column names that have already been scanned or processed,
                                to avoid re-generating proposals for them.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a detected issue
                    and a proposed rule or action to address it. Each proposal includes:
                    - "type" (str): The category of the detected issue (e.g., "Redundant Column", "Null Check", "Range Check").
                    - "column" (str): The name of the column affected.
                    - "reason" (str): A description of why the proposal was generated.
                    - "rule_data" (dict): A dictionary containing the details of the proposed
                                          cleaning rule or action.
    """
    proposals = []
    
    if df.empty:
        return proposals
        
    # Dataset-level checks run on the full frame before sampling.
    proposals.extend(_check_dataset_level(df))

    # Sample down to 10,000 rows if dataset is larger, to ensure scouts run fast
    full_df = df
    sampled = len(df) > 10000
    if sampled:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)

    repeated = set(df.columns[df.columns.duplicated(keep=False)])
    for col in df.columns:
        # Duplicate names make df[col] a DataFrame; they are reported at dataset level instead.
        if col in repeated:
            continue
        # Skip columns that have already been completely processed (backward compatibility).
        if col in scanned_columns: 
            continue
        
        null_proposals = _check_null_values(df, col)
        col_sample = list(null_proposals)
        const_props = []

        # If a column was already flagged as Redundant (near-complete nullity → drop),
        # skip _check_constant_column to prevent a duplicate drop_column proposal.
        # A 100% null column trivially has nunique() == 0 which also triggers Constant Value.
        already_redundant = any(p['type'] == 'Redundant Column' for p in null_proposals)
        if not already_redundant:
            # Constant-ness is cheap and must be exact: one differing value outside the sample counts.
            const_props = _check_constant_column(full_df, col)

        col_sample.extend(_check_numeric_diagnostics(df, col))
        col_sample.extend(_check_string_diagnostics(df, col))
        if sampled:
            for p in col_sample:
                p["reason"] += SAMPLE_NOTE
        proposals.extend(col_sample[:len(null_proposals)])
        proposals.extend(const_props)
        proposals.extend(col_sample[len(null_proposals):])
                    
    # Filter out individual proposals that have been accepted or dismissed.
    # The format in scanned_columns is "column:proposal_type".
    filtered_proposals = []
    for p in proposals:
        key = f"{p['column']}:{p['type']}"
        if key not in scanned_columns:
            filtered_proposals.append(p)
            
    return filtered_proposals
