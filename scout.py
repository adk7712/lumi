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

def _check_null_values(df: pd.DataFrame, col: str) -> list[dict]:
    """Identify and flag columns with a significant percentage of null values."""
    proposals = []
    null_count = df[col].isnull().sum()
    if null_count > 0:
        null_pct = (null_count / len(df)) * 100
        if null_pct > SCOUT_THRESHOLDS["NULL_DROP_PCT"]:
            proposals.append({
                "type": "Redundant Column", "column": col, "reason": f"{null_pct:.1f}% empty (Near-complete nullity)",
                "rule_data": {"action": "drop_column", "column": col}
            })
        else:
            proposals.append({
                "type": "Null Check", "column": col, "reason": f"{null_pct:.1f}% missing values",
                "rule_data": {"type": "Null Check", "col": col, "desc": f"{col} is NOT NULL"}
            })
    return proposals

def _check_constant_column(df: pd.DataFrame, col: str) -> list[dict]:
    """Identify columns with no variance (all values are the same)."""
    proposals = []
    if df[col].nunique() <= SCOUT_THRESHOLDS["CONSTANT_UNIQUE_MAX"]:
        proposals.append({
            "type": "Constant Value", "column": col, "reason": "Zero variance (all values are identical)",
            "rule_data": {"action": "drop_column", "column": col}
        })
    return proposals

def _check_numeric_diagnostics(df: pd.DataFrame, col: str) -> list[dict]:
    """Analyze numeric columns for statistical properties like outliers and skewness."""
    proposals = []
    if pd.api.types.is_numeric_dtype(df[col]):
        # Outlier Detection (IQR)
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        l, u = Q1 - SCOUT_THRESHOLDS["OUTLIER_IQR_SCALE"] * IQR, Q3 + SCOUT_THRESHOLDS["OUTLIER_IQR_SCALE"] * IQR
        
        # Use a mask for efficiency and clarity
        outlier_mask = (df[col] < l) | (df[col] > u)
        outliers = outlier_mask.sum()
        
        if outliers > 0:
            proposals.append({
                "type": "Range Check", "column": col, "reason": f"{outliers} statistical outliers detected",
                "rule_data": {"type": "Range Check", "col": col, "min": float(l), "max": float(u), "desc": f"{col} within statistical bounds"}
            })
        
        # Skewness Warning
        skew = df[col].skew()
        if abs(skew) > SCOUT_THRESHOLDS["HIGH_SKEWNESS_ABS"]:
            proposals.append({
                "type": "Distribution Warning", "column": col, "reason": f"High skewness ({skew:.2f}) detected",
                "rule_data": {
                    "type": "Informational",
                    "desc": f"Distribution Warning for '{col}': skewness is {skew:.2f}"
                }
            })
    return proposals

def _check_string_diagnostics(df: pd.DataFrame, col: str) -> list[dict]:
    """Analyze text-based columns for mixed types, high cardinality, and whitespace."""
    proposals = []
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        non_nulls = df[col].dropna()
        if len(non_nulls) > 0:
            # Mixed Type Detection
            num_pct = pd.to_numeric(non_nulls, errors='coerce').notnull().mean()
            if SCOUT_THRESHOLDS["MIXED_TYPE_NUM_PCT_MIN"] <= num_pct < SCOUT_THRESHOLDS["MIXED_TYPE_NUM_PCT_MAX"]:
                proposals.append({
                    "type": "Type Cast", "column": col, "reason": f"Mixed types ({num_pct:.1%} numeric values hidden in text)",
                    "rule_data": {"action": "cast_type", "column": col, "dtype": "float64"}
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
            if df[col].dtype == 'object':
                has_whitespace = df[col].dropna().astype(str).str.contains(r"^\s+|\s+$").any()
                if has_whitespace:
                    proposals.append({
                        "type": "Formatting Issue", "column": col, "reason": "Leading or trailing whitespace detected",
                        "rule_data": {"action": "strip_whitespace", "column": col}
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

    for col in df.columns:
        # Skip columns that have already been processed or whose proposals have been handled.
        if col in scanned_columns: 
            continue
        
        proposals.extend(_check_null_values(df, col))
        proposals.extend(_check_constant_column(df, col))
        proposals.extend(_check_numeric_diagnostics(df, col))
        proposals.extend(_check_string_diagnostics(df, col))
                    
    return proposals
