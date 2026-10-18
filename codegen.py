import json
import math
import numbers

KNN_DEFAULT_NEIGHBORS = 5
FUZZY_MATCH_THRESHOLD = 85
FUZZY_MAX_UNIQUE = 1000
KNN_MAX_FEATURES = 10

_RELATIONAL_OPS = (">", "<", "==", "!=", ">=", "<=")
_DATETIME_ACCESSORS = {
    "year": "dt.year",
    "month": "dt.month",
    "day": "dt.day",
    "day_of_week": "dt.day_name()",
    "hour": "dt.hour",
}
_TEXT_METHODS = {"lowercase": "lower", "uppercase": "upper", "titlecase": "title"}

# Shared header for the exported script / notebook.  The helper mirrors
# state_manager.downcast_dtypes, which the live app applies to every uploaded
# file, so exported results match the in-app preview.
_IMPORTS = "import warnings\nimport pandas as pd\nimport numpy as np\n"
_DOWNCAST_HELPER = (
    "def _downcast_dtypes(df):\n"
    "    \"\"\"Downcast numeric dtypes exactly like Lumi does on upload (e.g. int8/float32).\"\"\"\n"
    "    if df.empty:\n"
    "        return df\n"
    "    df = df.copy()\n"
    "    for col in df.columns:\n"
    "        if pd.api.types.is_float_dtype(df[col]):\n"
    "            df[col] = pd.to_numeric(df[col], downcast='float')\n"
    "        elif pd.api.types.is_integer_dtype(df[col]):\n"
    "            df[col] = pd.to_numeric(df[col], downcast='integer')\n"
    "    return df\n"
    "\n\n"
    "def _cap_series(s, lo, hi):\n"
    "    \"\"\"Clip to [lo, hi]; integer columns stay integer when both bounds are integral.\"\"\"\n"
    "    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):\n"
    "        if all(b is None or float(b) == int(b) for b in (lo, hi)):\n"
    "            info = np.iinfo(s.dtype.numpy_dtype if hasattr(s.dtype, 'numpy_dtype') else s.dtype)\n"
    "            lo = None if lo is None else max(int(lo), int(info.min))\n"
    "            hi = None if hi is None else min(int(hi), int(info.max))\n"
    "            return s.clip(lower=lo, upper=hi)\n"
    "        s = s.astype('float64')\n"
    "    return s.clip(lower=lo, upper=hi)\n"
    "\n\n"
    "def _numeric_to_datetime(s, col):\n"
    "    \"\"\"Numeric -> datetime as years (1000-2999) or YYYYMMDD; else warn and return None.\"\"\"\n"
    "    nn = s[s.notna()]\n"
    "    if len(nn) == 0:\n"
    "        return pd.to_datetime(s, errors='coerce')\n"
    "    f = nn.astype('float64')\n"
    "    if not (f % 1 == 0).all():\n"
    "        warnings.warn(f\"'{col}' has non-integer numeric values; not converted to datetime. Column left unchanged.\")\n"
    "        return None\n"
    "    if f.between(1000, 2999).all():\n"
    "        fmt = '%Y'\n"
    "    elif f.between(10000101, 29991231).all():\n"
    "        fmt = '%Y%m%d'\n"
    "    else:\n"
    "        warnings.warn(f\"'{col}' is numeric but not years (1000-2999) or YYYYMMDD dates; not converted to datetime. Column left unchanged.\")\n"
    "        return None\n"
    "    strs = s.map(lambda v: None if pd.isna(v) else str(int(v)))\n"
    "    return pd.to_datetime(strs, format=fmt, errors='coerce')\n"
)


def _lit(v) -> str:
    """Python literal for a scalar bound/value; keeps int vs float like the engine."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return repr(v)
    if isinstance(v, numbers.Integral):
        return repr(int(v))
    if isinstance(v, numbers.Real):
        f = float(v)
        if math.isnan(f):
            return "float('nan')"
        if math.isinf(f):
            return "float('inf')" if f > 0 else "float('-inf')"
        return repr(f)
    return repr(v)


def _comment(text) -> str:
    """Make arbitrary user text safe for a single-line # comment."""
    return " ".join(str(text).splitlines())


def _valid_bound(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return False
    return True


def _codegen_drop_column(step: dict) -> list[str]:
    col = step.get('column')
    return [
        f"    # Remove the specified column from the DataFrame.",
        f"    df = df.drop(columns=[{repr(col)}])"
    ]

def _codegen_drop_nulls(step: dict) -> list[str]:
    col = step.get('column')
    return [
        f"    # Remove rows where the specified column has null values.",
        f"    df = df.dropna(subset=[{repr(col)}])"
    ]

def _codegen_fill_null(step: dict) -> list[str]:
    col = step.get('column')
    v = step.get('value')
    c_repr = repr(col)
    code = []
    if v in ["mean", "median"]:
        code.append(f"    # Fill nulls with the column {v} (non-numeric entries are ignored when computing it).")
        code.append(f"    _num = pd.to_numeric(df[{c_repr}], errors='coerce')")
        code.append(f"    if _num.notnull().any():")
        code.append(f"        df[{c_repr}] = df[{c_repr}].fillna(_num.{v}())")
        code.append(f"    else:")
        code.append(f"        warnings.warn({repr(f'Cannot calculate {v} for non-numeric column {col!r}.')})")
    elif v == "mode":
        code.append(f"    # Fill nulls with the most frequent value (skipped if the column has no mode).")
        code.append(f"    _mode = df[{c_repr}].mode()")
        code.append(f"    if not _mode.empty:")
        code.append(f"        df[{c_repr}] = df[{c_repr}].fillna(_mode[0])")
    elif v in ["knn", "iterative"]:
        code.append(f"    # Advanced imputation using scikit-learn on the target plus its {KNN_MAX_FEATURES} most-correlated numeric columns.")
        code.append(f"    try:")
        if v == "knn":
            code.append(f"        from sklearn.impute import KNNImputer")
            code.append(f"        imputer = KNNImputer(n_neighbors={KNN_DEFAULT_NEIGHBORS})")
        else:
            code.append(f"        from sklearn.experimental import enable_iterative_imputer")
            code.append(f"        from sklearn.impute import IterativeImputer")
            code.append(f"        imputer = IterativeImputer(random_state=42)")
        code.append(f"        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()")
        code.append(f"        if {c_repr} not in numeric_cols:")
        code.append(f"            raise TypeError(f\"{v.upper()} imputation requires a numeric column. {{{c_repr}!r}} is {{df[{c_repr}].dtype}}.\")")
        code.append(f"        _others = [_c for _c in numeric_cols if _c != {c_repr}]")
        code.append(f"        if len(_others) > {KNN_MAX_FEATURES}:")
        code.append(f"            _corrs = df[numeric_cols].corrwith(df[{c_repr}]).abs().fillna(0)")
        code.append(f"            _selected = [{c_repr}] + sorted(_others, key=lambda _c: _corrs.get(_c, 0), reverse=True)[:{KNN_MAX_FEATURES}]")
        code.append(f"        else:")
        code.append(f"            _selected = numeric_cols")
        code.append(f"        _imputed = pd.DataFrame(imputer.fit_transform(df[_selected].copy()), columns=_selected, index=df.index)")
        code.append(f"        df[{c_repr}] = _imputed[{c_repr}]")
        code.append(f"    except Exception as e:")
        code.append(f"        warnings.warn(f\"{v.upper()} imputation on {{{c_repr}!r}} skipped: {{e}}\")")
    else:
        code.append(f"    # Fill null values with a custom constant.")
        code.append(f"    df[{c_repr}] = df[{c_repr}].fillna({_lit(v)})")
    return code

def _codegen_cap_range(step: dict) -> list[str]:
    col = step.get('column')
    c_repr = repr(col)
    lo, hi = _lit(step['min']), _lit(step['max'])
    return [
        f"    # Cap values in the specified column to be within the defined min and max bounds.",
        f"    df[{c_repr}] = _cap_series(df[{c_repr}], {lo}, {hi})"
    ]

def _codegen_cast_type(step: dict) -> list[str]:
    col = step.get('column')
    c_repr = repr(col)
    if step['dtype'] == "datetime64[ns]":
        return [
            f"    # Convert column to datetime, coercing errors to NaT.",
            f"    # Numeric columns are read as years / YYYYMMDD (never as nanoseconds); otherwise left unchanged with a warning.",
            f"    if pd.api.types.is_numeric_dtype(df[{c_repr}]) and not pd.api.types.is_bool_dtype(df[{c_repr}]):",
            f"        _dt = _numeric_to_datetime(df[{c_repr}], {c_repr})",
            f"        if _dt is not None:",
            f"            df[{c_repr}] = _dt",
            f"    else:",
            f"        df[{c_repr}] = pd.to_datetime(df[{c_repr}], errors='coerce')"
        ]
    target_dtype = step['dtype']
    if target_dtype in ['int64', 'int32', 'int']:
        target_dtype = "Int64"
    if target_dtype in ['string', 'object']:
        return [
            f"    # Convert column to target string/object type.",
            f"    df[{c_repr}] = df[{c_repr}].astype({repr(target_dtype)})"
        ]
    return [
        f"    # Convert column to target numeric type, coercing errors to NaN.",
        f"    df[{c_repr}] = pd.to_numeric(df[{c_repr}], errors='coerce').astype({repr(target_dtype)})"
    ]


def _rule_mask_lines(r: dict, indent: str, var: str = "mask") -> list[str] | None:
    """Lines assigning a boolean *violation* mask to `var`, mirroring rule_utils.evaluate_rule.

    Returns None for rules that can never be violated (Informational / unknown types),
    and a constant all-False mask for Range Checks with unusable bounds.
    """
    t = r.get('type')
    if t == "Null Check":
        return [f"{indent}{var} = df[{repr(r['col'])}].isnull()"]
    if t == "Range Check":
        lo, hi = r.get('min'), r.get('max')
        if not (_valid_bound(lo) and _valid_bound(hi)):
            return [f"{indent}{var} = pd.Series(False, index=df.index)  # invalid bounds: rule cannot be evaluated"]
        c = repr(r['col'])
        return [
            f"{indent}# Nulls are not range violations (that is the Null Check's job); unparseable values are.",
            f"{indent}_col_num = pd.to_numeric(df[{c}], errors='coerce')",
            f"{indent}{var} = (_col_num < {_lit(lo)}) | (_col_num > {_lit(hi)}) | (_col_num.isnull() & df[{c}].notnull())",
        ]
    if t == "Relational Check":
        a = f"df[{repr(r['col_a'])}]"
        b = f"df[{repr(r['col_b'])}]" if r.get('target_type') == 'Feature' else _lit(r.get('value'))
        op = r.get('op')
        if op not in _RELATIONAL_OPS:
            return [f"{indent}{var} = pd.Series(True, index=df.index)  # unsupported operator {op!r}"]
        return [
            f"{indent}try:",
            f"{indent}    _valid = ({a} {op} {b})",
            f"{indent}except TypeError:",
            f"{indent}    _valid = pd.Series(False, index=df.index)  # incomparable types: every row violates",
            f"{indent}{var} = ~_valid",
        ]
    if t == "Custom Expression":
        # Query describes the VIOLATION condition — rows that match are flagged.
        return [
            f"{indent}_violated_idx = df.query({repr(r['query'])}).index",
            f"{indent}{var} = df.index.isin(_violated_idx)",
        ]
    return None


def _codegen_drop_violated(step: dict) -> list[str]:
    r = step['rule']
    lines = _rule_mask_lines(r, "    ", "_mask")
    if lines is None:
        return []
    return [f"    # Remove rows violating rule: {_comment(r.get('desc', 'N/A'))}"] + lines + ["    df = df[~_mask]"]

def _codegen_replace(step: dict) -> list[str]:
    col = step.get('column')
    f_repr, r_repr = repr(step['find']), repr(step['replace'])
    regex_param = f", regex={bool(step.get('regex', False))}"
    if col == "All":
        return [
            f"    # Replace in all object/string columns, keeping each column's dtype.",
            f"    for c in df.select_dtypes(include=['object', 'string']).columns:",
            f"        df[c] = df[c].replace({f_repr}, {r_repr}{regex_param}).astype(df[c].dtype)"
        ]
    c_repr = repr(col)
    return [
        f"    # Replace in the specified column, keeping its dtype.",
        f"    df[{c_repr}] = df[{c_repr}].replace({f_repr}, {r_repr}{regex_param}).astype(df[{c_repr}].dtype)"
    ]

def _codegen_strip_whitespace(step: dict) -> list[str]:
    col = step.get('column')
    if col == "All":
        return [
            f"    # Strip whitespace from all object/string columns, preserving NaNs.",
            f"    for c in df.select_dtypes(include=['object', 'string']).columns:",
            f"        mask = df[c].notnull()",
            f"        df.loc[mask, c] = df.loc[mask, c].astype(str).str.strip().astype(df[c].dtype)"
        ]
    c_repr = repr(col)
    return [
        f"    # Strip whitespace from the specified column, preserving NaNs.",
        f"    mask = df[{c_repr}].notnull()",
        f"    df.loc[mask, {c_repr}] = df.loc[mask, {c_repr}].astype(str).str.strip().astype(df[{c_repr}].dtype)"
    ]

def _normalize_text_body(method: str, target: str, indent: str, label: str) -> list[str]:
    """Lines normalizing `target` in place (nulls preserved, dtype restored)."""
    if method == "fuzzy_dedupe":
        return [
            f"{indent}_s = {target}",
            f"{indent}unique_vals = _s.dropna().unique()",
            f"{indent}if len(unique_vals) > {FUZZY_MAX_UNIQUE}:",
            f"{indent}    warnings.warn(f\"Fuzzy deduplication skipped on {{{label}!r}}: too many unique values ({{len(unique_vals)}} > {FUZZY_MAX_UNIQUE}).\")",
            f"{indent}else:",
            f"{indent}    from thefuzz import process",
            f"{indent}    mapping = {{}}",
            f"{indent}    handled = set()",
            f"{indent}    for v in unique_vals:",
            f"{indent}        if v in handled: continue",
            f"{indent}        for match, score in process.extract(v, unique_vals, limit=10):",
            f"{indent}            if score > {FUZZY_MATCH_THRESHOLD}: mapping[match] = v; handled.add(match)",
            f"{indent}    {target} = _s.replace(mapping).astype(_s.dtype)",
        ]
    if method == "remove_punctuation":
        op = "str.replace(f'[{string.punctuation}]', '', regex=True)"
    else:
        op = f"str.{_TEXT_METHODS[method]}()"
    return [
        f"{indent}_s = {target}",
        f"{indent}{target} = _s.astype(str).{op}.where(_s.notnull(), _s).astype(_s.dtype)",
    ]

def _codegen_normalize_text(step: dict) -> list[str]:
    col = step.get('column')
    method = step.get('value', 'lowercase')
    if method not in _TEXT_METHODS and method not in ("remove_punctuation", "fuzzy_dedupe"):
        return [f"    # Unknown text normalization method {method!r}: no change."]
    code = [f"    # Normalize text using {method} method (nulls are preserved)."]
    if method == "remove_punctuation":
        code.append("    import string")
    if col == "All":
        code.append("    for c in df.select_dtypes(include=['object', 'string']).columns:")
        code.extend(_normalize_text_body(method, "df[c]", "        ", "c"))
    else:
        c_repr = repr(col)
        code.append(f"    if pd.api.types.is_object_dtype(df[{c_repr}]) or pd.api.types.is_string_dtype(df[{c_repr}]):")
        code.extend(_normalize_text_body(method, f"df[{c_repr}]", "        ", c_repr))
        code.append("    else:")
        code.append(f"        warnings.warn(f\"normalize_text skipped on non-text column {{{c_repr}!r}} ({{df[{c_repr}].dtype}}).\")")
    return code

def _codegen_log_transform(step: dict) -> list[str]:
    col = step.get('column')
    c_repr = repr(col)
    return [
        f"    # Apply log(1+x) transformation to handle outliers (numeric columns only).",
        f"    if pd.api.types.is_numeric_dtype(df[{c_repr}]):",
        f"        df[{c_repr}] = np.log1p(df[{c_repr}].clip(lower=0))",
        f"    else:",
        f"        warnings.warn(f\"Log transformation requires a numeric column. {{{c_repr}!r}} is {{df[{c_repr}].dtype}}.\")"
    ]

def _codegen_rename_column(step: dict) -> list[str]:
    col = step.get('column')
    new_name = step['value']
    return [
        f"    # Rename column {col!r} to {new_name!r}.",
        f"    df = df.rename(columns={{{repr(col)}: {repr(new_name)}}})"
    ]

def _codegen_reorder_columns(step: dict) -> list[str]:
    new_order = step.get('value', [])
    if not isinstance(new_order, list):
        return ["    # Reorder columns skipped: expected a list of column names."]
    return [
        f"    # Reorder columns to specified layout, preserving any remaining columns.",
        f"    _cols = [c for c in {repr(new_order)} if c in df.columns]",
        f"    _rem = [c for c in df.columns if c not in _cols]",
        f"    df = df[_cols + _rem]"
    ]

def _codegen_extract_datetime(step: dict) -> list[str]:
    col = step.get('column')
    component = step.get('component')
    new_col = step.get('new_column')
    if not new_col:
        return [f"    # extract_datetime on {col!r} skipped: no target column name specified."]
    accessor = _DATETIME_ACCESSORS.get(component)
    if accessor is None:
        return [f"    # extract_datetime on {col!r} skipped: unknown datetime component {component!r}."]
    return [
        f"    # Extract {component} component from {col!r} into {new_col!r}.",
        f"    if pd.api.types.is_numeric_dtype(df[{repr(col)}]) and not pd.api.types.is_bool_dtype(df[{repr(col)}]):",
        f"        _dt = _numeric_to_datetime(df[{repr(col)}], {repr(col)})",
        f"    else:",
        f"        _dt = pd.to_datetime(df[{repr(col)}], errors='coerce')",
        f"    if _dt is not None:",
        f"        df[{repr(new_col)}] = _dt.{accessor}"
    ]

def _codegen_drop_duplicates(step: dict) -> list[str]:
    return [
        "    # Remove duplicate rows from the DataFrame.",
        "    df = df.drop_duplicates()"
    ]

def _codegen_drop_empty_columns(step: dict) -> list[str]:
    return [
        "    # Remove columns that are completely empty (all nulls).",
        "    empty_cols = [c for c in df.columns if df[c].isnull().all()]",
        "    df = df.drop(columns=empty_cols)"
    ]

def _codegen_drop_empty_rows(step: dict) -> list[str]:
    return [
        "    # Remove rows that are completely empty (all nulls).",
        "    df = df.dropna(how='all')"
    ]

def _codegen_normalize_column_names(step: dict) -> list[str]:
    method = step.get('value', 'snake_case')
    exprs = {
        'snake_case': "re.sub(r'_+', '_', re.sub(r'[^a-zA-Z0-9_]', '', _orig.strip().replace(' ', '_').replace('-', '_'))).lower()",
        'lowercase': "_orig.lower()",
        'uppercase': "_orig.upper()",
        'remove_spaces': "_orig.replace(' ', '')",
    }
    if method not in exprs:
        return [f"    # Unknown column-name normalization method {method!r}: no change."]
    code = [f"    # Normalize column names using {method} format."]
    if method == 'snake_case':
        code.append("    import re")
    code.extend([
        "    _new_cols = []",
        "    for _orig in df.columns:",
        f"        _val = {exprs[method]}",
        "        _new_cols.append(_val if _val else f\"column_{_orig}\")",
        "    df.columns = _new_cols",
    ])
    return code

_CODEGEN_REGISTRY = {
    "drop_column": _codegen_drop_column,
    "drop_nulls": _codegen_drop_nulls,
    "fill_null": _codegen_fill_null,
    "cap_range": _codegen_cap_range,
    "cast_type": _codegen_cast_type,
    "drop_violated": _codegen_drop_violated,
    "replace": _codegen_replace,
    "strip_whitespace": _codegen_strip_whitespace,
    "normalize_text": _codegen_normalize_text,
    "log_transform": _codegen_log_transform,
    "rename_column": _codegen_rename_column,
    "reorder_columns": _codegen_reorder_columns,
    "extract_datetime": _codegen_extract_datetime,
    "drop_duplicates": _codegen_drop_duplicates,
    "drop_empty_columns": _codegen_drop_empty_columns,
    "drop_empty_rows": _codegen_drop_empty_rows,
    "normalize_column_names": _codegen_normalize_column_names,
}

def _generate_clean_data_lines(recipe: list) -> list[str]:
    code = [
        "    # Match the dtypes Lumi uses after upload (numeric downcasting) so results agree with the app.",
        "    df = _downcast_dtypes(df)",
    ]
    if not recipe:
        code.append("    # No cleaning steps applied")
    else:
        for step in recipe:
            action = step.get('action')
            generator = _CODEGEN_REGISTRY.get(action)
            if generator:
                code.extend(generator(step))
            else:
                msg = f"Unknown action {action!r} in recipe was skipped (not supported by export)."
                code.append(f"    # WARNING: {msg}")
                code.append(f"    warnings.warn({repr(msg)})")
    code.append("    return df")
    return code

def _generate_validate_data_lines(rules: list) -> list[str]:
    code = []
    code.append("    \"\"\"")
    code.append("    Validates the DataFrame against the rules defined in the Lumi Rulebook.")
    code.append("    Returns a dictionary of {rule_description: violation_count_or_error}.")
    code.append("    \"\"\"")
    code.append("    violations = {}")

    active_rules = []
    if rules:
        active_rules = [r for r in rules if r.get('enabled', True) and r.get('type') != 'Informational']

    if not active_rules:
        code.append("    # No active validation rules defined")
    else:
        for r in active_rules:
            desc = r.get('desc', 'N/A')
            desc_repr = repr(desc)
            code.append(f"    # Rule: {_comment(r.get('type'))}")
            code.append(f"    # Description: {_comment(desc)}")
            code.append("    try:")
            try:
                lines = _rule_mask_lines(r, "        ")
            except KeyError as e:
                lines = [f"        raise KeyError({repr(str(e))})  # malformed rule definition"]
            if lines is None:
                lines = ["        mask = pd.Series(False, index=df.index)"]
            code.extend(lines)
            code.append(f"        count = mask.sum()")
            code.append(f"        if count > 0:")
            code.append(f"            violations[{desc_repr}] = int(count)")
            code.append("    except Exception as e:")
            code.append(f"        violations[{desc_repr}] = f\"Error: {{type(e).__name__}} - {{str(e)}}\"")
            code.append("") # Spacer between rules

    code.append("    return violations")
    return code

def generate_pipeline_code(recipe: list, rules: list = None) -> str:
    """
    Generates standalone Python code for a given cleaning recipe and optional validation rules.
    """
    code = [_IMPORTS, _DOWNCAST_HELPER, "def clean_data(df):"]
    code.extend(_generate_clean_data_lines(recipe))
    code.append("\n\ndef validate_data(df):")
    code.extend(_generate_validate_data_lines(rules))

    # Add example runner block
    code.append("\nif __name__ == \"__main__\":")
    code.append("    # Example usage:")
    code.append("    # df = pd.read_csv(\"your_data.csv\")")
    code.append("    # df_cleaned = clean_data(df)")
    code.append("    # report = validate_data(df_cleaned)")
    code.append("    # if report:")
    code.append("    #     print(\"Validation failed with issues:\", report)")
    code.append("    # else:")
    code.append("    #     print(\"Validation passed!\")")
    code.append("    pass")

    return "\n".join(code)

def generate_notebook_code(recipe: list, rules: list = None) -> str:
    """
    Generates a Jupyter Notebook (.ipynb) as a JSON-encoded string for the given recipe and rules.
    """
    clean_lines = ["def clean_data(df):\n"] + [f"{line}\n" for line in _generate_clean_data_lines(recipe)]
    validate_lines = ["def validate_data(df):\n"] + [f"{line}\n" for line in _generate_validate_data_lines(rules)]
    import_lines = [f"{line}\n" for line in _IMPORTS.rstrip("\n").split("\n")]
    import_lines += ["\n"] + [f"{line}\n" for line in _DOWNCAST_HELPER.rstrip("\n").split("\n")]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Lumi Data Cleaning & Validation Pipeline\n",
                "\n",
                "This notebook contains the data cleaning and validation pipeline automatically generated by **Lumi**.\n",
                "\n",
                "### Notebook structure:\n",
                "1. **Imports cell**: Loads `pandas` and `numpy` dependencies and the dtype helper.\n",
                "2. **`clean_data(df)` function**: Applies the sequential data cleaning recipe.\n",
                "3. **`validate_data(df)` function**: Runs assertions and checks active validation rules.\n",
                "4. **Example Runner**: Demonstrates how to run the pipeline on your datasets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": import_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": clean_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": validate_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Example Usage:\n",
                "# df = pd.read_csv(\"your_data.csv\")\n",
                "# df_cleaned = clean_data(df)\n",
                "# violations = validate_data(df_cleaned)\n",
                "# if violations:\n",
                "#     print(\"Validation failed with issues:\", violations)\n",
                "# else:\n",
                "#     print(\"Validation passed!\")\n"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    return json.dumps(notebook, indent=2)
