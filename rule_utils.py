import pandas as pd
from query_guard import UnsafeQueryError, validate_query_expression, safe_query_index  # noqa: F401


def _relational_violations(a, b, op):
    """Boolean violation mask for a Relational Check (``a <op> b`` describes the VALID state).

    Semantics (the engine and the exported script share this exact source):
      * a null (NaN / None / pd.NA / NaT) in ``a`` or in ``b`` is a violation for EVERY
        operator, for float and nullable dtypes alike;
      * cells that cannot be compared (e.g. 2 < 'x') violate individually - one bad cell
        never flags the whole column;
      * an unsupported operator flags every row.
    Always returns a plain boolean Series (never contains pd.NA).
    """
    import operator
    ops = {">": operator.gt, "<": operator.lt, "==": operator.eq,
           "!=": operator.ne, ">=": operator.ge, "<=": operator.le}
    a = pd.Series(a)
    b_is_series = isinstance(b, pd.Series)
    if op not in ops:
        return pd.Series(True, index=a.index, dtype=bool)
    fn = ops[op]
    null = a.isna().to_numpy(dtype=bool)
    if b_is_series:
        null = null | b.isna().to_numpy(dtype=bool)
    else:
        try:
            if bool(pd.isna(b)):
                return pd.Series(True, index=a.index, dtype=bool)
        except (TypeError, ValueError):
            pass
    try:
        valid = fn(a, b)
        valid = pd.Series(valid, index=a.index).fillna(False).astype(bool).to_numpy()
    except (TypeError, ValueError):
        # Some cell is not comparable: decide row by row, vectorised path stays the fast case.
        av = a.tolist()
        bv = b.tolist() if b_is_series else None
        valid = []
        for i, x in enumerate(av):
            y = bv[i] if b_is_series else b
            try:
                valid.append(bool(fn(x, y)) if not null[i] else False)
            except (TypeError, ValueError):
                valid.append(False)
        valid = pd.Series(valid, dtype=bool).to_numpy()
    return pd.Series(~(valid & ~null), index=a.index, dtype=bool)


def evaluate_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """
    Evaluates a rule against a DataFrame and returns a boolean mask for violating rows.

    Args:
        df (pd.DataFrame): The DataFrame to evaluate.
        rule (dict): The rule definition.

    Returns:
        pd.Series: A boolean Series where True indicates a row violates the rule.
    """
    mask = pd.Series(False, index=df.index)

    try:
        rule_type = rule.get('type')
        if rule_type == "Null Check":
            mask = df[rule['col']].isnull()
        elif rule_type == "Range Check":
            col_name = rule['col']  # KeyError if missing — intentional for malformed rules
            rule_min = rule.get('min')
            rule_max = rule.get('max')
            # Guard against NaN/None/inf in bounds (can happen from corrupted JSON)
            import math
            if rule_min is None or rule_max is None:
                mask = pd.Series(False, index=df.index)
            elif isinstance(rule_min, float) and (math.isnan(rule_min) or math.isinf(rule_min)):
                mask = pd.Series(False, index=df.index)
            elif isinstance(rule_max, float) and (math.isnan(rule_max) or math.isinf(rule_max)):
                mask = pd.Series(False, index=df.index)
            else:
                # Coerce to numeric to avoid errors with mixed types before comparison
                col_numeric = pd.to_numeric(df[col_name], errors='coerce')
                # Flag values outside the range OR non-null values that could not be coerced.
                # Genuine nulls are NOT range violations — that is the Null Check's job.
                out_of_range_mask = (col_numeric < rule_min) | (col_numeric > rule_max)
                unparseable_mask = col_numeric.isnull() & df[col_name].notnull()
                # Nullable dtypes (Int64/Float64/boolean) yield pd.NA in comparisons: NA -> not a violation.
                mask = (out_of_range_mask.fillna(False).astype(bool)
                        | unparseable_mask.fillna(False).astype(bool))
        elif rule_type == "Relational Check":
            a = df[rule['col_a']]
            b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
            op = rule['op']
            
            # The rule describes the valid state (e.g., "Age > 18"); nulls on either side and
            # incomparable cells are violations (see _relational_violations).
            mask = _relational_violations(a, b, op)
        elif rule_type == "Custom Expression":
            # The query describes the VIOLATION condition — rows that match are flagged.
            # e.g., query = "age > 100 or age < 0" flags those exact rows as violations.
            violated_indices = safe_query_index(df, rule['query'])
            mask = pd.Series(df.index.isin(violated_indices), index=df.index, dtype=bool)
        elif rule_type == "Informational":
            # Informational rules don't have "violations", so we return an all-False mask.
            mask = pd.Series(False, index=df.index)

    except (KeyError, TypeError, ValueError) as e:
        # Re-raise the exception to be handled by the caller,
        # which can provide more context (e.g., in UI).
        raise ValueError(f"Error evaluating rule ({rule.get('desc', 'N/A')}): {type(e).__name__} - {e}") from e
    except Exception as e:
        raise ValueError(f"An unexpected error occurred evaluating rule ({rule.get('desc', 'N/A')}): {type(e).__name__} - {e}") from e

    return mask


def create_resolution_step(rule: dict, method: str) -> dict:
    """
    Maps a selected resolution method and rule definition to a concrete cleaning step.
    """
    rule_type = rule.get('type')
    
    if rule_type == "Null Check":
        if method == "Drop Rows":
            return {"action": "drop_nulls", "column": rule['col']}
        elif "Imputer" in method:
            # e.g., "KNN Imputer" -> "knn", "Iterative Imputer" -> "iterative"
            return {"action": "fill_null", "column": rule['col'], "value": method.split()[0].lower()}
        else:
            # e.g., "Fill with Mean" -> "mean", "Fill with Median" -> "median"
            return {"action": "fill_null", "column": rule['col'], "value": method.split()[-1].lower()}
            
    elif rule_type == "Range Check":
        if method == "Drop Rows":
            return {"action": "drop_violated", "rule": rule}
        elif method == "Log Transform":
            return {"action": "log_transform", "column": rule['col']}
        else:
            # "Cap at Bounds"
            return {"action": "cap_range", "column": rule['col'], "min": rule['min'], "max": rule['max']}
            
    else:
        # Default fallback for other rule types (e.g. Relational Check, Custom Expression)
        # where the only method is "Drop Violated Rows"
        return {"action": "drop_violated", "rule": rule}


from report import generate_evidence_report
