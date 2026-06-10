import pandas as pd

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
            # Coerce to numeric to avoid errors with mixed types before comparison
            col_numeric = pd.to_numeric(df[rule['col']], errors='coerce')
            # Flag values outside the range OR values that could not be coerced (are NaN)
            out_of_range_mask = (col_numeric < rule['min']) | (col_numeric > rule['max'])
            nan_mask = col_numeric.isnull()
            mask = out_of_range_mask | nan_mask
        elif rule_type == "Relational Check":
            a = df[rule['col_a']]
            b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
            op = rule['op']
            
            # The rule describes the valid state (e.g., "Age > 18").
            # To correctly handle NaNs as violations, we first evaluate the valid condition
            # and then invert the result. Comparisons with NaN always return False.
            # Thus, ~False will correctly flag NaNs as True (violations).
            if op == ">": valid = (a > b)
            elif op == "<": valid = (a < b)
            elif op == "==": valid = (a == b)
            elif op == "!=": valid = (a != b)
            elif op == ">=": valid = (a >= b)
            elif op == "<=": valid = (a <= b)
            else: valid = pd.Series(False, index=df.index)
            
            mask = ~valid
        elif rule_type == "Custom Expression":
            # df.query() returns rows that satisfy the condition (non-violators).
            # We need to find the violators.
            valid_indices = df.query(rule['query']).index
            mask = ~df.index.isin(valid_indices)
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
