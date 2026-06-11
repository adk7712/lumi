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


def generate_evidence_report(df: pd.DataFrame, rules: list) -> str:
    """
    Generates a Markdown evidence report summarizing active rule evaluations.
    """
    import datetime
    
    total_rows = len(df)
    active_rules = [r for r in rules if r.get('enabled', True)]
    total_rules = len(active_rules)
    
    # Pre-evaluate rules and calculate violations
    violations_summary = []
    total_violations = 0
    
    for r in active_rules:
        r_type = r['type']
        desc = r['desc']
        status = "PASSED"
        count = 0
        indices = []
        error_msg = None
        
        if r_type != "Informational":
            try:
                mask = evaluate_rule(df, r)
                count = mask.sum()
                if count > 0:
                    status = "FAILED"
                    total_violations += count
                    # Get index of violating rows (limit to first 100 for report size protection)
                    indices = df.index[mask].tolist()[:100]
            except Exception as e:
                status = "ERROR"
                error_msg = str(e)
        else:
            status = "INFO"
            
        violations_summary.append({
            'rule': r,
            'type': r_type,
            'desc': desc,
            'status': status,
            'count': count,
            'indices': indices,
            'error': error_msg
        })
        
    report = []
    report.append("# LUMI - Data Validation Evidence Report\n")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Rows Evaluated: {total_rows}")
    report.append(f"Total Active Rules: {total_rules}")
    report.append(f"Total Rule Violations: {total_violations}\n")
    
    report.append("## Rule Evaluation Summary\n")
    report.append("| Rule Type | Description | Status | Violation Count |")
    report.append("| :--- | :--- | :--- | :--- |")
    for s in violations_summary:
        report.append(f"| {s['type']} | `{s['desc']}` | {s['status']} | {s['count'] if s['type'] != 'Informational' else 'N/A'} |")
    report.append("\n")
    
    # Detail violations
    has_details = any(s['count'] > 0 or s['status'] == "ERROR" for s in violations_summary if s['type'] != "Informational")
    if has_details:
        report.append("## Violation Details\n")
        for s in violations_summary:
            if s['type'] == "Informational":
                continue
            if s['status'] == "FAILED":
                report.append(f"### ❌ {s['type']}: `{s['desc']}`")
                report.append(f"* **Violation Count:** {s['count']}")
                idx_str = ", ".join(map(str, s['indices']))
                if s['count'] > 100:
                    idx_str += ", ... (truncated)"
                report.append(f"* **Violating Row Indices:** `[{idx_str}]`")
                report.append("")
            elif s['status'] == "ERROR":
                report.append(f"### ⚠️ {s['type']}: `{s['desc']}`")
                report.append(f"* **Status:** Evaluation Error")
                report.append(f"* **Error Message:** `{s['error']}`")
                report.append("")
                
    return "\n".join(report)
