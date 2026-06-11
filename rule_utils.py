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
    Generates a Markdown evidence report summarizing active rule evaluations,
    dataset cleaning metrics, and data lineage history.
    """
    import datetime
    import streamlit as st
    
    total_rows = len(df)
    active_rules = [r for r in rules if r.get('enabled', True)]
    total_rules = len(active_rules)
    
    # 1. Gather Before/After Dataset Metrics
    metrics_block = []
    has_metrics = False
    try:
        # Check if running within a Streamlit context with active state
        if st.runtime.exists() and 'intermediate_states' in st.session_state and st.session_state.intermediate_states:
            orig_df = st.session_state.intermediate_states[0][3]
            orig_rows, orig_cols = orig_df.shape
            orig_nulls = orig_df.isnull().sum().sum()
            orig_size = orig_df.size
            orig_null_pct = (orig_nulls / orig_size * 100) if orig_size > 0 else 0
            
            clean_rows, clean_cols = df.shape
            clean_nulls = df.isnull().sum().sum()
            clean_size = df.size
            clean_null_pct = (clean_nulls / clean_size * 100) if clean_size > 0 else 0
            
            metrics_block.append("## Data Cleaning Impact Metrics\n")
            metrics_block.append("| Metric | Original Dataset | Cleaned Dataset | Change |")
            metrics_block.append("| :--- | :--- | :--- | :--- |")
            metrics_block.append(f"| **Dimensions** | {orig_rows} rows × {orig_cols} cols | {clean_rows} rows × {clean_cols} cols | {clean_rows - orig_rows} rows, {clean_cols - orig_cols} cols |")
            metrics_block.append(f"| **Missing Values** | {orig_nulls} | {clean_nulls} | {clean_nulls - orig_nulls} |")
            metrics_block.append(f"| **Null Density** | {orig_null_pct:.2f}% | {clean_null_pct:.2f}% | {clean_null_pct - orig_null_pct:+.2f}% |")
            metrics_block.append("\n")
            has_metrics = True
    except Exception:
        # Gracefully degrade if intermediate_states are not mockable or not present
        pass

    # 2. Gather Data Lineage / Audit Log
    lineage_block = []
    has_lineage = False
    try:
        if st.runtime.exists() and 'cleaning_recipe' in st.session_state and st.session_state.cleaning_recipe:
            recipe = st.session_state.cleaning_recipe
            lineage_block.append("## Data Lineage & Audit Log\n")
            lineage_block.append("The following cleaning sequence was successfully executed to build the cleaned dataset:\n")
            for idx, step in enumerate(recipe, 1):
                act = step.get('action')
                col = step.get('column', 'Dataset')
                
                # Render readable description
                if act == "drop_column":
                    desc = f"Dropped column `{col}`"
                elif act == "rename_column":
                    desc = f"Renamed column `{col}` to `{step.get('value')}`"
                elif act == "strip_whitespace":
                    desc = f"Stripped leading/trailing whitespace in column `{col}`"
                elif act == "normalize_text":
                    desc = f"Normalized text in column `{col}` using method: `{step.get('value')}`"
                elif act == "cast_type":
                    desc = f"Casted column `{col}` to data type: `{step.get('dtype')}`"
                elif act == "fill_null":
                    desc = f"Filled missing values in column `{col}` with strategy/value: `{step.get('value')}`"
                elif act == "cap_range":
                    desc = f"Capped values in column `{col}` within bounds: `[{step.get('min')}, {step.get('max')}]`"
                elif act == "extract_datetime":
                    desc = f"Extracted `{step.get('component')}` component from column `{col}` into new column `{step.get('new_column')}`"
                elif act == "drop_violated":
                    rule_desc = step.get('rule', {}).get('desc', 'custom check')
                    desc = f"Dropped rows violating validation constraint: `{rule_desc}`"
                elif act == "reorder_columns":
                    desc = f"Reordered columns to custom order: `{step.get('value')}`"
                elif act == "replace":
                    desc = f"Replaced values matching `{step.get('find')}` with `{step.get('replace')}` (Regex: `{step.get('regex')}`)"
                else:
                    desc = f"Applied custom action `{act}` on column `{col}`"
                    
                lineage_block.append(f"{idx}. **{act.replace('_', ' ').title()}**: {desc}")
            lineage_block.append("\n")
            has_lineage = True
    except Exception:
        pass

    # 3. Pre-evaluate rules and calculate violations
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
    
    if has_metrics:
        report.extend(metrics_block)
        
    if has_lineage:
        report.extend(lineage_block)
        
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
