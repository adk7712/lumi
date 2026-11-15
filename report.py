import re
import pandas as pd
import datetime


def _code(value, in_table: bool = False) -> str:
    """Markdown inline-code span for arbitrary text: backticks inside the value are handled by
    using a longer fence, and (in a table cell) ``|`` and line breaks are escaped."""
    text = " ".join(str(value).splitlines())
    if in_table:
        text = text.replace("|", "\\|")
    runs = [len(m) for m in re.findall(r"`+", text)]
    fence = "`" * (max(runs) + 1 if runs else 1)
    pad = " " if runs else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _cell(value) -> str:
    """Plain text safe for a markdown table cell."""
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def _describe_step(idx: int, step: dict) -> str:
    act = step.get('action')
    col = step.get('column', 'Dataset')
    c = _code(col)

    if act == "drop_column":
        desc = f"Dropped column {c}"
    elif act == "rename_column":
        desc = f"Renamed column {c} to {_code(step.get('value'))}"
    elif act == "strip_whitespace":
        desc = f"Stripped leading/trailing whitespace in column {c}"
    elif act == "normalize_text":
        desc = f"Normalized text in column {c} using method: {_code(step.get('value'))}"
    elif act == "cast_type":
        desc = f"Casted column {c} to data type: {_code(step.get('dtype'))}"
    elif act == "fill_null":
        desc = f"Filled missing values in column {c} with strategy/value: {_code(step.get('value'))}"
    elif act == "cap_range":
        desc = f"Capped values in column {c} within bounds: {_code('[' + str(step.get('min')) + ', ' + str(step.get('max')) + ']')}"
    elif act == "extract_datetime":
        desc = f"Extracted {_code(step.get('component'))} component from column {c} into new column {_code(step.get('new_column'))}"
    elif act == "drop_violated":
        rule_desc = (step.get('rule') or {}).get('desc', 'custom check')
        desc = f"Dropped rows violating validation constraint: {_code(rule_desc)}"
    elif act == "reorder_columns":
        desc = f"Reordered columns to custom order: {_code(step.get('value'))}"
    elif act == "replace":
        desc = f"Replaced values matching {_code(step.get('find'))} with {_code(step.get('replace'))} (Regex: {_code(step.get('regex'))})"
    elif act == "drop_nulls":
        desc = f"Dropped rows with missing values in column {c}"
    elif act == "drop_duplicates":
        desc = "Removed fully duplicate rows from the dataset"
    elif act == "drop_empty_columns":
        desc = "Dropped columns that contain only missing values"
    elif act == "drop_empty_rows":
        desc = "Dropped rows in which every value is missing"
    elif act == "log_transform":
        desc = f"Applied log(1 + x) transform to numeric column {c} (negative values clipped to 0)"
    elif act == "normalize_column_names":
        desc = f"Normalized all column names using method: {_code(step.get('value', 'snake_case'))}"
    else:
        desc = f"Applied custom action {_code(act)} on column {c}"

    return f"{idx}. **{str(act).replace('_', ' ').title()}**: {desc}"


def generate_evidence_report(
    df: pd.DataFrame, 
    rules: list, 
    cleaning_recipe: list = None, 
    original_df: pd.DataFrame = None
) -> str:
    """
    Generates a Markdown evidence report summarizing active rule evaluations,
    dataset cleaning metrics, and data lineage history.
    """
    from rule_utils import evaluate_rule
    
    total_rows = len(df)
    active_rules = [r for r in rules if r.get('enabled', True)]
    total_rules = len(active_rules)
    
    # 1. Gather Before/After Dataset Metrics
    metrics_block = []
    has_metrics = False
    try:
        if original_df is not None:
            orig_df = original_df
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
        # Gracefully degrade if original_df is not available or not valid
        pass

    # 2. Gather Data Lineage / Audit Log
    lineage_block = []
    has_lineage = False
    if cleaning_recipe and len(cleaning_recipe) > 0:
        lineage_block.append("## Data Lineage & Audit Log\n")
        lineage_block.append("The following steps were applied in order to build the cleaned dataset; steps that were skipped or produced warnings are noted in the audit log:\n")
        for idx, step in enumerate(cleaning_recipe, 1):
            try:
                lineage_block.append(_describe_step(idx, step))
            except Exception as e:
                lineage_block.append(f"{idx}. Could not describe step {idx} ({type(e).__name__}).")
        lineage_block.append("\n")
        has_lineage = True

    # 3. Pre-evaluate rules and calculate violations
    violations_summary = []
    total_violations = 0
    
    for r in active_rules:
        r_type = r.get('type') or 'Unknown'
        desc = r.get('desc') or r.get('type') or '(no description)'
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
        report.append(f"| {_cell(s['type'])} | {_code(s['desc'], in_table=True)} | {s['status']} | {s['count'] if s['type'] != 'Informational' else 'N/A'} |")
    report.append("\n")
    
    # Detail violations
    has_details = any(s['count'] > 0 or s['status'] == "ERROR" for s in violations_summary if s['type'] != "Informational")
    if has_details:
        report.append("## Violation Details\n")
        for s in violations_summary:
            if s['type'] == "Informational":
                continue
            if s['status'] == "FAILED":
                report.append(f"### ❌ {s['type']}: {_code(s['desc'])}")
                report.append(f"* **Violation Count:** {s['count']}")
                idx_str = ", ".join(map(str, s['indices']))
                if s['count'] > 100:
                    idx_str += ", ... (truncated)"
                report.append(f"* **Violating Row Indices:** `[{idx_str}]`")
                report.append("")
            elif s['status'] == "ERROR":
                report.append(f"### ⚠️ {s['type']}: {_code(s['desc'])}")
                report.append(f"* **Status:** Evaluation Error")
                report.append(f"* **Error Message:** {_code(s['error'])}")
                report.append("")
                
    return "\n".join(report)
