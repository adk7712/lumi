import pandas as pd
import datetime

def generate_evidence_report(
    df: pd.DataFrame, 
    rules: list, 
    cleaning_recipe: list = None, 
    intermediate_states: list = None
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
        if intermediate_states and len(intermediate_states) > 0:
            orig_df = intermediate_states[0][3]
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
        if cleaning_recipe and len(cleaning_recipe) > 0:
            lineage_block.append("## Data Lineage & Audit Log\n")
            lineage_block.append("The following cleaning sequence was successfully executed to build the cleaned dataset:\n")
            for idx, step in enumerate(cleaning_recipe, 1):
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
