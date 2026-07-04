import streamlit as st
import pandas as pd
from rule_utils import evaluate_rule
from ui_utils import get_heatmap_styles, VIOLATION_PREVIEW_LIMIT

def render_violation_browser(df):
    st.divider()
    st.subheader("Violation Browser")

    active_rules_for_heatmap = [r for r in st.session_state.rules if r.get('enabled', True) and r.get('type') != "Informational"]

    if not active_rules_for_heatmap:
        st.info("No active rules to check for violations.")
    else:
        combined_mask = pd.Series(False, index=df.index)
        errors = []
        for rule in active_rules_for_heatmap:
            try:
                combined_mask |= evaluate_rule(df, rule)
            except Exception as e:
                errors.append(f"Violation Browser Rule Error ({rule.get('desc', 'N/A')}): {str(e)}")
                continue
        
        if errors:
            for err in errors:
                st.toast(err, icon="⚠️")

        violation_df = df[combined_mask]

        if violation_df.empty:
            st.success("No violations found in the current dataset.")
        else:
            st.warning(f"Found {len(violation_df):,} rows with violations. Showing top {VIOLATION_PREVIEW_LIMIT}.")
            heatmap_sdf, _ = get_heatmap_styles(violation_df, active_rules_for_heatmap)
            st.dataframe(violation_df.head(VIOLATION_PREVIEW_LIMIT).style.apply(lambda _: heatmap_sdf.head(VIOLATION_PREVIEW_LIMIT), axis=None), width="stretch")
