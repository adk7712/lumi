import html
import streamlit as st
import pandas as pd
from lumi_state.steps import get_rule_masks
from ui_utils import get_heatmap_styles, VIOLATION_PREVIEW_LIMIT

def _render_legend(rules):
    """Colour key for the heatmap so colour is not the only carrier of meaning."""
    items = []
    for r in rules:
        color = html.escape(str(r.get('color', '')), quote=True)
        desc = html.escape(str(r.get('desc', 'Rule')))
        items.append(
            f'<span style="display:inline-block;width:0.8em;height:0.8em;background:{color};'
            f'border-radius:2px;margin-right:0.4em;vertical-align:middle;"></span>{desc}'
        )
    st.markdown(
        '<div class="violation-legend" style="font-size:0.8rem;margin-bottom:0.5rem;">'
        '<b>Colour key:</b> ' + " &nbsp;&nbsp; ".join(items) + '</div>',
        unsafe_allow_html=True,
    )


def render_violation_browser(df):
    st.divider()
    st.subheader("Violation Browser")

    active_rules_for_heatmap = [r for r in st.session_state.rules if r.get('enabled', True) and r.get('type') != "Informational"]

    if not active_rules_for_heatmap:
        st.info("No active rules to check for violations.")
    else:
        combined_mask = pd.Series(False, index=df.index)
        errors = []
        masks = get_rule_masks(df)
        for idx, rule in enumerate(st.session_state.rules):
            if not rule.get('enabled', True) or rule.get('type') == "Informational":
                continue
            result = masks.get(idx)
            if isinstance(result, Exception):
                errors.append(f"Violation Browser Rule Error ({rule.get('desc', 'N/A')}): {str(result)}")
            elif result is not None:
                combined_mask |= result
        
        if errors:
            for err in errors:
                st.toast(err)

        violation_df = df[combined_mask]

        if violation_df.empty:
            st.success("No violations found in the current dataset.")
        else:
            if len(violation_df) > VIOLATION_PREVIEW_LIMIT:
                st.warning(f"Found {len(violation_df):,} rows with violations. Showing top {VIOLATION_PREVIEW_LIMIT}.")
            else:
                st.warning(f"Found {len(violation_df):,} row(s) with violations.")
            # Slice first, then style: styling every violating row is wasted work past the limit.
            preview_df = violation_df.head(VIOLATION_PREVIEW_LIMIT).reset_index(drop=True)
            _render_legend(active_rules_for_heatmap)
            heatmap_sdf, _ = get_heatmap_styles(preview_df, active_rules_for_heatmap)
            st.dataframe(preview_df.style.apply(lambda _: heatmap_sdf, axis=None), width="stretch")
