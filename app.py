import streamlit as st
# Trigger hot reload for overview view updates with new global placeholders
from ui_utils import inject_custom_css
from state_manager import initialize_state, load_data, MAX_SAMPLE_ROWS
from views import (
    render_overview_tab,
    render_diagnostics_tab,
    render_insights_tab,
    render_rulebook_tab,
    render_transformations_tab,
    render_audit_log_tab,
    render_pipeline_preview_tab,
    render_violation_browser,
    render_landing_page
)
from scout import generate_proposals

# Set page config
st.set_page_config(
    page_title="Lumi",
    layout="wide"
)

# Inject Custom CSS
inject_custom_css(st)

# Initialize Session State
initialize_state()

# --- HEADER (only shown after a dataset is loaded) ---
if st.session_state.raw_data is not None:
    h_col1, h_col2 = st.columns([10, 2], vertical_alignment="bottom")
    with h_col1:
        st.markdown('<div class="lumi-logo-button">', unsafe_allow_html=True)
        if st.button("LUMI", key="lumi_logo"):
            initialize_state(from_reset=True)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with h_col2:
        u_c1, u_c2 = st.columns(2)
        if u_c1.button("Undo", key="undo_btn", width="stretch", disabled=len(st.session_state.cleaning_recipe) == 0):
            st.session_state.cleaning_recipe.pop()
            st.session_state.intermediate_states.pop()
            st.toast("Last step undone")
            st.rerun()
        if u_c2.button("Reset", key="reset_all", width="stretch"):
            initialize_state(from_reset=True)
            st.rerun()
else:
    st.subheader("LUMI")

st.divider()

# Welcome view if no data is loaded yet
if st.session_state.raw_data is None:
    render_landing_page()
    st.stop()

# Get the latest dataframe from cache
df = st.session_state.intermediate_states[-1][3]

# --- TABS ---
tab_overview, tab_diagnostics, tab_insights, tab_rulebook, tab_transformations, tab_audit, tab_pipeline = st.tabs([
    "Overview", "Diagnostics", "Visual Insights", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"
])

with tab_overview:
    render_overview_tab(df)

with tab_diagnostics:
    render_diagnostics_tab(df)

with tab_insights:
    render_insights_tab(df)

with tab_rulebook:
    render_rulebook_tab(df)

with tab_transformations:
    render_transformations_tab(df)

with tab_audit:
    render_audit_log_tab()

with tab_pipeline:
    render_pipeline_preview_tab(df)

# Bottom violation browser
render_violation_browser(df)
