import streamlit as st
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
    render_violation_browser
)
from scout import generate_proposals

# Set page config
st.set_page_config(page_title="Lumi", layout="wide")

# Inject Custom CSS
inject_custom_css(st)

# Initialize Session State
initialize_state()

# --- HEADER ---
h_col1, h_col2, h_col3 = st.columns([6, 2, 2], vertical_alignment="bottom")
with h_col1: # Main column for title
    st.subheader("LUMI")
with h_col2: # Column for file uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], label_visibility="collapsed", key="global_uploader")
    if uploaded_file:
        # Optimization: Use file attributes for a lightweight identifier instead of full-file hashing
        # to prevent memory bottlenecks on large datasets.
        file_id = f"{uploaded_file.file_id}_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_file_hash != file_id:
            is_large = uploaded_file.size > 50 * 1024 * 1024  # 50MB limit warning
            if is_large:
                st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.", icon="⚠️")
            raw_df = load_data(uploaded_file, nrows=MAX_SAMPLE_ROWS if is_large else None)
            st.session_state.original_full_data = raw_df
            if not is_large and len(raw_df) > MAX_SAMPLE_ROWS:
                st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
            else:
                st.session_state.raw_data = raw_df

            st.session_state.last_file_hash = file_id
            # Reset dependent state
            st.session_state.active_features = []
            st.session_state.scanned_columns, st.session_state.cleaning_recipe, st.session_state.rules = set(), [], []

            # Reset intermediate states
            base_df = st.session_state.raw_data
            bh = int((1 - (base_df.isnull().sum().sum() / base_df.size)) * 100) if base_df.size > 0 else 0
            st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]

            st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
            st.toast("Dataset Analyzed")
            st.rerun()

with h_col3: # Column for reset button
    u_c1, u_c2 = st.columns(2)
    if u_c1.button("Undo", key="undo_btn", width="stretch", disabled=len(st.session_state.cleaning_recipe) == 0):
        st.session_state.cleaning_recipe.pop()
        st.session_state.intermediate_states.pop()
        st.toast("Last step undone")
        st.rerun()
    if u_c2.button("Reset", key="reset_all", width="stretch"):
        initialize_state(from_reset=True)
        st.rerun()

st.divider()

# Get the latest dataframe from cache
df = st.session_state.intermediate_states[-1][3]

# --- TABS ---
tab1, tab2, tab_insights, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Diagnostics", "Visual Insights", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"
])

with tab1:
    render_overview_tab(df)

with tab2:
    render_diagnostics_tab(df)

with tab_insights:
    render_insights_tab(df)

with tab3:
    render_rulebook_tab(df)

with tab4:
    render_transformations_tab(df)

with tab5:
    render_audit_log_tab()

with tab6:
    render_pipeline_preview_tab(df)

# Bottom violation browser
render_violation_browser(df)
