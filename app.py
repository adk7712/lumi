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
st.set_page_config(
    page_title="Lumi — Interactive Data Cleaning & Quality Validation Pipeline Generator",
    layout="wide"
)

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

# Welcome view if no data is loaded yet
if st.session_state.raw_data is None:
    # Render SEO metadata and key capabilities
    st.markdown("""
    <head>
        <meta name="description" content="Lumi is a professional web app for automated data scouting, interactive profiling diagnostics, and validation pipelines. Export Python script or Jupyter Notebook.">
        <meta name="keywords" content="data cleaning, validation pipeline, data profiling, pandas quality checks, python data cleaning app">
    </head>
    <main class="welcome-hero">
        <h1 class="welcome-title" id="welcome-header">Lumi — Interactive Data Cleaning & Quality Validation</h1>
        <p class="welcome-subtitle">Identify issues, configure rules, and compile production-ready data pipelines instantly.</p>
    </main>
    
    <section class="welcome-grid">
        <article class="welcome-card" id="card-scout">
            <span class="material-icons welcome-card-icon" style="color: #f1c40f;">search</span>
            <h3 class="welcome-card-title">Proactive Scouting</h3>
            <p class="welcome-card-desc">Automatically scans datasets for structural inconsistencies, high cardinality, outliers, and missing patterns with recommended solutions.</p>
        </article>
        <article class="welcome-card" id="card-diagnose">
            <span class="material-icons welcome-card-icon" style="color: #2ecc71;">insights</span>
            <h3 class="welcome-card-title">Interactive Diagnostics</h3>
            <p class="welcome-card-desc">Rich visual diagnostics showing null distributions, unique counts, Z-score outliers, and cross-feature correlation filters.</p>
        </article>
        <article class="welcome-card" id="card-rules">
            <span class="material-icons welcome-card-icon" style="color: #3498db;">rule</span>
            <h3 class="welcome-card-title">Tailored Rulebook</h3>
            <p class="welcome-card-desc">Enforce rigorous validation standards with custom null checks, relational asserts, range checks, and custom Pandas query expressions.</p>
        </article>
        <article class="welcome-card" id="card-export">
            <span class="material-icons welcome-card-icon" style="color: #9b59b6;">code</span>
            <h3 class="welcome-card-title">One-Click Pipeline Export</h3>
            <p class="welcome-card-desc">Export your entire interactive data cleaning recipe and active quality rules as standalone Python scripts or Jupyter Notebooks.</p>
        </article>
    </section>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="welcome-uploader-container">', unsafe_allow_html=True)
    welcome_uploader = st.file_uploader(
        "Upload Dataset (CSV or XLSX)", 
        type=["csv", "xlsx"], 
        key="welcome_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if welcome_uploader:
        file_id = f"{welcome_uploader.file_id}_{welcome_uploader.name}_{welcome_uploader.size}"
        is_large = welcome_uploader.size > 50 * 1024 * 1024
        if is_large:
            st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.", icon="⚠️")
        raw_df = load_data(welcome_uploader, nrows=MAX_SAMPLE_ROWS if is_large else None)
        st.session_state.original_full_data = raw_df
        if not is_large and len(raw_df) > MAX_SAMPLE_ROWS:
            st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        else:
            st.session_state.raw_data = raw_df

        st.session_state.last_file_hash = file_id
        st.session_state.active_features = []
        st.session_state.scanned_columns, st.session_state.cleaning_recipe, st.session_state.rules = set(), [], []

        base_df = st.session_state.raw_data
        bh = int((1 - (base_df.isnull().sum().sum() / base_df.size)) * 100) if base_df.size > 0 else 0
        st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
        st.toast("Dataset Analyzed")
        st.rerun()
        
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
