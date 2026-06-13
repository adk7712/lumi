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
if st.session_state.raw_data is not None:
    h_col1, h_col2, h_col3 = st.columns([6, 2, 2], vertical_alignment="bottom")
    with h_col1: # Main column for title
        st.subheader("LUMI")
    with h_col2: # Column for file uploader
        st.markdown('<div class="header-uploader-marker"></div>', unsafe_allow_html=True)
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
else:
    st.subheader("LUMI")

st.divider()

# Welcome view if no data is loaded yet
if st.session_state.raw_data is None:
    st.markdown("""
    <head>
        <meta name="description" content="Lumi is a professional web app for automated data scouting, interactive profiling diagnostics, and validation pipelines. Export Python script or Jupyter Notebook.">
        <meta name="keywords" content="data cleaning, validation pipeline, data profiling, pandas quality checks, python data cleaning app">
    </head>

    <!-- Animated background orbs -->
    <div class="welcome-bg">
        <div class="welcome-orb welcome-orb-1"></div>
        <div class="welcome-orb welcome-orb-2"></div>
        <div class="welcome-orb welcome-orb-3"></div>
    </div>

    <main class="welcome-hero">
        <span class="welcome-badge">✦ Open Source Data Quality Tool</span>
        <h1 class="welcome-title" id="welcome-header">Clean, Validate &amp; Export<br>Production-Ready Data Pipelines</h1>
        <p class="welcome-subtitle">Upload any CSV or Excel file. Lumi automatically scouts for issues, lets you build<br>validation rules, and exports a standalone Python script — all in one session.</p>
    </main>
    """, unsafe_allow_html=True)

    # Render CTA button or Uploader
    if not st.session_state.show_uploader:
        st.markdown('<div class="get-started-marker"></div>', unsafe_allow_html=True)
        if st.button("Upload Your Dataset →", key="get_started_btn"):
            st.session_state.show_uploader = True
            st.rerun()
        st.markdown('<p class="cta-helper">Free · No sign-up · Works with CSV &amp; XLSX</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="welcome-uploader-marker"></div>', unsafe_allow_html=True)
        welcome_uploader = st.file_uploader(
            "Drop your dataset here or click Browse",
            type=["csv", "xlsx"],
            key="welcome_uploader"
        )

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

    st.markdown("""
    <section class="welcome-grid">
        <article class="welcome-card" id="card-scout">
            <div class="welcome-card-step">1</div>
            <span class="material-icons welcome-card-icon" style="color: #f1c40f;">search</span>
            <h3 class="welcome-card-title">Proactive Scouting</h3>
            <p class="welcome-card-desc">Automatically scans for structural inconsistencies, high cardinality, outliers, and missing patterns — with recommended fixes.</p>
        </article>
        <article class="welcome-card" id="card-diagnose">
            <div class="welcome-card-step">2</div>
            <span class="material-icons welcome-card-icon" style="color: #2ecc71;">insights</span>
            <h3 class="welcome-card-title">Interactive Diagnostics</h3>
            <p class="welcome-card-desc">Visual diagnostics with null distributions, unique counts, Z-score outliers, and cross-feature correlation heatmaps.</p>
        </article>
        <article class="welcome-card" id="card-rules">
            <div class="welcome-card-step">3</div>
            <span class="material-icons welcome-card-icon" style="color: #3498db;">rule</span>
            <h3 class="welcome-card-title">Tailored Rulebook</h3>
            <p class="welcome-card-desc">Enforce validation standards with null checks, relational asserts, range checks, and custom Pandas query expressions.</p>
        </article>
        <article class="welcome-card" id="card-export">
            <div class="welcome-card-step">4</div>
            <span class="material-icons welcome-card-icon" style="color: #9b59b6;">code</span>
            <h3 class="welcome-card-title">One-Click Export</h3>
            <p class="welcome-card-desc">Export your entire cleaning recipe and quality rules as standalone Python scripts or Jupyter Notebooks.</p>
        </article>
    </section>
    """, unsafe_allow_html=True)
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
