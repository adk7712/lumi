import streamlit as st
from state_manager import load_data, MAX_SAMPLE_ROWS
from scout import generate_proposals

def render_landing_page():
    st.markdown("""
    <head>
        <meta name="description" content="Lumi is a professional web app for automated data scouting, interactive profiling diagnostics, and validation pipelines. Export Python script or Jupyter Notebook.">
        <meta name="keywords" content="data cleaning, validation pipeline, data profiling, pandas quality checks, python data cleaning app">
    </head>
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
        cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
        with cta_col:
            st.markdown('<div class="get-started-marker"></div>', unsafe_allow_html=True)
            if st.button("Upload Your Dataset →", key="get_started_btn", use_container_width=True):
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

    st.markdown("""<section class="welcome-grid">
        <article class="welcome-card" id="card-scout"><span class="material-icons welcome-card-icon" style="color: #f1c40f;">search</span><h3 class="welcome-card-title">Proactive Scouting</h3><p class="welcome-card-desc">Automatically scans for structural inconsistencies, high cardinality, outliers, and missing patterns — with recommended fixes.</p></article>
        <article class="welcome-card" id="card-diagnose"><span class="material-icons welcome-card-icon" style="color: #a855f7;">insights</span><h3 class="welcome-card-title">Interactive Diagnostics</h3><p class="welcome-card-desc">Visual diagnostics with null distributions, unique counts, Z-score outliers, and cross-feature correlation heatmaps.</p></article>
        <article class="welcome-card" id="card-rules"><span class="material-icons welcome-card-icon" style="color: #8b5cf6;">rule</span><h3 class="welcome-card-title">Tailored Rulebook</h3><p class="welcome-card-desc">Enforce validation standards with null checks, relational asserts, range checks, and custom Pandas query expressions.</p></article>
        <article class="welcome-card" id="card-export"><span class="material-icons welcome-card-icon" style="color: #c084fc;">code</span><h3 class="welcome-card-title">One-Click Export</h3><p class="welcome-card-desc">Export your entire cleaning recipe and quality rules as standalone Python scripts or Jupyter Notebooks.</p></article>
    </section>
    <section class="welcome-showcase">
        <div class="welcome-showcase-card" id="card-pipeline">
            <div>
                <span class="wsc-label">Pipeline Export</span>
                <h2 class="wsc-title">Validated data you can trust.</h2>
                <p class="wsc-desc">Lumi bridges the gap between raw exploration and production engineering. Stop manually re-cleaning the same messy datasets over and over.</p>
                <ul class="wsc-bullets">
                    <li>100% client-side privacy — data never leaves your browser</li>
                    <li>Native Pandas integration — scripts run anywhere Python does</li>
                    <li>Export as .py script or Jupyter Notebook (.ipynb)</li>
                </ul>
            </div>
            <div class="wsc-code"><span class="code-comment"># Lumi Generated Pipeline
import pandas as pd</span>
<span class="code-keyword">def</span> <span class="code-func">validate_data</span>(df):
    <span class="code-comment"># Check for missing values</span>
    df = df.dropna(subset=[<span class="code-string">'user_id'</span>])
    <span class="code-comment"># Range check: age [0-120]</span>
    df = df[df[<span class="code-string">'age'</span>].between(0, 120)]
    <span class="code-comment"># Outlier detection via Z-score</span>
    z = (df[<span class="code-string">'spend'</span>] - df[<span class="code-string">'spend'</span>].mean()) / df[<span class="code-string">'spend'</span>].std()
    <span class="code-keyword">return</span> df[z.abs() &lt;= 3]</div>
        </div>
        <div class="welcome-showcase-card" id="card-audit">
            <div>
                <span class="wsc-label">Rulebook &amp; Audit Log</span>
                <h2 class="wsc-title">Every change tracked. Every rule enforced.</h2>
                <p class="wsc-desc">Define null checks, range constraints, relational asserts, and custom Pandas expressions. See exactly which rows fail which rules, and undo any step instantly.</p>
                <ul class="wsc-bullets">
                    <li>Rule violations highlighted row-by-row in the Violation Browser</li>
                    <li>Undo individual steps or reset the entire recipe at any time</li>
                    <li>Enable / disable rules without deleting them</li>
                </ul>
            </div>
            <div class="wsc-visual">
                <div class="wsc-rule-row"><div class="wsc-rule-dot" style="background:#a855f7;"></div>Null Check — <span style="color:#c084fc;">user_id</span> must not be null<span class="wsc-rule-badge badge-pass">PASS</span></div>
                <div class="wsc-rule-row"><div class="wsc-rule-dot" style="background:#8b5cf6;"></div>Range Check — <span style="color:#c084fc;">age</span> between 0 and 120<span class="wsc-rule-badge badge-warn">3 rows</span></div>
                <div class="wsc-rule-row"><div class="wsc-rule-dot" style="background:#6d28d9;"></div>Custom — <span style="color:#c084fc;">spend</span> &gt; 0 AND status == 'active'<span class="wsc-rule-badge badge-fail">12 rows</span></div>
                <div style="margin-top:0.5rem;font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.05em;text-transform:uppercase;">Audit Trail</div>
                <div class="wsc-audit-row"><span class="wsc-audit-step">01</span>strip_whitespace → <span class="wsc-audit-col">All columns</span></div>
                <div class="wsc-audit-row"><span class="wsc-audit-step">02</span>normalize_text → <span class="wsc-audit-col">status</span> (lowercase)</div>
                <div class="wsc-audit-row"><span class="wsc-audit-step">03</span>cast_type → <span class="wsc-audit-col">age</span> to int64</div>
            </div>
        </div>
    </section>""", unsafe_allow_html=True)
