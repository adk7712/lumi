import streamlit as st
import os
from state_manager import load_data, MAX_SAMPLE_ROWS
from scout import generate_proposals

# Load template paths
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
with open(os.path.join(TEMPLATES_DIR, 'landing_hero.html'), 'r') as f:
    HERO_HTML = f.read()

with open(os.path.join(TEMPLATES_DIR, 'landing_showcase.html'), 'r') as f:
    SHOWCASE_HTML = f.read()
def render_landing_page():
    st.html(HERO_HTML)

    # 1. Handle Custom Button Query Param Action
    if st.query_params.get("action") == "upload_dataset":
        st.session_state.show_uploader = True
        st.query_params.clear()
        st.rerun()

    # Render CTA button or Uploader
    if not st.session_state.show_uploader:
        cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
        with cta_col:
            custom_button_html = """
            <div style="text-align: center; margin: 10px 0;">
                <a href="./?action=upload_dataset" target="_self" class="premium-cta-btn">
                    <strong>Upload</strong> Your Dataset →
                </a>
            </div>
            """
            st.html(custom_button_html)
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

    st.html(SHOWCASE_HTML)
