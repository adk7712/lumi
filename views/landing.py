import streamlit as st
import urllib.parse
import os
from state_manager import load_data, MAX_SAMPLE_ROWS, calculate_health, LARGE_FILE_THRESHOLD_BYTES
from scout import generate_proposals

def render_landing_page():
    # Load template paths and assets
    TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

    with open(os.path.join(TEMPLATES_DIR, 'landing_hero.html'), 'r') as f:
        hero_html = f.read()
    st.html(hero_html)

    cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
    with cta_col:
        st.markdown('<div class="welcome-uploader-marker"></div>', unsafe_allow_html=True)
        welcome_uploader = st.file_uploader(
            "Drop your dataset here or click Browse",
            type=["csv", "xlsx"],
            key="welcome_uploader"
        )
    st.markdown('<p class="cta-helper">Free · No sign-up · Works with CSV &amp; XLSX</p>', unsafe_allow_html=True)

    if welcome_uploader:
        file_id = f"{welcome_uploader.file_id}_{welcome_uploader.name}_{welcome_uploader.size}"
        is_large = welcome_uploader.size > LARGE_FILE_THRESHOLD_BYTES
        if is_large:
            st.toast("Large file detected (>50MB). Loading first 10,000 rows for responsiveness.")
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
        bh = calculate_health(base_df)
        st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
        st.toast("Dataset Analyzed")
        st.rerun()

    with open(os.path.join(TEMPLATES_DIR, 'landing_showcase.html'), 'r') as f:
        showcase_html = f.read()
    st.html(showcase_html)

    # st.iframe runs inside a real sandboxed iframe and can access the parent
    # page's DOM via window.parent.document (same-origin). We encode the JS as
    # a data: URI since st.iframe only accepts URLs, not raw HTML.
    _js = """
    <script>
    (function() {
        var doc = window.parent.document;

        function applyDragStyle(dz) {
            dz.style.setProperty('border-style', 'dashed', 'important');
            dz.style.setProperty('border-width', '2px', 'important');
            dz.style.setProperty('border-color', 'rgba(255, 75, 75, 0.85)', 'important');
            dz.style.setProperty('background', 'rgba(255, 75, 75, 0.07)', 'important');
            var el = dz.parentElement;
            for (var i = 0; i < 6; i++) {
                if (!el) break;
                el.style.setProperty('border', 'none', 'important');
                el.style.setProperty('outline', 'none', 'important');
                el.style.setProperty('box-shadow', 'none', 'important');
                el = el.parentElement;
            }
        }

        function clearDragStyle(dz) {
            dz.style.removeProperty('border-style');
            dz.style.removeProperty('border-width');
            dz.style.removeProperty('border-color');
            dz.style.removeProperty('background');
        }

        function patchDropzone() {
            var dz = doc.querySelector('[data-testid="stFileUploaderDropzone"]');
            if (!dz) return false;
            dz.addEventListener('dragenter', function() { applyDragStyle(dz); }, true);
            dz.addEventListener('dragover',  function() { applyDragStyle(dz); }, true);
            dz.addEventListener('dragleave', function() { clearDragStyle(dz); }, true);
            dz.addEventListener('drop',      function() { clearDragStyle(dz); }, true);
            var obs = new MutationObserver(function() {
                if (dz.getAttribute('style')) applyDragStyle(dz);
            });
            obs.observe(dz, { attributes: true, attributeFilter: ['style', 'class'] });
            return true;
        }

        var interval = setInterval(function() {
            if (patchDropzone()) clearInterval(interval);
        }, 150);
    })();
    </script>
    """
    st.iframe(
        src="data:text/html;charset=utf-8," + urllib.parse.quote(_js),
        height=1
    )
