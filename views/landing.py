import streamlit as st
import urllib.parse
import os
from pathlib import Path
from state_manager import (
    calculate_file_hash,
    process_uploaded_file,
    load_session_state,
    load_db_session,
    LARGE_FILE_THRESHOLD_BYTES
)

def render_landing_page():
    # Render clean background grid and orbs
    st.html('<div class="welcome-bg welcome-grid-bg"><div class="welcome-orb welcome-orb-1"></div><div class="welcome-orb welcome-orb-2"></div><div class="welcome-orb welcome-orb-3"></div></div>')

    # Task 1: Check query parameters for an existing session recovery
    session_id = st.query_params.get("session")
    if session_id:
        from persistence import load_session
        db_session = load_session(session_id)
        if db_session:
            filename = db_session.get("filename", "dataset")
            project_name = db_session.get("project_name", "Untitled Project")
            
            st.markdown(
                f'<div style="text-align: center; margin-top: 6rem; margin-bottom: 2.5rem; position: relative; z-index: 1;">'
                f'<h1 style="font-weight: 800; font-size: 2.8rem; letter-spacing: 0.05em; color: #ffffff; margin-bottom: 0.5rem;">RESUME WORKSPACE</h1>'
                f'<p style="color: #a3a3a3; font-size: 1.05rem;">Project: <strong>{project_name}</strong></p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
            with cta_col:
                st.markdown(
                    f'<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">'
                    f'<p style="color: #e5e5e5; font-size: 0.95rem; margin-bottom: 0.5rem;">Lumi found a saved session. Please re-upload the original file to resume:</p>'
                    f'<strong style="color: #ff4b4b; font-size: 1.1rem;">{filename}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                resume_uploader = st.file_uploader(
                    "Drop the file here or browse",
                    type=["csv", "xlsx"],
                    key="resume_uploader"
                )
                
                if resume_uploader:
                    success = load_db_session(session_id, resume_uploader)
                    if success:
                        st.rerun()
                    else:
                        st.error("Failed to restore session. Please make sure you uploaded the correct file.")
                        
                st.write("")
                if st.button("Start a new project instead", key="btn_cancel_resume", use_container_width=True):
                    st.query_params.pop("session", None)
                    st.session_state.session_id = None
                    st.rerun()
            st.stop()
        else:
            # Clear invalid session parameter to avoid locking the user
            st.query_params.pop("session", None)
            st.session_state.session_id = None
            st.rerun()

    # If there is a pending local cache restore dialog
    if st.session_state.get("pending_restore_hash"):
        st.markdown(
            '<div style="text-align: center; margin-top: 6rem; margin-bottom: 2.5rem; position: relative; z-index: 1;">'
            '<h1 style="font-weight: 800; font-size: 2.8rem; letter-spacing: 0.05em; color: #ffffff; margin-bottom: 0.5rem;">LUMI</h1>'
            '<p style="color: #a3a3a3; font-size: 1.05rem;">Restore saved workspace?</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        prompt_col_l, prompt_col, prompt_col_r = st.columns([1, 2, 1])
        with prompt_col:
            st.markdown(
                '<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">'
                '<p style="color: #e5e5e5; font-size: 0.95rem; margin-bottom: 1.5rem;">Lumi found an unfinished session for this dataset. Would you like to restore your rules and cleaning steps?</p>'
                '</div>',
                unsafe_allow_html=True
            )
            
            c1, c2 = st.columns(2)
            if c1.button("Yes, Restore Session", key="btn_restore_session", use_container_width=True):
                load_session_state(st.session_state.pending_restore_hash, st.session_state.temp_uploader_file)
                st.session_state.pop("pending_restore_hash", None)
                st.session_state.pop("temp_uploader_file", None)
                st.rerun()
            if c2.button("No, Start Fresh", key="btn_discard_session", use_container_width=True):
                process_uploaded_file(st.session_state.temp_uploader_file, st.session_state.pending_restore_hash)
                # Delete cache
                cache_file = Path(".lumi_cache") / f"{st.session_state.pending_restore_hash}.json"
                if cache_file.exists():
                    try: cache_file.unlink()
                    except Exception: pass
                st.session_state.pop("pending_restore_hash", None)
                st.session_state.pop("temp_uploader_file", None)
                st.rerun()
        st.stop()

    # Render centered uploader title & description
    st.markdown(
        '<div style="text-align: center; margin-top: 6rem; margin-bottom: 2.5rem; position: relative; z-index: 1;">'
        '<h1 style="font-weight: 800; font-size: 2.8rem; letter-spacing: 0.05em; color: #ffffff; margin-bottom: 0.5rem;">LUMI</h1>'
        '<p style="color: #a3a3a3; font-size: 1.05rem;">Upload your CSV or Excel dataset to start cleaning and validation</p>'
        '</div>',
        unsafe_allow_html=True
    )

    cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
    with cta_col:
        st.markdown('<div class="welcome-uploader-marker"></div>', unsafe_allow_html=True)
        welcome_uploader = st.file_uploader(
            "Drop your dataset here or click Browse",
            type=["csv", "xlsx"],
            key="welcome_uploader"
        )
    st.markdown('<p class="cta-helper" style="position: relative; z-index: 1;">Free · No sign-up · Works with CSV &amp; XLSX</p>', unsafe_allow_html=True)

    if welcome_uploader:
        file_hash = calculate_file_hash(welcome_uploader)
        cache_file = Path(".lumi_cache") / f"{file_hash}.json"
        
        is_testing = st.session_state.get("_is_testing", False)
        
        if cache_file.exists() and not is_testing and st.session_state.get("pending_restore_hash") != file_hash:
            st.session_state.pending_restore_hash = file_hash
            st.session_state.temp_uploader_file = welcome_uploader
            st.rerun()
        else:
            process_uploaded_file(welcome_uploader, file_hash)
            st.rerun()

    # st.iframe Dropzone mutation observer patch
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
