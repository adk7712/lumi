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
from ui_utils import is_auth_configured, get_logged_in_user, handle_signout, show_auth_dialog

def render_iframe_dropzone_patch():
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

def render_landing_page():
    # Render clean background grid and orbs
    st.html('<div class="welcome-bg welcome-grid-bg"><div class="welcome-orb welcome-orb-1"></div><div class="welcome-orb welcome-orb-2"></div><div class="welcome-orb welcome-orb-3"></div></div>')
    # Top right login/status bar is now rendered in the main card content to be always visible
    # Primary detection: check browser cookie via controller
    cookie_session_id = None
    try:
        from streamlit_cookies_controller import CookieController
        controller = CookieController()
        cookie_session_id = controller.get("lumi_session")
    except Exception:
        pass

    # Secondary fallback: URL query parameter
    fallback_session_id = st.query_params.get("session")
    
    # Check if we should resume
    active_resume_id = st.session_state.get("resume_session_id") or fallback_session_id

    # 1. Recovery prompt page (active resume mode)
    if active_resume_id:
        from persistence import load_session
        db_session = load_session(active_resume_id)
        if db_session:
            filename = db_session.get("filename", "dataset")
            project_name = db_session.get("project_name", "Untitled Project")
            expected_columns = db_session.get("scanned_columns", set())
            
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
                    f'<p style="color: #e5e5e5; font-size: 0.95rem; margin-bottom: 0.5rem;">Please re-upload the original file to resume:</p>'
                    f'<strong style="color: #ff4b4b; font-size: 1.1rem;">{filename}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown('<div class="welcome-uploader-marker"></div>', unsafe_allow_html=True)
                resume_uploader = st.file_uploader(
                    "Drop the file here or browse",
                    type=["csv", "xlsx"],
                    key="resume_uploader"
                )
                
                if resume_uploader:
                    # Mismatch detection (Requirement 7)
                    uploaded_filename = resume_uploader.name
                    try:
                        import pandas as pd
                        if uploaded_filename.endswith(".csv"):
                            df_header = pd.read_csv(resume_uploader, nrows=0)
                        else:
                            df_header = pd.read_excel(resume_uploader, nrows=0)
                        uploaded_columns = set(df_header.columns)
                    except Exception:
                        uploaded_columns = set()

                    mismatch = False
                    reasons = []
                    if uploaded_filename != filename:
                        mismatch = True
                        reasons.append(f"Filename mismatch: expected '{filename}', got '{uploaded_filename}'")
                    if expected_columns and not expected_columns.issubset(uploaded_columns):
                        mismatch = True
                        reasons.append("Column headers mismatch (some expected columns are missing in the uploaded file)")

                    if mismatch:
                        st.warning("Warning: The uploaded file does not match the expected dataset.")
                        for reason in reasons:
                            st.write(f"- {reason}")
                        
                        w_c1, w_c2 = st.columns(2)
                        if w_c1.button("Proceed anyway", key="proceed_anyway_btn", use_container_width=True):
                            success = load_db_session(active_resume_id, resume_uploader)
                            if success:
                                st.session_state.pop("resume_session_id", None)
                                st.rerun()
                            else:
                                st.error("Failed to restore session.")
                        if w_c2.button("Cancel upload", key="cancel_upload_btn", use_container_width=True):
                            st.rerun()
                    else:
                        success = load_db_session(active_resume_id, resume_uploader)
                        if success:
                            st.session_state.pop("resume_session_id", None)
                            st.rerun()
                        else:
                            st.error("Failed to restore session.")

                st.write("")
                if st.button("Start a new project instead", key="btn_cancel_resume", use_container_width=True):
                    st.query_params.pop("session", None)
                    st.session_state.pop("resume_session_id", None)
                    try:
                        from streamlit_cookies_controller import CookieController
                        controller = CookieController()
                        controller.remove("lumi_session")
                    except Exception:
                        pass
                    st.rerun()
            render_iframe_dropzone_patch()
            st.stop()
        else:
            # Clean invalid/expired sessions gracefully
            st.query_params.pop("session", None)
            st.session_state.pop("resume_session_id", None)
            try:
                from streamlit_cookies_controller import CookieController
                controller = CookieController()
                controller.remove("lumi_session")
            except Exception:
                pass
            st.rerun()

    # 2. Dismissible Session Banner (Primary Flow)
    if cookie_session_id and not st.session_state.get("cookie_session_dismissed"):
        from persistence import load_session
        db_session = load_session(cookie_session_id)
        if db_session:
            filename = db_session.get("filename", "dataset")
            step_count = db_session.get("step_count", 0)
            
            with st.container(border=True):
                c_text, c_resume, c_fresh, c_dismiss = st.columns([5, 2, 2, 1])
                c_text.markdown(f"**Continue your last session?** ({filename}, {step_count} steps)")
                
                if c_resume.button("Resume", key="cookie_resume_btn", use_container_width=True):
                    st.session_state.resume_session_id = cookie_session_id
                    st.rerun()
                if c_fresh.button("Start fresh", key="cookie_fresh_btn", use_container_width=True):
                    try:
                        from streamlit_cookies_controller import CookieController
                        controller = CookieController()
                        controller.remove("lumi_session")
                    except Exception:
                        pass
                    st.session_state.pop("resume_session_id", None)
                    st.query_params.pop("session", None)
                    st.rerun()
                if c_dismiss.button("✕", key="cookie_dismiss_btn", use_container_width=True):
                    st.session_state.cookie_session_dismissed = True
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

    c_login_l, c_login, c_login_r = st.columns([1.5, 1, 1.5])
    with c_login:
        user_email = get_logged_in_user()
        if user_email:
            st.markdown(f'<div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: #a3a3a3;">Logged in as <strong>{user_email}</strong></div>', unsafe_allow_html=True)
            if st.button("Sign Out", key="landing_signout_btn", use_container_width=True):
                handle_signout()
        else:
            if st.button("Sign In to Sync Projects", key="landing_signin_btn", use_container_width=True):
                if is_auth_configured():
                    try:
                        st.login()
                        st.rerun()
                    except Exception:
                        show_auth_dialog()
                else:
                    show_auth_dialog()


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
    render_iframe_dropzone_patch()
