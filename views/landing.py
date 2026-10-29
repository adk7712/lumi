import html
import streamlit as st
import urllib.parse
import os
from pathlib import Path
from state_manager import (
    calculate_file_hash,
    process_uploaded_file,
    load_session_state,
    load_db_session,
    has_resume_cache,
    discard_resume_cache,
    LARGE_FILE_THRESHOLD_BYTES
)
from ui_utils import is_auth_configured, get_logged_in_user, is_authenticated_user, handle_signout, show_auth_dialog, show_signout_dialog, render_sign_in_button, format_timestamp, remove_cookie, SESSION_COOKIE
from persistence import count_user_sessions, get_user_projects

def recipe_fits_headers(df_header, recipe) -> list:
    """Problems replaying ``recipe`` on a header-only frame of the uploaded file ([] = it fits).

    The saved scanned columns are post-cleaning names (renames, normalisation), so they cannot be
    compared with the raw headers. Replaying the recipe on ``nrows=0`` checks the original names.
    """
    if not recipe:
        return []
    from engine import apply_recipe
    _, messages = apply_recipe(df_header, recipe)
    return [m for m in messages if "not found" in m]


def _should_process_upload(file_hash) -> bool:
    """False for the same file that just failed to load (its error stays on screen; no rerun loop).

    Any other file is a new attempt and clears the previous error.
    """
    if st.session_state.get("load_error") and st.session_state.get("_load_error_hash") == file_hash:
        return False
    st.session_state.pop("load_error", None)
    st.session_state.pop("_load_error_hash", None)
    return True


def render_iframe_dropzone_patch():
    _js = """
    <script>
    (function() {
        var doc = (window.parent && window.parent.document) ? window.parent.document : document;

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
    st.html(_js)

def render_landing_page():
    # Render clean background grid and orbs
    st.html('<div class="welcome-bg welcome-grid-bg"><div class="welcome-orb welcome-orb-1"></div><div class="welcome-orb welcome-orb-2"></div><div class="welcome-orb welcome-orb-3"></div></div>')
    # Top right login/status bar is now rendered in the main card content to be always visible
    # Primary detection: check browser cookie via controller
    # Only HMAC-signed session cookies are trusted; unsigned legacy or forged values are ignored
    # (the user simply gets a fresh guest session).
    from ui_utils import read_session_cookie
    cookie_session_id = read_session_cookie()

    # Secondary fallback: URL query parameter. It must carry the same HMAC signature as the
    # cookie; a bare session id is a bearer token anyone could paste (session fixation).
    from ui_utils import verify_cookie_value
    fallback_session_id = verify_cookie_value(st.query_params.get("session"), purpose="session")
    
    # Check if we should resume
    is_testing = st.session_state.get("_is_testing", False)
    if is_testing or is_authenticated_user():
        active_resume_id = st.session_state.get("resume_session_id")
    else:
        active_resume_id = st.session_state.get("resume_session_id") or fallback_session_id

    # 1. Recovery prompt page (active resume mode)
    if active_resume_id:
        cache_key = f"_cached_resume_{active_resume_id}"
        if cache_key not in st.session_state:
            from persistence import load_session, SessionLoadError
            try:
                st.session_state[cache_key] = load_session(active_resume_id, get_logged_in_user())
            except SessionLoadError:
                # Not the same as "expired": keep the cookie and the resume id, and do not cache.
                st.error("Could not reach the database. Your saved workspace is kept. Try again in a moment.")
                st.button("Try again", key="btn_retry_resume")
                st.stop()
        db_session = st.session_state[cache_key]
        if db_session:
            filename = db_session.get("filename", "dataset")
            project_name = db_session.get("project_name", "Untitled Project")
            # DB-derived values are user-controlled: escape before raw-HTML interpolation.
            filename_html = html.escape(str(filename or ""))
            project_name_html = html.escape(str(project_name or ""))
            saved_recipe = db_session.get("cleaning_recipe") or []

            st.markdown(
                f'<div style="text-align: center; margin-top: 6rem; margin-bottom: 2.5rem; position: relative; z-index: 1;">'
                f'<h1 style="font-weight: 800; font-size: 2.8rem; letter-spacing: 0.05em; color: #ffffff; margin-bottom: 0.5rem;">RESUME WORKSPACE</h1>'
                f'<p style="color: #a3a3a3; font-size: 1.05rem;">Project: <strong>{project_name_html}</strong></p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
            with cta_col:
                st.markdown(
                    f'<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">'
                    f'<p style="color: #e5e5e5; font-size: 0.95rem; margin-bottom: 0.5rem;">Please re-upload the original file to resume:</p>'
                    f'<strong style="color: #ff4b4b; font-size: 1.1rem;">{filename_html}</strong>'
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
                    from lumi_state.core import UnsafeFileError
                    unsafe_error = None
                    try:
                        # same reader as the upload (delimiter/encoding detection, str headers)
                        from state_manager import read_tabular
                        df_header, _ = read_tabular(resume_uploader, name=uploaded_filename, nrows=0)
                    except UnsafeFileError as e:
                        unsafe_error = e
                        df_header = None
                    except Exception:
                        df_header = None

                    mismatch = False
                    reasons = []
                    if uploaded_filename != filename:
                        mismatch = True
                        reasons.append(f"Filename mismatch: expected '{filename}', got '{uploaded_filename}'")
                    # Match on the original headers: the saved recipe must replay on them.
                    header_problems = (["The file could not be read"] if df_header is None
                                       else recipe_fits_headers(df_header, saved_recipe))
                    if header_problems:
                        mismatch = True
                        reasons.append("Column headers mismatch (the saved cleaning steps refer to columns missing in the uploaded file)")

                    if unsafe_error is not None:
                        st.error(str(unsafe_error))
                    elif mismatch:
                        st.warning("Warning: The uploaded file does not match the expected dataset.")
                        for reason in reasons:
                            st.write(f"- {reason}")
                        
                        w_c1, w_c2 = st.columns(2)
                        if w_c1.button("Proceed anyway", key="proceed_anyway_btn", use_container_width=True):
                            resume_uploader.seek(0)
                            success = load_db_session(active_resume_id, resume_uploader, preloaded_session=db_session)
                            if success:
                                st.session_state.pop("resume_session_id", None)
                                st.session_state.pop(cache_key, None)
                                st.rerun()
                            else:
                                st.error(st.session_state.get("load_error") or "Failed to restore session.")
                        if w_c2.button("Cancel upload", key="cancel_upload_btn", use_container_width=True):
                            st.rerun()
                    else:
                        resume_uploader.seek(0)
                        success = load_db_session(active_resume_id, resume_uploader, preloaded_session=db_session)
                        if success:
                            st.session_state.pop("resume_session_id", None)
                            st.session_state.pop(cache_key, None)
                            st.rerun()
                        else:
                            st.error(st.session_state.get("load_error") or "Failed to restore session.")

                st.write("")
                if st.button("Start a new project instead", key="btn_cancel_resume", use_container_width=True):
                    st.query_params.pop("session", None)
                    st.session_state.pop("resume_session_id", None)
                    remove_cookie(SESSION_COOKIE)
                    st.rerun()
            render_iframe_dropzone_patch()
            st.stop()
        else:
            # Clean invalid/expired sessions gracefully
            st.query_params.pop("session", None)
            st.session_state.pop("resume_session_id", None)
            remove_cookie(SESSION_COOKIE)
            st.rerun()

    # 2. Dismissible Session Banner (Primary Flow for anonymous guests only)
    is_testing = st.session_state.get("_is_testing", False)
    if cookie_session_id and not is_authenticated_user() and not is_testing and not st.session_state.get("cookie_session_dismissed"):
        cookie_cache_key = f"_cached_cookie_{cookie_session_id}"
        if cookie_cache_key not in st.session_state:
            from persistence import load_session, SessionLoadError
            try:
                st.session_state[cookie_cache_key] = load_session(cookie_session_id, get_logged_in_user())
            except SessionLoadError:
                pass  # DB unreachable: skip the banner this run, keep the cookie, do not cache
        db_session = st.session_state.get(cookie_cache_key)
        if db_session:
            filename = db_session.get("filename", "dataset")
            step_count = db_session.get("step_count", 0)
            
            with st.container(border=True):
                c_text, c_resume, c_fresh, c_dismiss = st.columns([5, 2, 2, 1])
                c_text.markdown(f"**Continue your last session?** ({filename}, {step_count} step{'' if step_count == 1 else 's'})")
                
                if c_resume.button("Resume", key="cookie_resume_btn", use_container_width=True):
                    st.session_state.resume_session_id = cookie_session_id
                    st.session_state[f"_cached_resume_{cookie_session_id}"] = db_session
                    st.rerun()
                if c_fresh.button("Start fresh", key="cookie_fresh_btn", use_container_width=True):
                    remove_cookie(SESSION_COOKIE)
                    st.session_state.pop("resume_session_id", None)
                    st.session_state.pop(cookie_cache_key, None)
                    st.query_params.pop("session", None)
                    st.rerun()
                if c_dismiss.button("✕", key="cookie_dismiss_btn", use_container_width=True):
                    st.session_state.cookie_session_dismissed = True
                    st.session_state.pop(cookie_cache_key, None)
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
                # Delete this owner's cache first (owner is resolved before a new session id is assigned)
                discard_resume_cache(st.session_state.pending_restore_hash)
                process_uploaded_file(st.session_state.temp_uploader_file, st.session_state.pending_restore_hash)
                st.session_state.pop("pending_restore_hash", None)
                st.session_state.pop("temp_uploader_file", None)
                st.rerun()
        st.stop()

    # Top right header with Profile / Sign In
    c_hdr_l, c_hdr_r = st.columns([4, 1.2])
    with c_hdr_r:
        if is_authenticated_user():
            u_email = get_logged_in_user()
            u_email_safe = html.escape(u_email).replace("@", "&#64;") if u_email else ""
            st.markdown('<div class="profile-popover-container">', unsafe_allow_html=True)
            with st.popover("", icon=":material/person:", help="Account Settings"):
                st.markdown('<div style="font-weight: 600; font-size: 0.8rem; color: #a3a3a3; letter-spacing: 0.05em; margin-bottom: 0.2rem;">ACCOUNT</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.95rem; font-weight: 500; color: #ffffff; margin-bottom: 0.5rem; word-break: break-all; pointer-events: none;">{u_email_safe}</div>', unsafe_allow_html=True)
                st.divider()
                if st.button("Sign Out", key="landing_popover_signout_btn", use_container_width=True):
                    show_signout_dialog()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Straight to Google in production; the dialog (Google + dev email) when dev auth is on.
            render_sign_in_button("landing_header_signin_btn")

    # Render centered uploader title & description
    st.markdown(
        '<div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem; position: relative; z-index: 1;">'
        '<h1 style="font-weight: 800; font-size: 2.8rem; letter-spacing: 0.05em; color: #ffffff; margin-bottom: 0.5rem;">LUMI</h1>'
        '<p style="color: #a3a3a3; font-size: 1.05rem;">Interactive dataset cleaning &amp; validation workflow generator</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # For authenticated users, show their active workspaces directly on the landing page
    if is_authenticated_user():
        user_email = get_logged_in_user()
        projects = get_user_projects(user_email)
        if projects:
            st.markdown(
                '<div style="max-width: 650px; margin: 0 auto 1rem auto; text-align: center;">'
                '<h3 style="font-weight: 700; color: #ffffff; margin-bottom: 0.25rem;">Your Workspaces</h3>'
                '<p style="color: #a3a3a3; font-size: 0.9rem;">Select a workspace to resume or upload a new dataset below</p>'
                '</div>',
                unsafe_allow_html=True
            )
            
            c_l, c_main, c_r = st.columns([1, 2.5, 1])
            with c_main:
                for p in projects:
                    p_id = p["session_id"]
                    p_name = p["project_name"] or p["filename"] or "Untitled Workspace"
                    is_pinned = bool(p.get("pinned", 0))
                    pin_icon = "📌 " if is_pinned else ""
                    step_text = f"{p.get('step_count', 0)} step{'' if p.get('step_count', 0) == 1 else 's'}"
                    fn_text = p.get('filename', '')
                    ts_str = format_timestamp(p.get("updated_at"))
                    caption_parts = [part for part in [fn_text, step_text, ts_str] if part]

                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1.2])
                        with col1:
                            st.markdown(f"**{pin_icon}{p_name}**")
                            st.caption(" · ".join(caption_parts))
                        with col2:
                            if st.button("Open", key=f"landing_open_ws_{p_id}", type="primary", use_container_width=True):
                                from state_manager import switch_workspace
                                switch_workspace(p_id)
            st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
            st.divider()

    cta_spacer_l, cta_col, cta_spacer_r = st.columns([1, 2, 1])
    with cta_col:
        st.markdown('<div class="welcome-uploader-marker"></div>', unsafe_allow_html=True)
        welcome_uploader = st.file_uploader(
            "Drop a new dataset here or click Browse",
            type=["csv", "xlsx"],
            key="welcome_uploader"
        )
    st.markdown('<p class="cta-helper" style="position: relative; z-index: 1;">Free · No sign-up · Works with CSV &amp; XLSX</p>', unsafe_allow_html=True)

    # Last failed upload (empty / malformed / rejected file): shown until the next upload attempt.
    if st.session_state.get("load_error"):
        _err_l, err_col, _err_r = st.columns([1, 2, 1])
        err_col.error(st.session_state.load_error)


    if welcome_uploader:
        u_email = get_logged_in_user()
        is_auth = is_authenticated_user()
        cnt = count_user_sessions(u_email) if u_email else 0
        if not is_testing and is_auth and cnt >= 5:
            st.error("Workspace limit reached (5/5). Please delete an existing workspace from the sidebar before creating a new one.")
            st.stop()
            
        file_hash = calculate_file_hash(welcome_uploader)
        if not _should_process_upload(file_hash):
            render_iframe_dropzone_patch()
            st.stop()  # same file that just failed: keep its error, do not loop through reruns

        is_testing = st.session_state.get("_is_testing", False)
        
        # Only offer a restore when *this* owner (signed session id for guests) saved progress
        # for this file; another visitor uploading the same file never sees it (V1).
        if has_resume_cache(file_hash) and not is_testing and not is_authenticated_user() and st.session_state.get("pending_restore_hash") != file_hash:
            st.session_state.pending_restore_hash = file_hash
            st.session_state.temp_uploader_file = welcome_uploader
            st.rerun()
        else:
            process_uploaded_file(welcome_uploader, file_hash)
            st.rerun()

    # st.iframe Dropzone mutation observer patch
    render_iframe_dropzone_patch()
