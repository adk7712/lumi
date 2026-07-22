import streamlit as st
import os
from ui_utils import inject_custom_css, inject_posthog
from state_manager import initialize_state, load_data, MAX_SAMPLE_ROWS, get_state_at_step, save_session_state
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


# --- Footer Helper ---
_FOOTER_HTML = None
def render_footer():
    global _FOOTER_HTML
    if _FOOTER_HTML is None:
        footer_path = os.path.join(os.path.dirname(__file__), 'views', 'templates', 'footer.html')
        with open(footer_path, 'r') as f:
            _FOOTER_HTML = f.read()
    st.html(_FOOTER_HTML)

# Set page config
st.set_page_config(
    page_title="Lumi",
    page_icon="assets/lumi_logo_white.svg",
    layout="wide"
)

# Inject Custom CSS
inject_custom_css(st)

# Inject PostHog Analytics
inject_posthog(st)

# Initialize Session State
initialize_state()

# --- SIDEBAR AUTH & RECENT PROJECTS ---
with st.sidebar:
    st.markdown('<h2 style="font-weight: 700; font-size: 1.5rem; color: #ffffff;">Lumi Workspace</h2>', unsafe_allow_html=True)
    st.write("---")
    
    # Optional login via OIDC
    user_email = None
    try:
        if st.experimental_user and st.experimental_user.get("email"):
            user_email = st.experimental_user.get("email")
    except Exception:
        pass
        
    if user_email:
        st.markdown(f"Signed in as:<br/><strong>{user_email}</strong>", unsafe_allow_html=True)
        if st.button("Sign Out", key="auth_signout_btn", use_container_width=True):
            st.logout()
            st.rerun()
            
        # Task 2: Reconciliation flow - if there was an active anonymous session
        session_id = st.session_state.get("session_id")
        if session_id:
            from persistence import load_session, reconcile_session
            db_session = load_session(session_id)
            if db_session and db_session.get("user_id") != user_email:
                reconcile_session(session_id, user_email)
                st.toast("Workspace saved to your account!")
                
        # List recent projects
        st.write("---")
        st.markdown('<h3 style="font-weight: 600; font-size: 1.1rem; color: #ffffff;">Recent Projects</h3>', unsafe_allow_html=True)
        from persistence import get_user_projects
        projects = get_user_projects(user_email)
        if projects:
            for proj in projects:
                p_name = proj["project_name"] or f"Project ({proj['session_id'][:8]})"
                if st.button(p_name, key=f"side_proj_{proj['session_id']}", use_container_width=True):
                    st.query_params["session"] = proj["session_id"]
                    # Reset data to force the landing page to load the selected project
                    st.session_state.raw_data = None
                    st.session_state.session_id = proj["session_id"]
                    st.rerun()
        else:
            st.caption("No saved projects yet.")
    else:
        st.markdown('<p style="color: #a3a3a3; font-size: 0.9rem;">Sign in to sync your cleaning projects across devices.</p>', unsafe_allow_html=True)
        if st.button("Sign In", key="auth_signin_btn", use_container_width=True):
            st.login()
            st.rerun()

    # Share / Resume link section (only after the first step)
    if st.session_state.raw_data is not None and len(st.session_state.get("cleaning_recipe", [])) >= 1:
        session_id = st.session_state.get("session_id")
        if session_id:
            st.write("---")
            st.markdown('<h3 style="font-weight: 600; font-size: 1.1rem; color: #ffffff;">Resume Session</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #a3a3a3; font-size: 0.85rem;">Copy this link to resume your cleaning workflow later.</p>', unsafe_allow_html=True)
            
            # HTML Clipboard copy button
            copy_html = f"""
            <button id="copy-resume-btn" onclick="
                const url = window.location.origin + window.location.pathname + '?session={session_id}';
                navigator.clipboard.writeText(url).then(() => {{
                    const btn = document.getElementById('copy-resume-btn');
                    btn.innerText = 'Copied!';
                    btn.style.backgroundColor = '#27ae60';
                    setTimeout(() => {{
                        btn.innerText = 'Copy Resume Link';
                        btn.style.backgroundColor = '#ff4b4b';
                    }}, 2000);
                }});
            " style="
                background-color: #ff4b4b;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                font-family: 'JetBrains Mono', Courier, monospace;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                box-sizing: border-box;
                transition: background-color 0.2s ease;
            ">Copy Resume Link</button>
            """
            st.components.v1.html(copy_html, height=45)

# Welcome view if no data is loaded yet
if st.session_state.raw_data is None:
    render_landing_page()
    render_footer()
    st.stop()

# --- HEADER (only shown after a dataset is loaded) ---
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
        st.session_state.current_df = get_state_at_step(len(st.session_state.cleaning_recipe))
        save_session_state()
        st.toast("Last step undone")
        st.rerun()
    if u_c2.button("Reset", key="reset_all", width="stretch"):
        initialize_state(from_reset=True)
        st.rerun()

st.divider()

# Get the latest dataframe from cache
df = st.session_state.current_df

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

# --- FOOTER ---
render_footer()

