import streamlit as st
import os
from ui_utils import inject_custom_css, inject_posthog, is_auth_configured, get_logged_in_user, handle_signout, show_auth_dialog
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


# Resolve current user (used in header sign-out button)
user_email = get_logged_in_user()

# Reconciliation flow for signed-in users
if user_email:
    session_id = st.session_state.get("session_id")
    if session_id:
        from persistence import load_session, reconcile_session
        db_session = load_session(session_id)
        if db_session and db_session.get("user_id") != user_email:
            reconcile_session(session_id, user_email)
            st.toast("Workspace saved to your account!")

# Landing page (no dataset loaded yet)
if st.session_state.raw_data is None:
    render_landing_page()
    render_footer()
    st.stop()



# --- HEADER (only shown after a dataset is loaded) ---
h_col1, h_col2, h_col3 = st.columns([7, 3, 2])
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
with h_col3:
    user_email = get_logged_in_user()
    if user_email:
        st.markdown(f'<div style="font-size: 0.75rem; color: #a3a3a3; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;">{user_email}</div>', unsafe_allow_html=True)
        if st.button("Sign Out", key="header_signout_btn", use_container_width=True):
            handle_signout()
    else:
        if st.button("Sign In", key="header_signin_btn", use_container_width=True):
            if is_auth_configured():
                try:
                    st.login()
                    st.rerun()
                except Exception:
                    show_auth_dialog()
            else:
                show_auth_dialog()

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

