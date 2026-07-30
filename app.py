import streamlit as st
import os
import streamlit.components.v1 as components
from ui_utils import inject_custom_css, inject_posthog, is_auth_configured, get_logged_in_user, handle_signout, show_auth_dialog, show_signout_dialog, flush_pending_events
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
flush_pending_events()

# Initialize Session State
initialize_state()


# Resolve current user (used in header sign-out button)
user_email = get_logged_in_user()

# Reconciliation flow for signed-in users
if user_email:
    session_id = st.session_state.get("session_id")
    if session_id:
        from persistence import load_session, reconcile_session
        db_session = load_session(session_id, user_email)
        if db_session and db_session.get("user_id") != user_email:
            reconcile_session(session_id, user_email)
            st.toast("Workspace saved to your account!")

# Landing page (no dataset loaded yet)
if st.session_state.raw_data is None:
    render_landing_page()
    render_footer()
    st.stop()


# --- HEADER PLACEHOLDER ---
header_placeholder = st.container()

def handle_reset():
    initialize_state(from_reset=True)

def handle_undo():
    if len(st.session_state.cleaning_recipe) > 0:
        popped_step = st.session_state.cleaning_recipe.pop()
        if '_source_proposal_type' in popped_step and '_source_proposal_col' in popped_step:
            st.session_state.scanned_columns.discard(f"{popped_step['_source_proposal_col']}:{popped_step['_source_proposal_type']}")
        st.session_state.intermediate_states.pop()
        st.session_state.current_df = get_state_at_step(len(st.session_state.cleaning_recipe))
        save_session_state()

def render_header():
    with header_placeholder:
        h_col1, h_col2, h_col3 = st.columns([7, 3, 2])
        with h_col1:
            st.markdown('<div class="lumi-logo-button">', unsafe_allow_html=True)
            st.button("LUMI", key="lumi_logo", on_click=handle_reset)
            st.markdown('</div>', unsafe_allow_html=True)
        with h_col2:
            u_c1, u_c2 = st.columns(2)
            if u_c1.button("Undo", key="undo_btn", width="stretch", disabled=len(st.session_state.cleaning_recipe) == 0, on_click=handle_undo):
                st.toast("Last step undone")
            u_c2.button("Reset", key="reset_all", width="stretch", on_click=handle_reset)
        with h_col3:
            user_email = get_logged_in_user()
            if user_email:
                st.markdown(f'<div style="font-size: 0.75rem; color: #a3a3a3; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;">{user_email}</div>', unsafe_allow_html=True)
                if st.button("Sign Out", key="header_signout_btn", use_container_width=True):
                    show_signout_dialog()
            else:
                if is_auth_configured():
                    st.button(
                        "Sign In", 
                        key="header_signin_btn", 
                        use_container_width=True, 
                        on_click=st.login, 
                        kwargs={"provider": "google"}
                    )
                else:
                    if st.button("Sign In", key="header_signin_btn", use_container_width=True):
                        show_auth_dialog()

st.divider()

# --- TABS ---
TAB_NAMES = ["Overview", "Diagnostics", "Visual Insights", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"]

st.markdown("""
<script>
    const urlParams = new URLSearchParams(window.parent.location.search);
    const tabName = urlParams.get('tab');
    if (tabName) {
        const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"] p');
        tabs.forEach(tab => {
            if (tab.innerText === tabName) {
                tab.click();
            }
        });
    }
    
    // Add click listeners to all tabs to update URL
    const allTabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
    allTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const name = this.querySelector('p').innerText;
            const url = new URL(window.parent.location);
            url.searchParams.set('tab', name);
            window.parent.history.pushState({}, '', url);
        });
    });
</script>
""", unsafe_allow_html=True)

_query_tab = st.query_params.get("tab", "Overview")
if _query_tab != st.session_state.get("main_tabs", "Overview"):
    st.session_state["main_tabs"] = _query_tab if _query_tab in TAB_NAMES else "Overview"

tab_overview, tab_diagnostics, tab_insights, tab_rulebook, tab_transformations, tab_audit, tab_pipeline = st.tabs(
    TAB_NAMES,
    key="main_tabs"
)

with tab_overview:
    render_overview_tab(st.session_state.current_df)
with tab_diagnostics:
    render_diagnostics_tab(st.session_state.current_df)
with tab_insights:
    render_insights_tab(st.session_state.current_df)

with tab_rulebook:
    render_rulebook_tab(st.session_state.current_df)

with tab_transformations:
    render_transformations_tab(st.session_state.current_df)

with tab_audit:
    render_audit_log_tab()

with tab_pipeline:
    render_pipeline_preview_tab(st.session_state.current_df)

# Bottom violation browser
render_violation_browser(st.session_state.current_df)

# --- FOOTER ---
render_header()
render_footer()

