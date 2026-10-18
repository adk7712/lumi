import streamlit as st
from ui_utils import is_authenticated_user, get_logged_in_user, format_timestamp
from persistence import (
    get_user_projects,
    count_user_sessions,
    delete_session,
    rename_session,
    toggle_pin_session
)
from state_manager import (
    switch_workspace,
    initialize_state,
    delete_cached_file
)

@st.dialog("New Workspace")
def show_new_workspace_dialog():
    st.write("Create a new dataset cleaning workspace.")
    ws_name = st.text_input("Workspace Name", value="Untitled Workspace", key="new_ws_name_input")
    col1, col2 = st.columns(2)
    if col1.button("Create", type="primary", use_container_width=True, key="confirm_create_ws"):
        st.session_state._pending_workspace_name = ws_name
        initialize_state(from_reset=True)
        st.rerun()
    if col2.button("Cancel", use_container_width=True, key="cancel_create_ws"):
        st.rerun()

@st.dialog("Rename Workspace")
def show_rename_workspace_dialog(session_id: str, current_name: str):
    st.write(f"Rename **{current_name}**")
    new_name = st.text_input("New Name", value=current_name, key=f"rename_input_{session_id}")
    col1, col2 = st.columns(2)
    if col1.button("Save", type="primary", use_container_width=True, key=f"confirm_rename_{session_id}"):
        user_email = get_logged_in_user()
        rename_session(session_id, new_name, user_email)
        if st.session_state.get("session_id") == session_id:
            st.session_state.project_name = new_name
        st.rerun()
    if col2.button("Cancel", use_container_width=True, key=f"cancel_rename_{session_id}"):
        st.rerun()

@st.dialog("Delete Workspace")
def show_delete_workspace_dialog(session_id: str, project_name: str):
    st.warning(f"Are you sure you want to delete **{project_name}**? This cannot be undone.")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True, key=f"confirm_delete_{session_id}"):
        user_email = get_logged_in_user()
        delete_session(session_id, user_email)
        delete_cached_file(session_id)
        if st.session_state.get("session_id") == session_id:
            initialize_state(from_reset=True)
        st.toast(f"Deleted {project_name}")
        st.rerun()
    if col2.button("Cancel", use_container_width=True, key=f"cancel_delete_{session_id}"):
        st.rerun()

def render_workspace_sidebar():
    """Renders the workspace sidebar for authenticated users."""
    if not is_authenticated_user():
        return

    user_email = get_logged_in_user()
    projects = get_user_projects(user_email)
    active_session_id = st.session_state.get("session_id")

    with st.sidebar:
        st.markdown(
            '<div style="font-weight: 700; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem;">Workspaces</div>',
            unsafe_allow_html=True
        )

        at_limit = len(projects) >= 5
        if st.button("+ New Workspace", key="sidebar_new_ws_btn", use_container_width=True, disabled=at_limit):
            show_new_workspace_dialog()

        if at_limit:
            st.caption("Workspace limit reached (5/5)")

        st.divider()

        if not projects:
            st.caption("No workspaces yet. Upload a dataset to start!")
            return

        for p in projects:
            p_id = p["session_id"]
            p_name = p["project_name"] or p["filename"] or "Untitled"
            is_active = (p_id == active_session_id)
            is_pinned = bool(p.get("pinned", 0))

            pin_icon = "📌 " if is_pinned else ""
            display_title = f"{pin_icon}{p_name}"

            # Layout for workspace card: [Button / Title (active highlight)] [Options Popover]
            col_main, col_menu = st.columns([5, 1])

            with col_main:
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    display_title,
                    key=f"ws_select_{p_id}",
                    type=btn_type,
                    use_container_width=True,
                    disabled=is_active
                ):
                    switch_workspace(p_id)

            with col_menu:
                with st.popover("⋮", key=f"ws_menu_{p_id}", use_container_width=True):
                    if st.button("Rename", key=f"menu_rename_{p_id}", use_container_width=True):
                        show_rename_workspace_dialog(p_id, p_name)
                    
                    pin_label = "Unpin" if is_pinned else "Pin to top"
                    if st.button(pin_label, key=f"menu_pin_{p_id}", use_container_width=True):
                        toggle_pin_session(p_id, user_email)
                        st.rerun()
                        
                    if st.button("Delete", key=f"menu_del_{p_id}", use_container_width=True, type="primary"):
                        show_delete_workspace_dialog(p_id, p_name)

            ts_str = format_timestamp(p.get("updated_at"))
            fn_text = p.get("filename", "")
            step_text = f"{p.get('step_count', 0)} steps"
            caption_parts = [part for part in [fn_text, step_text, ts_str] if part]
            st.caption(" · ".join(caption_parts))
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
