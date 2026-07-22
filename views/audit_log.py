import streamlit as st
from state_manager import add_step, get_state_at_step, save_session_state

def render_audit_log_tab():
    session_id = st.session_state.get("session_id")
    if session_id and len(st.session_state.cleaning_recipe) >= 1:
        st.markdown('<h3 style="font-weight: 600; font-size: 1.1rem; color: #ffffff;">Share Workspace</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #a3a3a3; font-size: 0.85rem; margin-bottom: 0.5rem;">Copy this link to resume your cleaning workflow on another device or browser.</p>', unsafe_allow_html=True)
        
        # HTML Clipboard copy button
        copy_html = f"""
        <button id="copy-resume-btn" onclick="
            const url = window.location.origin + window.location.pathname + '?session={session_id}';
            navigator.clipboard.writeText(url).then(() => {{
                const btn = document.getElementById('copy-resume-btn');
                btn.innerText = 'Copied!';
                btn.style.backgroundColor = '#27ae60';
                setTimeout(() => {{
                    btn.innerText = 'Copy Shareable Link';
                    btn.style.backgroundColor = '#ff4b4b';
                }}, 2000);
            }});
        " style="
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', Courier, monospace;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            width: auto;
            margin-bottom: 1.5rem;
            transition: background-color 0.2s ease;
        ">Copy Shareable Link</button>
        """
        st.components.v1.html(copy_html, height=45)
        st.write("---")

    if not st.session_state.cleaning_recipe:
        st.caption("No transformations applied yet")
    else:
        for i, (desc, health, rows) in enumerate(st.session_state.intermediate_states):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f'<div class="recipe-step"><strong>{i + 1}. {desc}</strong> | Health: {health}% | Rows: {rows:,}</div>', unsafe_allow_html=True)

            if i > 0:
                if c2.button("Remove", key=f"rm_step_{i}", width="stretch"):
                    st.session_state.cleaning_recipe.pop(i-1)
                    remaining_recipe = st.session_state.cleaning_recipe[i-1:]
                    st.session_state.cleaning_recipe = st.session_state.cleaning_recipe[:i-1]
                    st.session_state.intermediate_states = st.session_state.intermediate_states[:i]
                    for r_step in remaining_recipe:
                        add_step(r_step)
                    st.session_state.current_df = get_state_at_step(len(st.session_state.cleaning_recipe))
                    save_session_state()
                    st.rerun()
