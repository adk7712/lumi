import streamlit as st
from state_manager import replay_steps, get_state_at_step, unresolve_broken_rules, revert_step_renames

def render_audit_log_tab():
    if not st.session_state.cleaning_recipe:
        st.caption("No transformations applied yet")
    else:
        for i, (desc, health, rows) in enumerate(st.session_state.intermediate_states):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f'<div class="recipe-step"><strong>{i + 1}. {desc}</strong> | Health: {health}% | Rows: {rows:,}</div>', unsafe_allow_html=True)

            if i > 0:
                if c2.button("Remove", key=f"rm_step_{i}", width="stretch"):
                    removed_step = st.session_state.cleaning_recipe.pop(i-1)
                    remaining_recipe = st.session_state.cleaning_recipe[i-1:]
                    st.session_state.cleaning_recipe = st.session_state.cleaning_recipe[:i-1]
                    st.session_state.intermediate_states = st.session_state.intermediate_states[:i]
                    # Reset current_df to the correct pre-removal state before replaying
                    # downstream steps, so add_step records accurate row counts.
                    st.session_state.current_df = get_state_at_step(len(st.session_state.cleaning_recipe))
                    revert_step_renames(removed_step, st.session_state.current_df)
                    replay_steps(remaining_recipe)
                    unresolve_broken_rules()
                    st.session_state.redo_stack = []
                    st.session_state._needs_rerun = True
