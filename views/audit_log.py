import html
import streamlit as st
from state_manager import replay_steps, get_state_at_step, unresolve_broken_rules, revert_step_renames, save_session_state, save_db_session, _update_project_cache
from lumi_state.steps import step_renames, get_column_dependencies

def _step_refs(step):
    """Column names a recipe step reads (single column, new_column excluded, reorder lists)."""
    refs = set()
    col = step.get('column')
    if isinstance(col, str):
        refs.add(col)
    if step.get('action') == 'reorder_columns' and isinstance(step.get('value'), list):
        refs.update(c for c in step['value'] if isinstance(c, str))
    return refs


def _removal_blockers(i):
    """Describe later steps/rules that use a column name created by recipe step i-1 (else [])."""
    ss = st.session_state
    created = set(step_renames(ss.cleaning_recipe[i - 1], get_state_at_step(i - 1)).values())
    if not created:
        return []
    blockers = []
    # Numbered like the visible list, which counts "Original Data" as 1: recipe index k is shown as k + 2.
    for n, later in enumerate(ss.cleaning_recipe[i:], start=i + 2):
        used = _step_refs(later) & created
        if used:
            blockers.append(f"step {n} ({later.get('action')}) uses {', '.join(sorted(used))}")
    for name in sorted(created):
        rules = get_column_dependencies(name)
        if rules:
            blockers.append(f"rule(s) {', '.join(rules)} use {name}")
    return blockers


def _remove_step(i):
    """Remove recipe step i-1 and replay the rest; restore the previous state on any failure."""
    ss = st.session_state
    try:
        blockers = _removal_blockers(i)
    except Exception:
        blockers = []  # the guarded removal below reports and rolls back any real failure
    if blockers:
        st.error(f"Cannot remove step {i + 1}: it renames columns that later work depends on ("
                 + "; ".join(blockers) + "). Use Undo to step back instead, or remove the dependent steps first.")
        return
    snapshot = {
        "recipe": list(ss.cleaning_recipe),
        "intermediate_states": list(ss.intermediate_states),
        "current_df": ss.current_df,
        "redo_stack": list(ss.get("redo_stack", [])),
        "rules": [dict(r) for r in ss.rules],
    }
    try:
        removed_step = ss.cleaning_recipe.pop(i-1)
        remaining_recipe = ss.cleaning_recipe[i-1:]
        ss.cleaning_recipe = ss.cleaning_recipe[:i-1]
        ss.intermediate_states = ss.intermediate_states[:i]
        # Reset current_df to the correct pre-removal state before replaying
        # downstream steps, so add_step records accurate row counts.
        ss.current_df = get_state_at_step(len(ss.cleaning_recipe))
        revert_step_renames(removed_step, ss.current_df)
        # Replay without saving: adjust rule flags first, then save once.
        replay_steps(remaining_recipe, persist=False)
        unresolve_broken_rules()
        ss.redo_stack = []
        save_session_state()
        save_db_session()
        _update_project_cache()
        ss._needs_rerun = True
    except Exception as e:
        ss.cleaning_recipe = snapshot["recipe"]
        ss.intermediate_states = snapshot["intermediate_states"]
        ss.current_df = snapshot["current_df"]
        ss.redo_stack = snapshot["redo_stack"]
        ss.rules = snapshot["rules"]
        st.error(f"Could not remove step {i + 1}: {e}. The previous state was restored.")

def render_audit_log_tab():
    if not st.session_state.cleaning_recipe:
        st.caption("No transformations applied yet")
    else:
        for i, (desc, health, rows) in enumerate(st.session_state.intermediate_states):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f'<div class="recipe-step"><strong>{i + 1}. {html.escape(str(desc))}</strong> | Health: {html.escape(str(health))}% | Rows: {rows:,}</div>', unsafe_allow_html=True)

            if i > 0:
                if c2.button("Remove", key=f"rm_step_{i}", width="stretch"):
                    _remove_step(i)
