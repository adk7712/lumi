"""Recipe steps, rules, replay/time-travel, proposals and column-rename sync."""
import streamlit as st
import pandas as pd
from engine import apply_recipe
from scout import generate_proposals
import logging
import re

logger = logging.getLogger("lumi.state")

from lumi_state.core import calculate_health
from lumi_state.session_io import save_session_state, save_db_session, _update_project_cache

def add_rule(rule_dict: dict, at_end: bool = True):
    """Applies color/enabled status to a rule and adds it to st.session_state.rules."""
    from ui_utils import get_safe_hue, queue_event
    rule = rule_dict.copy()
    if 'enabled' not in rule:
        rule['enabled'] = True
    if 'color' not in rule:
        hue = get_safe_hue(len(st.session_state.rules))
        rule['color'] = f"hsla({hue}, 70%, 50%, 0.4)"
        
    # Clear redo stack on new action (linear invalidation)
    if 'redo_stack' in st.session_state:
        st.session_state.redo_stack = []

    if at_end:
        st.session_state.rules.append(rule)
    else:
        st.session_state.rules.insert(0, rule)
    queue_event("issue_flagged", {"type": rule.get("type"), "col": rule.get("col")})
    save_session_state()
    save_db_session()
    _update_project_cache()

def add_step(step, quiet: bool = False, persist: bool = True):
    """Adds a cleaning step to the recipe, updates the cached state, and shows a toast.

    quiet=True suppresses toasts; persist=False skips disk/DB saves (caller must persist once).
    """
    st.session_state.cleaning_recipe.append(step)
    
    # Clear redo stack on new action (linear invalidation)
    st.session_state.redo_stack = []

    # Calculate delta state from current_df and cache metadata only (no full df copy)
    new_df, messages = apply_recipe(st.session_state.current_df, [step])
    if not quiet:
        for msg in messages:
            st.toast(msg)

    th = calculate_health(new_df)
    step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
    st.session_state.intermediate_states.append((step_desc, th, len(new_df)))
    st.session_state.current_df = new_df
    
    from ui_utils import queue_event
    queue_event("cleaning_step_added", {"action": step.get("action"), "column": step.get("column")})

    if not quiet:
        st.toast(f"Step Added: {step['action']}")
    if persist:
        save_session_state()
        save_db_session()
        _update_project_cache()


def replay_steps(steps):
    """Re-applies steps quietly (no toasts, no per-step saves) then persists once."""
    for step in steps:
        add_step(step, quiet=True, persist=False)
    save_session_state()
    save_db_session()
    _update_project_cache()


def get_state_at_step(n: int) -> pd.DataFrame:
    """Reconstructs the dataframe at step N by replaying the first N recipe steps from raw_data."""
    original = st.session_state.raw_data
    if n <= 0:
        return original.copy()
    recipe_so_far = st.session_state.cleaning_recipe[:n]
    df, _ = apply_recipe(original.copy(), recipe_so_far)
    return df


def get_column_dependencies(target: str) -> list[str]:
    """Returns the descriptions of any active validation rules that depend on the specified column."""
    dependent_rules = []
    for r in st.session_state.rules:
        if r.get('col') == target or r.get('col_a') == target or r.get('col_b') == target:
            dependent_rules.append(r['desc'])
        elif r.get('type') == "Custom Expression" and target in r.get('query', ''):
            dependent_rules.append(r['desc'])
    return dependent_rules


def _name_regex(name: str):
    """Regex matching `name` as a whole token (not inside a longer identifier/word)."""
    return re.compile(r'(?<![\w])' + re.escape(name) + r'(?![\w])')


_STR_OR_BACKTICK = re.compile(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`]*`)""")


def _rename_in_expression(expr: str, target: str, new_name: str) -> str:
    """Renames a column inside a pandas query: backticked and bare names, never string literals."""
    bare_new = new_name if new_name.isidentifier() else f"`{new_name}`"
    pattern = _name_regex(target)
    out = []
    for part in _STR_OR_BACKTICK.split(expr):
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            out.append(f"`{new_name}`" if part[1:-1] == target else part)
        elif part[:1] in ("'", '"') and _STR_OR_BACKTICK.fullmatch(part):
            out.append(part)
        else:
            out.append(pattern.sub(lambda m: bare_new, part))
    return "".join(out)


def _rename_in_text(text: str, target: str, new_name: str) -> str:
    return _name_regex(target).sub(lambda m: new_name, text)


def sync_column_rename(target: str, new_name: str):
    """Synchronizes rules, scanned_columns and active features when a column is renamed.

    Structured fields (col, col_a, col_b, _source_proposal_col) are matched exactly; free text
    (desc / Custom Expression query) uses token-boundary matching so 'age' never touches 'page'.
    Recipe steps are intentionally left alone: they replay in order, so earlier steps still
    legitimately use the old name and later steps were authored against the new one.
    """
    if target == new_name:
        return
    for rule in st.session_state.rules:
        old_desc = rule.get('desc', '')
        touched = False
        for field in ('col', 'col_a', 'col_b', '_source_proposal_col'):
            if rule.get(field) == target:
                rule[field] = new_name
                touched = field != '_source_proposal_col' or touched
        if rule.get('type') == "Custom Expression":
            old_query = rule.get('query', '')
            new_query = _rename_in_expression(old_query, target, new_name) if old_query else old_query
            if new_query != old_query:
                rule['query'] = new_query
                touched = True
                if old_desc == f"Violates: {old_query}":
                    rule['desc'] = f"Violates: {new_query}"
                    continue
        if touched and old_desc:
            if rule.get('type') == "Custom Expression":
                rule['desc'] = _rename_in_expression(old_desc, target, new_name)
            else:
                rule['desc'] = _rename_in_text(old_desc, target, new_name)

    # scanned_columns holds "col:type" keys
    scanned = st.session_state.get('scanned_columns')
    if scanned:
        renamed = {f"{new_name}:{k[len(target) + 1:]}" for k in scanned if k.startswith(target + ":")}
        if renamed:
            scanned.difference_update({k for k in list(scanned) if k.startswith(target + ":")})
            scanned.update(renamed)

    # Source-proposal tags on recipe steps feed scanned_columns on clear-all
    for step in st.session_state.get('cleaning_recipe', []):
        if step.get('_source_proposal_col') == target:
            step['_source_proposal_col'] = new_name

    # Sync active features in diagnostics tab
    if target in st.session_state.active_features:
        idx = st.session_state.active_features.index(target)
        st.session_state.active_features[idx] = new_name


def regenerate_proposals():
    """Helper to recalculate proposals when active rules/steps are removed."""
    if 'raw_data' in st.session_state and st.session_state.raw_data is not None:
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
