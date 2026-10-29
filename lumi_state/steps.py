"""Recipe steps, rules, replay/time-travel, proposals and column-rename sync."""
import streamlit as st
import pandas as pd
from engine import apply_recipe
from scout import generate_proposals
import hashlib
import json
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

def step_renames(step, df_before) -> dict:
    """``{old: new}`` column names that ``step`` renames when applied to ``df_before``."""
    action = step.get('action')
    if action == 'rename_column':
        col, new = step.get('column'), step.get('value')
        return {col: new} if df_before is not None and col in df_before.columns and new and new != col else {}
    if action == 'normalize_column_names' and df_before is not None:
        from engine_ops import predict_column_renames
        return predict_column_renames(df_before.columns.tolist(), step.get('value', 'snake_case'), only_changed=True)
    return {}


def revert_step_renames(step, df_before):
    """Point rules/proposals back at the pre-step names after ``step`` is undone or removed."""
    for old, new in step_renames(step, df_before).items():
        sync_column_rename(new, old)


def _is_problem(msg: str) -> bool:
    """True for a skip/warning/error message (vs. an informational one like "Removed 3 rows")."""
    return str(msg).startswith(("Warning", "Error", "An unexpected error")) or "skipped" in str(msg)


def _is_noop(before, after) -> bool:
    """True when ``after`` has the same columns, dtypes and values as ``before``."""
    try:
        return before.equals(after) and list(before.columns) == list(after.columns)
    except Exception:
        return False


def add_step(step, quiet: bool = False, persist: bool = True, clear_redo: bool = True,
             sync_renames: bool = True):
    """Adds a cleaning step to the recipe, updates the cached state, and shows a toast.

    Returns False (and records nothing) when an interactive step changed no data and only
    produced skip/error messages; those are kept in ``_step_warnings`` for the Transformations tab.

    quiet=True suppresses toasts; persist=False skips disk/DB saves (caller must persist once).
    clear_redo=False (used by redo) keeps the remaining redo stack; new actions clear it.
    sync_renames=True moves rules/proposals onto renamed columns before the save. Replays pass
    False: their rules already use the final names, and re-syncing could chain renames wrongly.
    """
    renames = step_renames(step, st.session_state.current_df) if sync_renames else {}

    # Calculate delta state from current_df and cache metadata only (no full df copy)
    new_df, messages = apply_recipe(st.session_state.current_df, [step])
    if not quiet:
        # Persistent (rendered in the Transformations tab until the next action); a toast alone
        # vanished before the user could read why nothing changed.
        problems = [m for m in messages if _is_problem(m)]
        st.session_state['_step_warnings'] = problems
        for msg in messages:
            st.toast(msg)
        if problems and _is_noop(st.session_state.current_df, new_df):
            # Only effect was a skip/error: recording it would add a no-op step to the recipe,
            # audit log and exported pipeline (and wipe the redo stack for nothing).
            return False

    st.session_state.cleaning_recipe.append(step)

    # Clear redo stack on new action (linear invalidation); redo itself preserves it
    if clear_redo:
        st.session_state.redo_stack = []

    th = calculate_health(new_df)
    step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
    st.session_state.intermediate_states.append((step_desc, th, len(new_df)))
    st.session_state.current_df = new_df
    for old, new in renames.items():
        sync_column_rename(old, new)

    from ui_utils import queue_event
    queue_event("cleaning_step_added", {"action": step.get("action"), "column": step.get("column")})

    if not quiet:
        st.toast(f"Step Added: {step['action']}")
    if persist:
        save_session_state()
        save_db_session()
        _update_project_cache()
    return True


def replay_steps(steps, persist: bool = True):
    """Re-applies steps quietly (no toasts, no per-step saves) then persists once.

    persist=False skips that save: the caller adjusts rules first and saves once itself.
    """
    for step in steps:
        add_step(step, quiet=True, persist=False, sync_renames=False)
    if persist:
        save_session_state()
        save_db_session()
        _update_project_cache()


def persist_all():
    """Save session state, DB session and project cache once."""
    save_session_state()
    save_db_session()
    _update_project_cache()


def _json_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def recipe_hash() -> str:
    return _json_hash(st.session_state.get('cleaning_recipe', []))


def rules_hash() -> str:
    """Hash of rule content incl. enabled flags; ignores derived `error`/`resolved` fields."""
    rules = [{k: v for k, v in r.items() if k not in ('error', 'resolved')}
             for r in st.session_state.get('rules', [])]
    return _json_hash(rules)


def content_signature(df=None) -> str:
    """Signature of everything rule evaluation and metrics depend on (not object identity)."""
    shape = (tuple(df.shape), tuple(map(str, df.columns))) if df is not None else None
    return _json_hash([
        recipe_hash(), rules_hash(), st.session_state.get('last_file_hash'),
        st.session_state.get('session_id'), shape,
    ])


def rule_uids(rules=None) -> list:
    """Stable per-rule widget ids (hash of type/columns/desc, de-duplicated), independent of list position."""
    rules = st.session_state.get('rules', []) if rules is None else rules
    seen, out = {}, []
    for r in rules:
        base = hashlib.sha1("|".join(str(r.get(k, '')) for k in ('type', 'col', 'col_a', 'col_b', 'desc')).encode("utf-8")).hexdigest()[:10]
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}_{n}")
    return out


def get_rule_masks(df) -> dict:
    """Violation mask (or the raised exception) per enabled, non-Informational rule index.

    Computed once per content signature and shared by the overview, rulebook and violation
    browser; only the latest entry is kept. Callers must not mutate the returned masks.
    """
    from rule_utils import evaluate_rule
    sig = content_signature(df)
    cached = st.session_state.get('_rule_masks_cache')
    if cached and cached[0] == sig:
        return cached[1]
    masks = {}
    for idx, rule in enumerate(st.session_state.get('rules', [])):
        if not rule.get('enabled', True) or rule.get('type') == "Informational":
            continue
        try:
            masks[idx] = evaluate_rule(df, rule)
        except Exception as e:
            masks[idx] = e
    st.session_state['_rule_masks_cache'] = (sig, masks)
    return masks


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

    # Pending proposals were generated against the old names (Accept All applies them in order)
    for p in st.session_state.get('proposals', []) or []:
        if p.get('column') == target:
            p['column'] = new_name
        data = p.get('rule_data') or {}
        touched = False
        for field in ('col', 'col_a', 'col_b', 'column'):
            if data.get(field) == target:
                data[field] = new_name
                touched = True
        if touched and data.get('desc'):
            data['desc'] = _rename_in_text(data['desc'], target, new_name)

    # Sync active features in diagnostics tab
    if target in st.session_state.active_features:
        idx = st.session_state.active_features.index(target)
        st.session_state.active_features[idx] = new_name


def unresolve_broken_rules() -> list:
    """Clear `resolved` on rules whose violations are back in current_df.

    Returns ``(index, desc)`` of each rule changed, so a redo can restore the flag.
    """
    from rule_utils import evaluate_rule
    changed = []
    df = st.session_state.get('current_df')
    for i, rule in enumerate(st.session_state.get('rules', [])):
        if not rule.get('resolved') or df is None:
            continue
        try:
            if evaluate_rule(df, rule).sum() > 0:
                rule['resolved'] = False
                changed.append((i, rule.get('desc')))
        except Exception:
            logger.debug("Could not re-evaluate rule %s", rule.get('desc'), exc_info=True)
    return changed


def reresolve_rules(changed: list):
    """Restore `resolved` for rules from ``unresolve_broken_rules`` that are clean again."""
    from rule_utils import evaluate_rule
    rules = st.session_state.get('rules', [])
    df = st.session_state.get('current_df')
    for i, desc in changed or []:
        if i < len(rules) and rules[i].get('desc') == desc and df is not None:
            try:
                if evaluate_rule(df, rules[i]).sum() == 0:
                    rules[i]['resolved'] = True
            except Exception:
                logger.debug("Could not re-evaluate rule %s", desc, exc_info=True)


_PROPOSAL_COL_FIELDS = ('col', 'col_a', 'col_b', 'column')


def _proposal_is_current(p, columns, rule_keys, step_keys) -> bool:
    """True when proposal ``p`` targets existing columns and is not already a rule or a step."""
    from scout import DATASET_KEY
    data = p.get('rule_data') or {}
    col = p.get('column')
    if col != DATASET_KEY and col not in columns:
        return False
    for field in _PROPOSAL_COL_FIELDS:
        ref = data.get(field)
        if isinstance(ref, str) and ref != DATASET_KEY and ref not in columns:
            return False
    if 'action' in data:
        return (data.get('action'), data.get('column')) not in step_keys
    return (data.get('type'), data.get('col'), data.get('col_a'), data.get('col_b')) not in rule_keys


def filter_current_proposals(proposals, df) -> list:
    """Drop proposals on columns ``df`` no longer has, and ones already covered by an
    active rule or an applied recipe step (e.g. after Undo/Redo or a rename)."""
    ss = st.session_state
    columns = set(df.columns) if df is not None else set()
    rule_keys = {(r.get('type'), r.get('col'), r.get('col_a'), r.get('col_b'))
                 for r in ss.get('rules', []) or []}
    step_keys = {(s.get('action'), s.get('column')) for s in ss.get('cleaning_recipe', []) or []}
    return [p for p in proposals or [] if _proposal_is_current(p, columns, rule_keys, step_keys)]


def regenerate_proposals():
    """Recalculate proposals against the current (post-recipe) data after rules/steps change.

    Scanning raw_data proposed rules on pre-rename column names and re-offered fixes the
    recipe already applies. scanned_columns keys follow renames, so they match current_df.
    """
    ss = st.session_state
    df = ss.get('current_df')
    if df is None:
        df = ss.get('raw_data')
    if df is None:
        return
    proposals = generate_proposals(df, ss.get('scanned_columns', set()))
    ss.proposals = filter_current_proposals(proposals, df)
