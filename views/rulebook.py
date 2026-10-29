import html
import streamlit as st
import pandas as pd
from state_manager import add_step, add_rule, save_session_state, save_db_session, regenerate_proposals
from query_guard import UnsafeQueryError, safe_query_index
from rule_utils import evaluate_rule, create_resolution_step, generate_evidence_report
from lumi_state.steps import get_rule_masks, recipe_hash, rules_hash, rule_uids, persist_all, filter_current_proposals

def _numeric_bounds(series, default=(0.0, 0.0)):
    """(min, max) of a numeric column as finite floats; ``default`` when the column is empty/all-NA.

    float() on pd.NA (nullable Int64/Float64 with no values) raises TypeError, and NaN/inf
    are not valid number_input defaults.
    """
    import math
    try:
        lo, hi = series.min(skipna=True), series.max(skipna=True)
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError, OverflowError):
        return default
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return default
    return lo, hi

def _accept_proposal(p, at_end=False):
    st.session_state.scanned_columns.add(f"{p['column']}:{p['type']}")
    data = p['rule_data'].copy()
    data['_source_proposal_type'] = p['type']
    data['_source_proposal_col'] = p['column']
    if 'action' in data:
        # add_step moves rules and pending proposals onto renamed columns before it saves
        add_step(data)
    else:
        add_rule(data, at_end=at_end)

def handle_accept_all():
    for p in st.session_state.proposals:
        _accept_proposal(p, at_end=True)
    st.session_state.proposals = []
    st.toast("All recommendations accepted")

def _live_proposal_index(p_idx, p):
    """Index of proposal ``p`` in the current list, or None when a click comes from a stale card
    (the list was regenerated or filtered since that card rendered)."""
    proposals = st.session_state.get("proposals") or []
    if 0 <= p_idx < len(proposals) and (proposals[p_idx] is p or proposals[p_idx] == p):
        return p_idx
    for i, q in enumerate(proposals):
        if q is p or q == p:
            return i
    return None

def handle_accept_proposal(p_idx, p):
    idx = _live_proposal_index(p_idx, p)
    if idx is None:
        return
    _accept_proposal(p)
    st.session_state.proposals.pop(idx)

def handle_dismiss_proposal(p_idx, p):
    idx = _live_proposal_index(p_idx, p)
    if idx is None:
        return
    st.session_state.scanned_columns.add(f"{p['column']}:{p['type']}")
    st.session_state.proposals.pop(idx)
    save_session_state()
    save_db_session()

def handle_clear_all():
    """Clear all rules. Cleaning steps are kept: they are already-applied data
    transformations (including ones created from rules) and are undoable via Undo."""
    for r in st.session_state.rules:
        if '_source_proposal_type' in r and '_source_proposal_col' in r:
            st.session_state.scanned_columns.discard(f"{r['_source_proposal_col']}:{r['_source_proposal_type']}")
    st.session_state.rules = []
    regenerate_proposals()
    save_session_state()
    save_db_session()

def _request_clear_rules():
    st.session_state._confirm_clear_rules = True

def _cancel_clear_rules():
    st.session_state.pop("_confirm_clear_rules", None)

def _confirm_clear_rules():
    st.session_state.pop("_confirm_clear_rules", None)
    handle_clear_all()
    st.session_state._needs_rerun = True

def render_clear_rules_confirm():
    """Inline two-step confirmation (flag-driven so it survives reruns and is cancelable)."""
    n = len(st.session_state.rules)
    st.warning(f"Remove all {n} rule{'s' if n != 1 else ''}? Cleaning steps you have already applied are kept. This cannot be undone.")
    c1, c2 = st.columns(2)
    c1.button("Yes, Clear Rules", type="primary", use_container_width=True, key="confirm_clear_rules_btn", on_click=_confirm_clear_rules)
    c2.button("Cancel", use_container_width=True, key="cancel_clear_rules_btn", on_click=_cancel_clear_rules)

def handle_remove_rule(idx):
    r = st.session_state.rules[idx]
    if '_source_proposal_type' in r and '_source_proposal_col' in r:
        st.session_state.scanned_columns.discard(f"{r['_source_proposal_col']}:{r['_source_proposal_type']}")
    st.session_state.rules.pop(idx)
    regenerate_proposals()
    save_session_state()
    save_db_session()

def handle_toggle_rule(idx, desc=None):
    """Flip a rule's Enable/Ignore flag and persist it (local cache, DB, project cache) right away,
    in the click callback, so a quick reload cannot restore the previous state."""
    rules = st.session_state.rules
    if not (0 <= idx < len(rules)) or (desc is not None and rules[idx].get('desc') != desc):
        return  # stale click (the list changed since this card rendered)
    rules[idx]['enabled'] = not rules[idx].get('enabled', True)
    persist_all()

def _apply_resolution(idx, rule, method):
    """Apply a resolution step, mark the rule resolved only if its violations are gone, then save once."""
    add_step(create_resolution_step(rule, method), persist=False)
    # Only mark resolved if violations are actually gone after applying the step.
    try:
        remaining = evaluate_rule(st.session_state.current_df, rule).sum()
        if remaining == 0:
            st.session_state.rules[idx]['resolved'] = True
    except Exception:
        pass
    persist_all()

def render_rulebook_tab(df):
    all_cols = df.columns.tolist()
    if st.session_state.proposals:
        # Never offer a proposal for a column the current data lacks, or one already applied
        # as a step / active rule (Accept would target a missing column or duplicate it).
        st.session_state.proposals = filter_current_proposals(st.session_state.proposals, df)
    # Always-present slot: the recommendations come and go, and without a fixed container the
    # rule builder below would shift position in the element tree (remounting its widgets).
    with st.container():
        if st.session_state.proposals:
            is_expanded = st.session_state.get("recommendations_expanded", True)
            with st.expander(f"Recommended Rules ({len(st.session_state.proposals)})", expanded=is_expanded):
                st.button("Accept All Recommendations", key="accept_all_proposals", width="stretch", on_click=handle_accept_all)

                # Render row by row (pairs) so cards stay aligned when card heights differ.
                for row_start in range(0, len(st.session_state.proposals), 2):
                    p_cols = st.columns(2)
                    for p_idx in range(row_start, min(row_start + 2, len(st.session_state.proposals))):
                        p = st.session_state.proposals[p_idx]
                        with p_cols[p_idx - row_start]:
                            st.markdown(f'<div class="proposal-box"><strong>{html.escape(str(p["title"]) + " (dataset)" if p["column"] == "__dataset__" else str(p["type"]) + " on " + str(p["column"]))}</strong><br/><small>{html.escape(str(p["reason"]))}</small></div>', unsafe_allow_html=True)
                            acc, dis = st.columns(2)
                            acc.button("Accept", key=f"p_acc_{p['column']}_{p['type']}_{p_idx}", width="stretch", on_click=handle_accept_proposal, args=(p_idx, p))
                            dis.button("Dismiss", key=f"p_dis_{p['column']}_{p['type']}_{p_idx}", width="stretch", on_click=handle_dismiss_proposal, args=(p_idx, p))
                        
            st.divider()

    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("New Rule")
        rtype = st.selectbox("Type", ["Null Check", "Range Check", "Relational Check", "Custom Expression", "Informational"], key="rule_type_select")
        if rtype == "Informational":
            note = st.text_area("Note/Warning", placeholder="e.g., This column contains high cardinality data.", key="info_note_input")
            if st.button("Add Rule", key="btn_add_info"):
                add_rule({"type": "Informational", "desc": note})
                st.session_state._needs_rerun = True
                
        elif rtype == "Custom Expression":
            with st.form(key="custom_expr_form", clear_on_submit=True):
                st.caption("Write a Pandas query that **flags bad rows** (violations). Matching rows will be treated as violations.")
                q_str = st.text_input("Pandas Query String", placeholder="age > 100 or age < 0", key="custom_query_input")
                submit_btn = st.form_submit_button("Add Rule")

            if submit_btn and q_str:
                try:
                    test_result = df.loc[safe_query_index(df, q_str)]

                    if len(df) > 0 and len(test_result) == 0:
                        st.warning("This query matches no rows on the current dataset — the rule will be added but show 0 violations. Check for typos or type mismatches.")
                        add_rule({"type": "Custom Expression", "query": q_str, "desc": f"Violates: {q_str}"})
                        st.session_state._needs_rerun = True
                        
                    else:
                        add_rule({"type": "Custom Expression", "query": q_str, "desc": f"Violates: {q_str}"})
                        st.session_state._needs_rerun = True
                        
                except UnsafeQueryError as e:
                    st.error(f"Expression not allowed: {e}. Use column names, comparisons (>, ==, in ...), and/or/not, arithmetic, .isnull()/.notnull(), or .str.contains/startswith/endswith('text'). No '@', other function calls or double underscores.")
                except Exception as e:
                    err_msg = str(e)
                    if "invalid syntax" in err_msg.lower():
                        friendly_err = "Syntax Error: The expression has invalid characters or structure (e.g., using '>>' instead of '>')."
                    elif "is not defined" in err_msg.lower() or "not found" in err_msg.lower():
                        friendly_err = "Column Error: One of the columns mentioned in your query doesn't exist in the dataset."
                    elif "cannot compare" in err_msg.lower() or "not supported between instances" in err_msg.lower() or "typeerror" in err_msg.lower():
                        friendly_err = "Type Error: You are trying to compare incompatible types (e.g., a number with a string)."
                    else:
                        friendly_err = f"Pandas Error: {err_msg}"
                    st.error(friendly_err)
        elif rtype == "Relational Check":
            tcol, op = st.selectbox("Feature A", all_cols, key="rel_feature_a"), st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="], key="rel_op")
            target_type = st.radio("Compare with", ["Another Feature", "Constant Value"], horizontal=True, key="rel_target_type_radio")
            if target_type == "Another Feature":
                col_b = st.selectbox("Feature B", all_cols, key="rel_feature_b")
                if st.button("Add Rule", key="btn_add_rel_feat"):
                    add_rule({"type": "Relational Check", "col_a": tcol, "op": op, "col_b": col_b, "target_type": "Feature", "desc": f"{tcol} {op} {col_b}"})
                    st.session_state._needs_rerun = True
                    
            else:
                val = st.text_input("Constant Value", key="rel_val_input")
                if st.button("Add Rule", key="btn_add_rel_val"):
                    try: final_val = float(val)
                    except (ValueError, TypeError): final_val = val
                    add_rule({"type": "Relational Check", "col_a": tcol, "op": op, "value": final_val, "target_type": "Value", "desc": f"{tcol} {op} {val}"})
                    st.session_state._needs_rerun = True
                    
        else:
            tcol = st.selectbox("Target Column", all_cols, key="rule_target_col")
            if rtype == "Range Check":
                if pd.api.types.is_numeric_dtype(df[tcol]):
                    num_col1, num_col2 = st.columns(2)
                    _lo, _hi = _numeric_bounds(df[tcol])
                    v_min, v_max = num_col1.number_input("Min", value=_lo, key="range_min_input"), num_col2.number_input("Max", value=_hi, key="range_max_input")
                    if st.button("Add Rule", key="btn_add_range"):
                        add_rule({"type": "Range Check", "col": tcol, "min": v_min, "max": v_max, "desc": f"{tcol} in [{v_min}, {v_max}]"})
                        st.session_state._needs_rerun = True
                        
                else:
                    st.warning(f"Range Checks are only applicable to numeric columns. '{tcol}' is {df[tcol].dtype}.")
            elif rtype == "Null Check":
                if st.button("Add Rule", key="btn_add_null"):
                    add_rule({"type": "Null Check", "col": tcol, "desc": f"{tcol} is NOT NULL"})
                    st.session_state._needs_rerun = True
                    
    with r2:
        st.subheader("Active Rules")
        if st.session_state.rules:
            btn_col1, btn_col2 = st.columns(2)
            # Cache the report content to avoid recalculating on every rerun/refresh
            cache_key = (st.session_state.get('session_id'), recipe_hash(), rules_hash(), len(df))
            if 'report_cache_key' not in st.session_state or st.session_state.report_cache_key != cache_key:
                st.session_state.report_cache_key = cache_key
                st.session_state.report_content = generate_evidence_report(
                    df, 
                    st.session_state.rules,
                    cleaning_recipe=st.session_state.cleaning_recipe,
                    original_df=st.session_state.raw_data
                )
            report_content = st.session_state.report_content
            btn_col1.download_button(
                label="Download Evidence Report",
                data=report_content,
                file_name="evidence_report.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_evidence_report_btn"
            )
            btn_col2.button("Clear Rules", use_container_width=True, key="clear_all_rules_btn", on_click=_request_clear_rules)
            if st.session_state.get("_confirm_clear_rules"):
                render_clear_rules_confirm()

        if not st.session_state.rules:
            st.session_state.pop("_confirm_clear_rules", None)
            st.info("Add a rule from the left panel")
        else:
            with st.container(height=600, border=False):
                # Violation masks are computed once per run signature and shared with the other views
                _masks = get_rule_masks(df)
                _uids = rule_uids()
                _violation_counts = {}
                for _idx, _rule in enumerate(st.session_state.rules):
                    _res = _masks.get(_idx)
                    if isinstance(_res, Exception):
                        if not isinstance(_res, (ValueError, KeyError, TypeError)):
                            raise _res
                        _violation_counts[_idx] = 0
                        _rule['error'] = str(_res)
                    else:
                        _violation_counts[_idx] = 0 if _res is None else int(_res.sum())
                        _rule.pop('error', None)

                for idx, rule in enumerate(st.session_state.rules):
                    v_count = _violation_counts.get(idx, 0)
                    uid = _uids[idx]

                    status_color, resolved = (rule['color'] if rule['enabled'] else "rgba(100,100,100,0.2)"), rule.get('resolved', False)
                    enabled_class = "enabled" if rule['enabled'] else "disabled"
                    resolved_html = f'<br/><span class="rule-status-resolved">Status: Resolved</span>' if resolved else ""
                    error_html = f'<br/><span class="rule-error-msg">Error: {html.escape(str(rule.get("error")))}</span>' if 'error' in rule else ""

                    v_text = f"Violations: {v_count}" if rule['type'] != "Informational" else "Type: Info"
                    st.markdown(f"""<div class="violation-card {enabled_class}">
<div class="violation-card-border" style="border-left: 8px solid {html.escape(str(status_color), quote=True)};">
<strong class="rule-type">{html.escape(str(rule["type"]))}</strong><br/>
<code class="rule-desc">{html.escape(str(rule["desc"]))}</code><br/>
<span class="rule-violations-count">{v_text}</span>
{resolved_html}
{error_html}
</div>
</div>""", unsafe_allow_html=True)

                    if v_count > 0 and not resolved and rule['type'] != "Informational":
                        if rule['type'] == "Null Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Resolution", ["Select resolution method...", "Drop Rows", "Fill with Mean", "Fill with Median", "KNN Imputer", "Iterative Imputer"], key=f"res_{uid}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_res_{uid}", width="stretch"):
                                with st.spinner("Applying resolution..."):
                                    _apply_resolution(idx, rule, res)
                                st.session_state._needs_rerun = True
                                
                        elif rule['type'] == "Range Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Res", ["Select resolution method...", "Drop Rows", "Cap at Bounds", "Log Transform"], key=f"range_res_{uid}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_range_res_{uid}", width="stretch"):
                                with st.spinner("Applying resolution..."):
                                    _apply_resolution(idx, rule, res)
                                st.session_state._needs_rerun = True
                                
                        else:
                            if st.button("Drop Violated Rows", key=f"gen_res_{uid}", width="stretch"):
                                with st.spinner("Dropping violated rows..."):
                                    _apply_resolution(idx, rule, "Drop Violated Rows")
                                st.session_state._needs_rerun = True
                                

                    btn_c1, btn_c2 = st.columns(2)
                    btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{uid}", width="stretch",
                                  on_click=handle_toggle_rule, args=(idx, rule.get('desc')))
                    btn_c2.button("Remove", key=f"del_{uid}", width="stretch", on_click=handle_remove_rule, args=(idx,))
                        
