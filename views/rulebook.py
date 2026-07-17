import streamlit as st
import pandas as pd
from state_manager import add_step, add_rule
from rule_utils import evaluate_rule, create_resolution_step, generate_evidence_report

def render_rulebook_tab(df):
    all_cols = df.columns.tolist()
    if st.session_state.proposals:
        with st.expander(f"Recommended Rules ({len(st.session_state.proposals)})", expanded=False):
            if st.button("Accept All Recommendations", key="accept_all_proposals", width="stretch"):
                for p in st.session_state.proposals:
                    st.session_state.scanned_columns.add(p['column'])
                    if 'action' in p['rule_data']:
                        add_step(p['rule_data'])
                    else:
                        add_rule(p['rule_data'], at_end=True)
                st.session_state.proposals = []
                st.toast("All recommendations accepted")
                st.rerun()

            p_cols = st.columns(2)
            for p_idx, p in enumerate(st.session_state.proposals):
                with p_cols[p_idx % 2]:
                    st.markdown(f'<div class="proposal-box"><strong>{p["type"]} on {p["column"]}</strong><br/><small>{p["reason"]}</small></div>', unsafe_allow_html=True)
                    acc, dis = st.columns(2)
                    if acc.button("Accept", key=f"p_acc_{p_idx}", width="stretch"):
                        st.session_state.scanned_columns.add(p['column'])
                        if 'action' in p['rule_data']:
                            add_step(p['rule_data'])
                        else:
                            add_rule(p['rule_data'], at_end=False)
                        st.session_state.proposals.pop(p_idx)
                        st.rerun()
                    if dis.button("Dismiss", key=f"p_dis_{p_idx}", width="stretch"):
                        st.session_state.scanned_columns.add(p['column'])
                        st.session_state.proposals.pop(p_idx)
                        st.rerun()
        st.divider()

    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("New Rule")
        rtype = st.selectbox("Type", ["Null Check", "Range Check", "Relational Check", "Custom Expression", "Informational"], key="rule_type_select")
        if rtype == "Informational":
            note = st.text_area("Note/Warning", placeholder="e.g., This column contains high cardinality data.", key="info_note_input")
            if st.button("Add Rule", key="btn_add_info"):
                add_rule({"type": "Informational", "desc": note})
                st.rerun()
        elif rtype == "Custom Expression":
            with st.form(key="custom_expr_form", clear_on_submit=True):
                q_str = st.text_input("Pandas Query String", placeholder="Age > 30 & Sex == 'male'", key="custom_query_input")
                submit_btn = st.form_submit_button("Add Rule")

            if submit_btn and q_str:
                try:
                    test_result = df.query(q_str)

                    if len(df) > 0 and len(test_result) == 0:
                        st.error("This query returned no matches on the dataset. Please check for typos or type mismatches (e.g., comparing a number to a string). Rule not added.")
                    else:
                        add_rule({"type": "Custom Expression", "query": q_str, "desc": f"Matches: {q_str}"})
                        st.rerun()
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
                    st.rerun()
            else:
                val = st.text_input("Constant Value", key="rel_val_input")
                if st.button("Add Rule", key="btn_add_rel_val"):
                    try: final_val = float(val)
                    except (ValueError, TypeError): final_val = val
                    add_rule({"type": "Relational Check", "col_a": tcol, "op": op, "value": final_val, "target_type": "Value", "desc": f"{tcol} {op} {val}"})
                    st.rerun()
        else:
            tcol = st.selectbox("Target Column", all_cols, key="rule_target_col")
            if rtype == "Range Check":
                if pd.api.types.is_numeric_dtype(df[tcol]):
                    num_col1, num_col2 = st.columns(2)
                    v_min, v_max = num_col1.number_input("Min", value=float(df[tcol].min()), key="range_min_input"), num_col2.number_input("Max", value=float(df[tcol].max()), key="range_max_input")
                    if st.button("Add Rule", key="btn_add_range"):
                        add_rule({"type": "Range Check", "col": tcol, "min": v_min, "max": v_max, "desc": f"{tcol} in [{v_min}, {v_max}]"})
                        st.rerun()
                else:
                    st.warning(f"Range Checks are only applicable to numeric columns. '{tcol}' is {df[tcol].dtype}.")
            elif rtype == "Null Check":
                if st.button("Add Rule", key="btn_add_null"):
                    add_rule({"type": "Null Check", "col": tcol, "desc": f"{tcol} is NOT NULL"})
                    st.rerun()
    with r2:
        st.subheader("Active Rules")
        if st.session_state.rules:
            btn_col1, btn_col2 = st.columns(2)
            # Cache the report content to avoid recalculating on every rerun/refresh
            cache_key = (len(df), len(st.session_state.rules), len(st.session_state.cleaning_recipe))
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
            if btn_col2.button("Clear All", use_container_width=True, key="clear_all_rules_btn"):
                st.session_state.rules, st.session_state.cleaning_recipe = [], []
                st.rerun()

        if not st.session_state.rules:
            st.info("Add a rule from the left panel")
        else:
            with st.container(height=600, border=False):
                for idx, rule in enumerate(st.session_state.rules):
                    v_count = 0
                    if rule['enabled']:
                        try:
                            mask = evaluate_rule(df, rule)
                            v_count = mask.sum()
                            rule.pop('error', None)
                        except (ValueError, KeyError, TypeError) as e:
                            rule['error'] = str(e)

                    status_color, resolved = (rule['color'] if rule['enabled'] else "rgba(100,100,100,0.2)"), rule.get('resolved', False)
                    enabled_class = "enabled" if rule['enabled'] else "disabled"
                    resolved_html = f'<br/><span class="rule-status-resolved">Status: Resolved</span>' if resolved else ""
                    error_html = f'<br/><span class="rule-error-msg">Error: {rule.get("error")}</span>' if 'error' in rule else ""

                    v_text = f"Violations: {v_count}" if rule['type'] != "Informational" else "Type: Info"
                    st.markdown(f"""<div class="violation-card {enabled_class}">
<div class="violation-card-border" style="border-left: 8px solid {status_color};">
<strong class="rule-type">{rule["type"]}</strong><br/>
<code class="rule-desc">{rule["desc"]}</code><br/>
<span class="rule-violations-count">{v_text}</span>
{resolved_html}
{error_html}
</div>
</div>""", unsafe_allow_html=True)

                    if v_count > 0 and not resolved and rule['type'] != "Informational":
                        if rule['type'] == "Null Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Resolution", ["Select resolution method...", "Drop Rows", "Fill with Mean", "Fill with Median", "KNN Imputer", "Iterative Imputer"], key=f"res_{idx}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_res_{idx}", width="stretch"):
                                add_step(create_resolution_step(rule, res))
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        elif rule['type'] == "Range Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Res", ["Select resolution method...", "Drop Rows", "Cap at Bounds", "Log Transform"], key=f"range_res_{idx}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_range_res_{idx}", width="stretch"):
                                add_step(create_resolution_step(rule, res))
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        else:
                            if st.button("Drop Violated Rows", key=f"gen_res_{idx}", width="stretch"):
                                add_step(create_resolution_step(rule, "Drop Violated Rows"))
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()

                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{idx}", width="stretch"):
                        st.session_state.rules[idx]['enabled'] = not rule['enabled']
                        st.rerun()
                    if btn_c2.button("Remove", key=f"del_{idx}", width="stretch"):
                        st.session_state.rules.pop(idx)
                        st.rerun()
