import streamlit as st
from state_manager import add_step
from streamlit_sortables import sort_items
from ui_utils import load_style

def render_transformations_tab(df):
    all_cols = df.columns.tolist()
    t_type = st.selectbox("Type", ["Find and Replace", "Normalize Text", "Cast Data Type", "Drop Column", "Strip Whitespace", "Rename Column", "Reorder Columns"], key="trans_type_select")
    if t_type == "Find and Replace":
        c1, c2, c3 = st.columns(3)
        sf, sr, target = c1.text_input("Find", key="find_input"), c2.text_input("Replace", key="replace_input"), c3.selectbox("Columns", ["All"] + all_cols, key="replace_target_col")
        use_regex = st.toggle("Use Regular Expressions", key="replace_use_regex")
        if st.button("Add Step", key="btn_fr"):
            add_step({"action": "replace", "column": target, "find": sf, "replace": sr, "regex": use_regex})
            st.rerun()
    elif t_type == "Normalize Text":
        c1, c2 = st.columns(2)
        target, method = c1.selectbox("Columns", ["All"] + all_cols, key="norm_target_col"), c2.selectbox("Method", ["lowercase", "uppercase", "titlecase", "remove_punctuation", "fuzzy_dedupe"], key="norm_method_select")
        if st.button("Add Step", key="btn_norm"):
            add_step({"action": "normalize_text", "column": target, "value": method})
            st.rerun()
    elif t_type == "Cast Data Type":
        c1, c2 = st.columns(2)
        target, dtype_t = c1.selectbox("Column", all_cols, key="cast_target_col"), c2.selectbox("Cast To", ["string", "float64", "int64", "datetime64[ns]"], key="cast_dtype_select")
        if st.button("Add Step", key="btn_cast"):
            add_step({"action": "cast_type", "column": target, "dtype": dtype_t})
            st.rerun()
    elif t_type == "Drop Column":
        target = st.selectbox("Target Column", all_cols, key="drop_target_col")

        # Collision Detection
        dependent_rules = []
        for r in st.session_state.rules:
            if r.get('col') == target or r.get('col_a') == target or r.get('col_b') == target:
                dependent_rules.append(r['desc'])
            elif r.get('type') == "Custom Expression" and target in r.get('query', ''):
                dependent_rules.append(r['desc'])

        if dependent_rules:
            st.warning(f"⚠️ Column '{target}' is used in the following rules: {', '.join(dependent_rules)}. Dropping it may break these rules.")

        if st.button("Add Step", key="btn_drop"):
            add_step({"action": "drop_column", "column": target})
            st.rerun()
    elif t_type == "Strip Whitespace":
        target = st.selectbox("Columns", ["All"] + all_cols, key="strip_target_col")
        if st.button("Add Step", key="btn_strip"):
            add_step({"action": "strip_whitespace", "column": target})
            st.rerun()
    elif t_type == "Rename Column":
        target = st.selectbox("Target Column", all_cols, key="rename_target_col")
        new_name = st.text_input("New Column Name", key="rename_new_name_input")
        if st.button("Add Step", key="btn_rename"):
            if not new_name.strip():
                st.error("Column name cannot be empty")
            elif new_name in all_cols:
                st.error(f"A column named '{new_name}' already exists.")
            else:
                add_step({"action": "rename_column", "column": target, "value": new_name})
                # Sync validation rules
                for rule in st.session_state.rules:
                    if rule.get('col') == target:
                        rule['col'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                    if rule.get('col_a') == target:
                        rule['col_a'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                    if rule.get('col_b') == target:
                        rule['col_b'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                # Sync active features in diagnostics tab
                if target in st.session_state.active_features:
                    idx = st.session_state.active_features.index(target)
                    st.session_state.active_features[idx] = new_name
                st.rerun()
    elif t_type == "Reorder Columns":
        if 'temp_col_order' not in st.session_state or set(st.session_state.temp_col_order) != set(all_cols):
            st.session_state.temp_col_order = list(all_cols)

        temp_cols = st.session_state.temp_col_order

        st.markdown("### Drag-and-Drop to Reorder Columns")
        st.info("Grab any column name card and drag to rearrange the column order. Click \"Apply Column Order\" once done.")

        if st.session_state.get('show_reorder_success'):
            st.toast("Column order applied successfully!", icon="✅")
            st.session_state.show_reorder_success = False

        sortable_style = load_style("sortables.css")
        sorted_cols = sort_items(temp_cols, direction="horizontal", custom_style=sortable_style, key="col_reorder_widget")

        if sorted_cols != temp_cols:
            st.session_state.temp_col_order = sorted_cols
            st.rerun()

        btn_placeholder = st.empty()
        if btn_placeholder.button("Apply Column Order", key="btn_apply_reorder"):
            loading_html = """
            <div style="display: inline-flex; align-items: center; gap: 12px; height: 38px; margin-bottom: 1rem;">
                <button disabled style="
                    border-radius: 6px;
                    border: 1px solid rgba(128, 128, 128, 0.2);
                    background-color: transparent;
                    color: inherit;
                    opacity: 0.4;
                    padding: 0px 16px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 14px;
                    height: 38px;
                    cursor: not-allowed;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    box-sizing: border-box;
                ">Apply Column Order</button>
                <div class="spinner-circle"></div>
            </div>
            """
            btn_placeholder.markdown(loading_html, unsafe_allow_html=True)
            import time
            time.sleep(0.8)
            add_step({"action": "reorder_columns", "value": list(st.session_state.temp_col_order)})
            st.session_state.show_reorder_success = True
            st.rerun()
