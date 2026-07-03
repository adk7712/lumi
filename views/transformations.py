import streamlit as st
import pandas as pd
from state_manager import add_step, get_column_dependencies, sync_column_rename
from streamlit_sortables import sort_items
from ui_utils import load_style, render_loading_spinner

def render_transformations_tab(df):
    all_cols = df.columns.tolist()
    t_type = st.selectbox("Type", ["Find and Replace", "Normalize Text", "Cast Data Type", "Drop Column", "Strip Whitespace", "Rename Column", "Reorder Columns", "Extract Datetime", "Remove Duplicates", "Drop Empty Columns", "Drop Empty Rows"], key="trans_type_select")
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
        dependent_rules = get_column_dependencies(target)
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
                sync_column_rename(target, new_name)
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
            btn_placeholder.markdown(render_loading_spinner("Apply Column Order"), unsafe_allow_html=True)
            import time
            time.sleep(0.8)
            add_step({"action": "reorder_columns", "value": list(st.session_state.temp_col_order)})
            st.session_state.show_reorder_success = True
            st.rerun()
    elif t_type == "Extract Datetime":
        datetime_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        
        if not datetime_cols:
            st.warning("⚠️ No datetime columns detected in the active dataset. Convert a column using 'Cast Data Type' to 'datetime64[ns]' first, or select any column to coerce.")
            target_cols = all_cols
        else:
            target_cols = datetime_cols
            
        c1, c2 = st.columns(2)
        target = c1.selectbox("Source Column", target_cols, key="datetime_extract_col")
        component = c2.selectbox("Component to Extract", ["year", "month", "day", "day_of_week", "hour"], key="datetime_component_select")
        
        # Default name helper: {target}_{component}
        default_new_name = f"{target}_{component}" if target else f"extracted_{component}"
        new_name = st.text_input("New Column Name", value=default_new_name, key="datetime_new_col_name")
        
        if st.button("Add Step", key="btn_extract_datetime"):
            if not new_name.strip():
                st.error("New column name cannot be empty")
            elif new_name in all_cols:
                st.error(f"A column named '{new_name}' already exists.")
            else:
                add_step({
                    "action": "extract_datetime",
                    "column": target,
                    "new_column": new_name,
                    "component": component
                })
                st.rerun()
    elif t_type == "Remove Duplicates":
        st.info("This will remove all exact duplicate rows from the active dataset.")
        if st.button("Add Step", key="btn_remove_dups"):
            add_step({"action": "drop_duplicates"})
            st.rerun()
    elif t_type == "Drop Empty Columns":
        st.info("This will drop any columns that contain 100% missing (null) values.")
        if st.button("Add Step", key="btn_drop_empty_cols"):
            add_step({"action": "drop_empty_columns"})
            st.rerun()
    elif t_type == "Drop Empty Rows":
        st.info("This will remove any rows where all values are completely missing (null).")
        if st.button("Add Step", key="btn_drop_empty_rows"):
            add_step({"action": "drop_empty_rows"})
            st.rerun()
