import streamlit as st
import pandas as pd
import plotly.express as px
import re
from rule_utils import evaluate_rule
from state_manager import add_step, sync_column_rename, calculate_health
from engine_ops import predict_column_renames

def render_overview_tab(df):
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    health = calculate_health(df)
    active_rules_list = [r for r in st.session_state.rules if r.get('enabled', True)]

    total_violations = 0
    for rule in active_rules_list:
        if rule.get('type') == "Informational":
            continue
        try:
            total_violations += evaluate_rule(df, rule).sum()
        except (ValueError, KeyError, TypeError) as e:
            st.toast(f"Overview Rule Error ({rule.get('desc', 'N/A')}): {type(e).__name__} - {str(e)}", icon="🚨")

    m_col1.metric("Health", f"{health}%")
    m_col2.metric("Rows", f"{len(df):,}")
    m_col3.metric("Columns", f"{len(df.columns)}")
    m_col4.metric("Duplicates", f"{df.duplicated().sum():,}")
    m_col5.metric("Violations", f"{total_violations:,}")
    m_col6.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    st.divider()

    o_col1, o_col2 = st.columns(2)
    with o_col1:
        st.subheader("Dataset Composition")
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']
        fig = px.pie(type_counts, names='Data Type', values='Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    with o_col2:
        st.subheader("Workspace Status")
        st.markdown(f"**Recipe Steps:** {len(st.session_state.cleaning_recipe)}  \n**Tracked Features:** {len(st.session_state.active_features)}  \n**Active Rules:** {len(active_rules_list)}")
        st.subheader("Quick Actions")
        duplicates_count = int(df.duplicated().sum())
        empty_cols = [c for c in df.columns if df[c].isnull().all()]
        empty_cols_count = len(empty_cols)
        empty_rows_count = int(df.isnull().all(axis=1).sum())

        # Scan for columns with leading/trailing whitespaces
        whitespace_cols = []
        for c in df.columns:
            if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]):
                non_null_strings = df[c].dropna().astype(str)
                if (non_null_strings != non_null_strings.str.strip()).any():
                    whitespace_cols.append(c)

        # Scan for column names containing spaces, dashes, or special characters
        unnormalized_cols = []
        for c in df.columns:
            if ' ' in c or '-' in c or any(not (char.isalnum() or char == '_') for char in c):
                unnormalized_cols.append(c)

        has_recommended = (duplicates_count > 0 or empty_cols_count > 0 or empty_rows_count > 0 or len(whitespace_cols) > 0 or len(unnormalized_cols) > 0)

        if has_recommended:
            with st.container(border=False):
                if duplicates_count > 0:
                    if st.button(
                        f"Remove {duplicates_count} Duplicate Rows",
                        key="qa_drop_duplicates",
                        width="stretch",
                        help=f"Removes the {duplicates_count} rows that contain identical values across all features."
                    ):
                        add_step({"action": "drop_duplicates"})
                        st.rerun()
                if empty_cols_count > 0:
                    cols_preview = ", ".join(empty_cols[:3]) + ("..." if len(empty_cols) > 3 else "")
                    if st.button(
                        f"Drop {empty_cols_count} Empty Columns",
                        key="qa_drop_empty_cols",
                        width="stretch",
                        help=f"Drops columns containing 100% missing values: {cols_preview}"
                    ):
                        add_step({"action": "drop_empty_columns"})
                        st.rerun()
                if empty_rows_count > 0:
                    if st.button(
                        f"Remove {empty_rows_count} Empty Rows",
                        key="qa_drop_empty_rows",
                        width="stretch",
                        help=f"Removes the {empty_rows_count} rows that are entirely empty."
                    ):
                        add_step({"action": "drop_empty_rows"})
                        st.rerun()
                if whitespace_cols:
                    cols_preview = ", ".join(whitespace_cols[:3]) + ("..." if len(whitespace_cols) > 3 else "")
                    if st.button(
                        "Strip Column Whitespace",
                        key="qa_strip_whitespace",
                        width="stretch",
                        help=f"Strips leading/trailing whitespaces from text columns: {cols_preview}"
                    ):
                        add_step({"action": "strip_whitespace", "column": "All"})
                        st.rerun()
                if unnormalized_cols:
                    cols_preview = ", ".join(f"'{c}'" for c in unnormalized_cols[:3]) + ("..." if len(unnormalized_cols) > 3 else "")
                    if st.button(
                        "Normalize Column Names",
                        key="qa_normalize_cols",
                        width="stretch",
                        help=f"Converts column headers to snake_case (lowercase with underscores) to avoid syntax errors: {cols_preview}"
                    ):
                        new_names = predict_column_renames(df.columns.tolist(), 'snake_case', only_changed=True)

                        add_step({"action": "normalize_column_names", "value": "snake_case"})
                        for orig, val in new_names.items():
                            sync_column_rename(orig, val)
                        st.rerun()
        else:
            st.markdown("*Your dataset looks clean! No quick actions recommended.*")

        if duplicates_count > 0:
            with st.expander(f"Preview {duplicates_count} Duplicate Rows", expanded=False):
                st.dataframe(df[df.duplicated(keep=False)], width="stretch")

    st.divider()

    # Side-by-side filters at the top of the section
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("Filter by Column Name", all_cols, default=[], placeholder="All columns active...", key="desc_cols")
    with f_col2:
        unique_dtypes = sorted(list(set(str(t) for t in df.dtypes)))
        selected_dtypes = st.multiselect("Filter by Data Type", unique_dtypes, default=[], placeholder="All datatypes active...", key="desc_dtypes")
    st.caption("*Tip: Leave filters empty to view all columns/datatypes by default.*")

    # Apply global filtering logic
    filtered_df = df
    if selected_cols:
        filtered_df = filtered_df[selected_cols]
    if selected_dtypes:
        matched_cols = [c for c in filtered_df.columns if str(filtered_df[c].dtype) in selected_dtypes]
        filtered_df = filtered_df[matched_cols] if matched_cols else pd.DataFrame()

    if filtered_df.empty or len(filtered_df.columns) == 0:
        st.info("No columns match the selected filters.")
    else:
        # Calculate summary metrics per column for the filtered subset
        summary_data = []
        for col in filtered_df.columns:
            null_count = int(filtered_df[col].isnull().sum())
            null_pct = f"{(null_count / len(filtered_df)) * 100:.1f}%" if len(filtered_df) > 0 else "0.0%"

            # Calculate summary values based on datatype compatibility
            mean_val = "N/A"
            min_val = "N/A"
            max_val = "N/A"
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                mean_val = f"{filtered_df[col].mean():.2f}" if not filtered_df[col].dropna().empty else "N/A"
                min_val = f"{filtered_df[col].min():.2f}" if not filtered_df[col].dropna().empty else "N/A"
                max_val = f"{filtered_df[col].max():.2f}" if not filtered_df[col].dropna().empty else "N/A"
            else:
                mode_series = filtered_df[col].mode()
                mean_val = f"Mode: {str(mode_series[0])}" if not mode_series.empty else "N/A"

            summary_data.append({
                "Column Name": col,
                "Data Type": str(filtered_df[col].dtype),
                "Null Count": null_count,
                "Null %": null_pct,
                "Unique Values": int(filtered_df[col].nunique()),
                "Mean / Mode": mean_val,
                "Min": min_val,
                "Max": max_val
            })

        summary_df = pd.DataFrame(summary_data)
        desc_df = filtered_df.describe(include='all').astype(str).replace('nan', 'NaN')

        # Display tabs side by side with df.describe as default
        t_describe, t_summary = st.tabs(["Descriptive Statistics (df.describe)", "Column Summary (df.info)"])
        with t_describe:
            st.dataframe(desc_df, width="stretch")
        with t_summary:
            st.dataframe(summary_df, width="stretch", hide_index=True)
