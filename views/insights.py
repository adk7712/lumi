import streamlit as st
import numpy as np
from ui_utils import plot_correlation_matrix, plot_missingness_map, plot_zscore_boxes, zscore_candidate_columns, MAX_BOX_COLUMNS

@st.fragment
def render_correlation_panel(df, df_state_key):
    st.markdown("### Feature Correlation")
    st.markdown("Shows relationships between numeric columns. Features without any correlation within the selected range are filtered out.")
    corr_range = st.slider("Correlation Range", -1.0, 1.0, (-1.0, 1.0), 0.05, key="corr_range_val")

    fig_corr = plot_correlation_matrix(df, corr_range, df_state_key)
    if fig_corr is not None:
        st.plotly_chart(fig_corr, width="stretch", theme="streamlit")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) <= 1:
            st.caption("Not enough numeric columns for correlation matrix.")
        else:
            st.info("No numeric columns have correlation within the selected range.")

@st.fragment
def render_missingness_panel(df, df_state_key):
    st.markdown("### Missingness Pattern Map")
    st.markdown("Visualizes where missing values occur across the rows of the dataset.")

    fig_null, is_null_sampled = plot_missingness_map(df, df_state_key)
    if fig_null is not None:
        st.plotly_chart(fig_null, width="stretch", theme="streamlit")
        if is_null_sampled:
            st.caption("Showing a representative sample of 1,000 rows for rendering performance.")
    else:
        if df.size > 0:
            st.success("No missing values found in the dataset!")
        else:
            st.caption("Dataset is empty.")


@st.fragment
def render_outlier_panel(df, df_state_key):
    st.markdown("### Standardized Outlier Box Plots")
    st.markdown("All numeric variables scaled to Z-scores for a direct, side-by-side comparison of outliers.")
    candidates = zscore_candidate_columns(df)
    if not candidates:
        st.caption("No numeric columns with variance available for outlier comparison.")
        return
    selected = candidates[:MAX_BOX_COLUMNS]
    if len(candidates) > MAX_BOX_COLUMNS:
        selected = st.multiselect(
            f"Columns to compare (max {MAX_BOX_COLUMNS})", candidates,
            default=candidates[:MAX_BOX_COLUMNS], max_selections=MAX_BOX_COLUMNS,
            key="outlier_box_cols",
        )
    if not selected:
        st.info("Select at least one column.")
        return
    fig, is_sampled = plot_zscore_boxes(df, tuple(selected), df_state_key)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        if is_sampled:
            st.caption("Showing a representative sample of 1,000 rows for rendering performance.")
