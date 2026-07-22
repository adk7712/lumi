import streamlit as st
import numpy as np
from ui_utils import plot_correlation_matrix, plot_missingness_map, plot_outlier_distribution

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
def render_outliers_panel(df, df_state_key):
    st.markdown("### Global Outlier Distribution")
    st.markdown("Compares distributions of all numeric features on a single box plot visualization to highlight outliers.")

    fig_outliers, is_outliers_sampled = plot_outlier_distribution(df, df_state_key)
    if fig_outliers is not None:
        st.markdown("*Note: Features are standardized to Z-scores (mean=0, std=1) to allow direct visual comparison across different scales.*")
        st.plotly_chart(fig_outliers, width="stretch", theme="streamlit")
        if is_outliers_sampled:
            st.caption("Showing representative sample of 1,000 rows for rendering performance.")
    else:
        st.caption("No numeric features to display outliers.")

def render_insights_tab(df):
    last_file_hash = st.session_state.get("last_file_hash", "none")
    step_count = len(st.session_state.get("cleaning_recipe", []))
    df_state_key = f"{last_file_hash}_{step_count}"
    
    render_correlation_panel(df, df_state_key)
    st.divider()
    render_missingness_panel(df, df_state_key)
    st.divider()
    render_outliers_panel(df, df_state_key)
