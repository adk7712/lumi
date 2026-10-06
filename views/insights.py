import streamlit as st
import numpy as np
from ui_utils import plot_correlation_matrix, plot_missingness_map

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

def render_insights_tab(df):
    st.info("Visualizations have been reorganized. The Correlation Matrix is now in the Diagnostics tab, and the Missingness Map is in the Overview tab.")
