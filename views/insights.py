import streamlit as st
import numpy as np
from ui_utils import plot_correlation_matrix, plot_missingness_map, plot_outlier_distribution

def render_insights_tab(df):
    # 1. Filtered Correlation Heatmap
    st.markdown("### Feature Correlation")
    st.markdown("Shows relationships between numeric columns. Features without any correlation within the selected range are filtered out.")
    corr_range = st.slider("Correlation Range", -1.0, 1.0, (-1.0, 1.0), 0.05, key="corr_range_val")

    fig_corr = plot_correlation_matrix(df, corr_range)
    if fig_corr is not None:
        st.plotly_chart(fig_corr, width="stretch", theme="streamlit")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) <= 1:
            st.caption("Not enough numeric columns for correlation matrix.")
        else:
            st.info("No numeric columns have correlation within the selected range.")

    st.divider()

    # 2. Missingness Map
    st.markdown("### Missingness Pattern Map")
    st.markdown("Visualizes where missing values occur across the rows of the dataset.")

    fig_null, is_null_sampled = plot_missingness_map(df)
    if fig_null is not None:
        st.plotly_chart(fig_null, width="stretch", theme="streamlit")
        if is_null_sampled:
            st.caption("Showing a representative sample of 1,000 rows for rendering performance.")
    else:
        if df.size > 0:
            st.success("🎉 No missing values found in the dataset!")
        else:
            st.caption("Dataset is empty.")

    st.divider()

    # 3. Outliers Grid
    st.markdown("### Global Outlier Distribution")
    st.markdown("Compares distributions of all numeric features on a single box plot visualization to highlight outliers.")

    fig_outliers, is_outliers_sampled = plot_outlier_distribution(df)
    if fig_outliers is not None:
        st.markdown("*Note: Features are standardized to Z-scores (mean=0, std=1) to allow direct visual comparison across different scales.*")
        st.plotly_chart(fig_outliers, width="stretch", theme="streamlit")
        if is_outliers_sampled:
            st.caption("Showing representative sample of 1,000 rows for rendering performance.")
    else:
        st.caption("No numeric features to display outliers.")
