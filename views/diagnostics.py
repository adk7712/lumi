import streamlit as st
import pandas as pd
import plotly.express as px
from ui_utils import render_diagnostic_metric, downsample_for_plot, DIAGNOSTIC_CHART_HEIGHT, MAX_CATEGORIES_DISPLAY

def render_diagnostics_tab(df):
    all_cols = df.columns.tolist()
    selected_features = st.multiselect("Analyze Columns", all_cols, key="active_features")

    if not selected_features:
        st.info("Select one or more columns above to begin analysis")
    else:
        grid_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with grid_cols[idx % 2]:
                with st.container(border=True):
                    st.subheader(col_name)
                    s1, s2, s3, s4 = st.columns(4)
                    render_diagnostic_metric(s1, "Type", str(df[col_name].dtype))
                    render_diagnostic_metric(s2, "Nulls", f"{df[col_name].isnull().sum()}")
                    render_diagnostic_metric(s3, "Unique", f"{df[col_name].nunique()}")
                    # Differentiate plotting and metric display based on data type for relevant insights.
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        render_diagnostic_metric(s4, "Skew", f"{df[col_name].skew():.2f}")
                        
                        plot_df = df[[col_name]].dropna()
                        plot_df, is_sampled = downsample_for_plot(plot_df)
                        fig = px.box(plot_df, x=col_name, height=DIAGNOSTIC_CHART_HEIGHT)
                    else:
                        top_val = df[col_name].mode()[0] if not df[col_name].mode().empty else "N/A"
                        render_diagnostic_metric(s4, "Top", str(top_val)[:10])
                        counts = df[col_name].value_counts()
                        num_uniques = len(counts)
                        if num_uniques <= MAX_CATEGORIES_DISPLAY:
                            chart_data = counts
                        else:
                            top_n = counts.head(MAX_CATEGORIES_DISPLAY - 1)
                            other_sum = counts.iloc[MAX_CATEGORIES_DISPLAY - 1:].sum()
                            chart_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
                        fig = px.bar(x=chart_data.index, y=chart_data.values, height=DIAGNOSTIC_CHART_HEIGHT)

                    # Cleanup chart aesthetics by removing redundant axis labels
                    fig.update_layout(xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig, width="stretch", theme="streamlit")
                    if pd.api.types.is_numeric_dtype(df[col_name]) and len(df) > 1000:
                        st.caption("Showing representative sample of 1,000 rows for rendering performance.")

                    # Detailed Collapsible Statistics (keeps heights equal between numeric/categorical)
                    with st.expander("Detailed Statistics", expanded=False):
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            desc = df[col_name].describe()
                            def fmt(val):
                                return f"{val:.2f}" if pd.notnull(val) else "N/A"
                            st.markdown(f"""
                            <div style="font-size: 0.72rem; line-height: 1.4; opacity: 0.85; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                <div><strong>Min:</strong> {fmt(desc.get('min'))}</div>
                                <div><strong>Q1 (25%):</strong> {fmt(desc.get('25%'))}</div>
                                <div><strong>Median (50%):</strong> {fmt(desc.get('50%'))}</div>
                                <div><strong>Q3 (75%):</strong> {fmt(desc.get('75%'))}</div>
                                <div><strong>Max:</strong> {fmt(desc.get('max'))}</div>
                                <div><strong>Mean:</strong> {fmt(desc.get('mean'))}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            desc = df[col_name].describe()
                            freq_val = desc.get('freq')
                            freq_pct = (freq_val / len(df)) * 100 if pd.notnull(freq_val) and len(df) > 0 else 0.0
                            null_count = df[col_name].isnull().sum()
                            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0.0
                            top_val = desc.get('top', 'N/A')

                            st.markdown(f"""
                            <div style="font-size: 0.72rem; line-height: 1.4; opacity: 0.85; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                <div style="grid-column: span 2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>Most Common:</strong> {str(top_val)[:25]}</div>
                                <div><strong>Frequency:</strong> {freq_val if pd.notnull(freq_val) else 'N/A'}</div>
                                <div><strong>Freq %:</strong> {freq_pct:.1f}%</div>
                                <div><strong>Null Count:</strong> {null_count}</div>
                                <div><strong>Null %:</strong> {null_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
