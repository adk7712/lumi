import streamlit as st
import plotly.express as px
from rule_utils import evaluate_rule

def render_overview_tab(df):
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    # Calculate overall data health percentage: (1 - proportion of null cells) * 100
    health = int((1 - (null_cells / total_cells)) * 100) if total_cells > 0 else 0
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
