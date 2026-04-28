import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Lumi Workspace", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Import Code-like Font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, span, label, button, .stCaption, strong, code, li, ul, div {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Remove Streamlit Header and Headers Icons */
    header { display: none !important; }
    [data-testid="stHeaderActionElements"] { display: none; }
    .stMarkdown a { display: none; }

    /* Push content up to the very top */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem;
    }

    /* Clean Metric Styling (No Box) */
    [data-testid="stMetric"] {
        background-color: transparent;
        border-left: 3px solid rgba(28, 131, 225, 0.5);
        padding: 2px 15px;
        margin-bottom: 10px;
    }

    /* Subtle Violation Card */
    .violation-card {
        padding: 10px 0;
        padding-right: 20px; /* Space for scrollbar */
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        margin-bottom: 10px;
        position: relative;
        z-index: 1;
    }

    /* Bigger Tab Labels and Spacing */
    .stTabs [data-baseweb="tab-list"] button,
    button[data-baseweb="tab"],
    button[role="tab"],
    button[data-testid="stBaseButton-tab"] {
        margin-right: 10px !important;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p,
    button[data-baseweb="tab"] p,
    button[role="tab"] p,
    button[role="tab"] span,
    button[data-testid="stBaseButton-tab"] p {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    </style>

""", unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if 'active_features' not in st.session_state:
    st.session_state.active_features = []
if 'rules' not in st.session_state:
    st.session_state.rules = []

@st.cache_data
def load_initial_data():
    try:
        return pd.read_csv("mock_data/train.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = load_initial_data()

# --- UTILS ---
def get_safe_hue(n):
    safe_hues = [200, 240, 280, 310, 185, 220, 260, 300, 330]
    return safe_hues[n % len(safe_hues)]

# --- CALLBACKS ---
def sync_data():
    st.session_state.active_features = []
    st.session_state.rules = []
    st.session_state.raw_data = load_initial_data()
    st.toast("Workspace Reset")

def data_edited():
    st.toast("Data Syncing")

def delete_rule(idx):
    st.session_state.rules.pop(idx)

def toggle_rule(idx):
    st.session_state.rules[idx]['enabled'] = not st.session_state.rules[idx]['enabled']

def add_rule(new_rule):
    st.session_state.rules.append(new_rule)
    st.toast("Rule Added")

# --- HEADER ---
st.title("LUMI")
st.divider()

# --- DATA PREP ---
df = st.session_state.raw_data
all_cols = df.columns.tolist()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Diagnostics", "Rulebook", "Find and Replace"])

with tab1:
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    health = int((1 - (null_cells / total_cells)) * 100) if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()

    # Calculate fresh total violations
    active_rules_list = [r for r in st.session_state.rules if r.get('enabled', True)]
    total_violations = 0
    for rule in active_rules_list:
        try:
            mask = pd.Series(False, index=df.index)
            if rule['type'] == "Null Check": mask = df[rule['col']].isnull()
            elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max'])
            elif rule['type'] == "Relational Check":
                a = df[rule['col_a']]
                b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                op = rule['op']
                if op == ">": mask = a <= b
                elif op == "<": mask = a >= b
                elif op == "==": mask = a != b
                elif op == "!=": mask = a == b
                elif op == ">=": mask = a < b
                elif op == "<=": mask = a > b
            total_violations += mask.sum()
        except: pass

    m_col1.metric("Health", f"{health}%")
    m_col2.metric("Rows", f"{len(df):,}")
    m_col3.metric("Columns", f"{len(df.columns)}")
    m_col4.metric("Duplicates", f"{duplicate_rows:,}")
    m_col5.metric("Violations", f"{total_violations:,}")
    m_col6.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

    st.divider()

    o_col1, o_col2 = st.columns(2)

    with o_col1:
        st.subheader("Dataset Composition")
        type_series = df.dtypes.astype(str)
        type_counts = type_series.value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']

        fig = px.pie(type_counts, names='Data Type', values='Count', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    with o_col2:
        st.subheader("Workspace Status")
        active_rules_list = [r for r in st.session_state.rules if r.get('enabled', True)]
        active_feats = len(st.session_state.active_features)
        active_rules_count = len(active_rules_list)

        total_violations = 0
        for rule in active_rules_list:
            try:
                mask = pd.Series(False, index=df.index)
                if rule['type'] == "Null Check": mask = df[rule['col']].isnull()
                elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max'])
                elif rule['type'] == "Relational Check":
                    a = df[rule['col_a']]
                    b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                    op = rule['op']
                    if op == ">": mask = a <= b
                    elif op == "<": mask = a >= b
                    elif op == "==": mask = a != b
                    elif op == "!=": mask = a == b
                    elif op == ">=": mask = a < b
                    elif op == "<=": mask = a > b
                total_violations += mask.sum()
            except: pass

        st.markdown(f"""
        <div style="padding: 10px 0;">
            <ul style="font-size: 1.1em; line-height: 2.0; list-style-type: none; padding-left: 0;">
                <li>Tracked Features: <strong>{active_feats}</strong></li>
                <li>Active Rules: <strong>{active_rules_count}</strong></li>
                <li>Detected Violations: <strong>{total_violations}</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    selected_features = st.multiselect("Analyze Columns", all_cols, default=st.session_state.active_features)
    st.session_state.active_features = selected_features

    if not selected_features:
        st.info("Select one or more columns above to begin analysis")
    else:
        grid_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with grid_cols[idx % 2]:
                st.subheader(col_name)
                st.caption(f"Pandas Type: {df[col_name].dtype}")

                nulls = df[col_name].isnull().sum()
                uniques = df[col_name].nunique()

                s1, s2, s3 = st.columns(3)
                s1.metric("Nulls", f"{nulls}")
                s2.metric("Unique", f"{uniques}")

                if nulls > 0:
                    with st.expander("Clean Tools", expanded=False):
                        if st.button("Drop Null Rows", key=f"dr_{col_name}"):
                            st.session_state.raw_data = df.dropna(subset=[col_name])
                            st.rerun()
                        fv = st.text_input("Fill Value", key=f"fi_{col_name}")
                        if st.button("Apply Fill", key=f"af_{col_name}"):
                            st.session_state.raw_data[col_name] = df[col_name].fillna(fv)
                            st.rerun()

                if pd.api.types.is_numeric_dtype(df[col_name]):
                    s3.metric("Skew", f"{df[col_name].skew():.2f}")
                    fig = px.box(df, y=col_name, height=250)
                    st.plotly_chart(fig, width="stretch", theme="streamlit")
                else:
                    s3.metric("Type", "Obj")
                    counts = df[col_name].value_counts().head(5)
                    fig = px.bar(x=counts.index, y=counts.values, height=250)
                    st.plotly_chart(fig, width="stretch", theme="streamlit")
                st.divider()

with tab3:
    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("New Rule")
        rtype = st.selectbox("Type", ["Null Check", "Range Check", "Relational Check"])

        if rtype == "Relational Check":
            tcol = st.selectbox("Target Feature", all_cols, key="rel_rt_a")
            op = st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="], key="rel_rt_op")
            target_type = st.radio("Compare with", ["Feature", "Custom Value"], horizontal=True)

            if target_type == "Another Feature":
                col_b = st.selectbox("Feature B", all_cols, key="rel_rt_b")
                desc = f"{tcol} {op} {col_b}"
                if st.button("Add Relational Rule"):
                    add_rule({
                        "type": "Relational Check", "col_a": tcol, "op": op, "col_b": col_b,
                        "target_type": "Feature", "desc": desc, "enabled": True,
                        "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"
                    })
            else:
                val = st.text_input("Constant Value", key="rel_rt_val")
                desc = f"{tcol} {op} {val}"
                if st.button("Add Relational Rule"):
                    final_val = val
                    try: final_val = float(val)
                    except: pass
                    add_rule({
                        "type": "Relational Check", "col_a": tcol, "op": op, "value": final_val,
                        "target_type": "Value", "desc": desc, "enabled": True,
                        "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"
                    })
        else:
            tcol = st.selectbox("Target Column", all_cols, key="rt_single")
            if rtype == "Range Check" and pd.api.types.is_numeric_dtype(df[tcol]):
                c_min, c_max = float(df[tcol].min()), float(df[tcol].max())
                st.caption(f"Actual Data Range: [{c_min:.2f}, {c_max:.2f}]")
                num_col1, num_col2 = st.columns(2)
                v_min = num_col1.number_input("Min Bound", value=c_min)
                v_max = num_col2.number_input("Max Bound", value=c_max)
                desc = f"{tcol} in [{v_min:.1f}, {v_max:.1f}]"
                if st.button("Add Range Rule"):
                    add_rule({
                        "type": "Range Check", "col": tcol, "min": v_min, "max": v_max,
                        "desc": desc, "enabled": True,
                        "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"
                    })
            elif rtype == "Null Check":
                desc = f"{tcol} is NOT NULL"
                if st.button("Add Null Rule"):
                    add_rule({
                        "type": "Null Check", "col": tcol, "desc": desc, "enabled": True,
                        "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"
                    })

    with r2:
        rh_col1, rh_col2 = st.columns([2, 1])
        rh_col1.subheader("Active Rules")
        if st.session_state.rules:
            if rh_col2.button("Clear All", width="stretch"):
                st.session_state.rules = []
                st.rerun()

        if not st.session_state.rules:
            st.info("Add a rule from the left panel to begin monitoring violations")
        else:
            # Dynamic height: grow progressively, scroll after 3 rules
            container_args = {"border": False}
            if len(st.session_state.rules) > 3:
                container_args["height"] = 600

            with st.container(**container_args):
                seen_descriptions = []
                for idx, rule in enumerate(st.session_state.rules):
                    mask = pd.Series(False, index=df.index)
                    if rule['enabled']:
                        try:
                            if rule['type'] == "Null Check": mask = df[rule['col']].isnull()
                            elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max'])
                            elif rule['type'] == "Relational Check":
                                a = df[rule['col_a']]
                                b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                                op = rule['op']
                                if op == ">": mask = a <= b
                                elif op == "<": mask = a >= b
                                elif op == "==": mask = a != b
                                elif op == "!=": mask = a == b
                                elif op == ">=": mask = a < b
                                elif op == "<=": mask = a > b
                        except: pass

                    v_count = mask.sum()
                    status_color = rule['color'] if rule['enabled'] else "rgba(100, 100, 100, 0.2)"
                    desc_style = "text-decoration: line-through; opacity: 0.5;" if not rule['enabled'] else ""

                    is_duplicate = rule['desc'] in seen_descriptions
                    seen_descriptions.append(rule['desc'])

                    st.markdown(f'<div class="violation-card">', unsafe_allow_html=True)
                    if is_duplicate:
                        st.markdown('<p style="color: #FFD700; font-weight: bold; font-size: 0.85em; margin: 0;">Warning: Duplicate logic</p>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="border-left: 8px solid {status_color}; padding-left: 15px; margin: 10px 0;">
                        <strong style="font-size: 1.0em; {desc_style}">{rule['type']}</strong><br/>
                        <code style="{desc_style}">{rule['desc']}</code><br/>
                        <span style="font-size: 0.85em; opacity: 0.7;">Violations: {v_count}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{idx}", width="stretch"):
                        toggle_rule(idx)
                        st.rerun()
                    if btn_c2.button("Remove", key=f"del_{idx}", width="stretch"):
                        delete_rule(idx)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    sf, sr = st.text_input("Find"), st.text_input("Replace")
    target = st.selectbox("Columns", ["All"] + all_cols)
    if st.button("Execute"):
        if target == "All": st.session_state.raw_data = df.replace(sf, sr)
        else: st.session_state.raw_data[target] = df[target].replace(sf, sr)
        st.rerun()

# --- LIVE VIEW ---
st.divider()
st.subheader("Live Lab")

def get_styles(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for rule in st.session_state.rules:
        if not rule['enabled']: continue
        try:
            if rule['type'] == "Null Check": mask = df[rule['col']].isnull(); style_df.loc[mask, rule['col']] = f"background-color: {rule['color']};"
            elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max']); style_df.loc[mask, rule['col']] = f"background-color: {rule['color']};"
            elif rule['type'] == "Relational Check":
                a = df[rule['col_a']]
                b = df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                op = rule['op']
                t_cols = [rule['col_a']]
                if rule.get('target_type') == 'Feature': t_cols.append(rule['col_b'])
                if op == ">": mask = a <= b
                elif op == "<": mask = a >= b
                elif op == "==": mask = a != b
                elif op == "!=": mask = a == b
                elif op == ">=": mask = a < b
                elif op == "<=": mask = a > b
                style_df.loc[mask, t_cols] = f"background-color: {rule['color']};"
        except: continue
    return style_df

st.caption("Heatmap View (Read-Only)")
st.dataframe(df.style.apply(lambda _: get_styles(df), axis=None), width="stretch")

st.divider()
st.caption("Editor (Manual)")
edited = st.data_editor(df, num_rows="dynamic", width="stretch", on_change=data_edited, key="ed")
if not edited.equals(df):
    st.session_state.raw_data = edited
    st.rerun()
