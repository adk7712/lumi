import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib

# Set page config
st.set_page_config(page_title="Lumi Workspace", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
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
        padding: 15px 0;
        padding-right: 20px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        margin-bottom: 10px;
        position: relative;
    }

    /* Tab Label Spacing */
    .stTabs [data-baseweb="tab-list"] button {
        margin-right: 15px !important;
    }

    /* Bigger Tab Labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p,
    button[data-baseweb="tab"] p,
    button[role="tab"] p,
    button[role="tab"] span,
    button[data-testid="stBaseButton-tab"] p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Recipe Timeline */
    .recipe-step {
        padding: 8px 12px;
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 5px;
        margin-bottom: 5px;
        border-left: 4px solid #4F8BF9;
    }

    /* Proposal Box */
    .proposal-box {
        padding: 12px;
        background-color: rgba(241, 196, 15, 0.05);
        border: 1px dashed #f1c40f;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def get_safe_hue(n):
    # Avoids Red (0-60, 340-360) and Green (80-160)
    # Palette of Blue, Purple, Cyan, and Deep Pink
    safe_hues = [200, 240, 280, 310, 185, 220, 260, 300, 330]
    return safe_hues[n % len(safe_hues)]

# --- INTELLIGENCE ENGINE (AUTO-SCOUT) ---
def generate_proposals(df):
    proposals = []
    for col in df.columns:
        if col in st.session_state.scanned_columns: continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            proposals.append({
                "type": "Null Check", "column": col, "reason": f"{null_pct:.1f}% missing values",
                "rule_data": {"type": "Null Check", "col": col, "desc": f"{col} is NOT NULL"}
            })
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            l, u = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = len(df[(df[col] < l) | (df[col] > u)])
            if outliers > 0:
                proposals.append({
                    "type": "Range Check", "column": col, "reason": f"{outliers} statistical outliers",
                    "rule_data": {"type": "Range Check", "col": col, "min": float(l), "max": float(u), "desc": f"{col} within statistical bounds"}
                })
        elif df[col].dtype == 'object':
            non_nulls = df[col].dropna()
            if len(non_nulls) > 0:
                num_pct = pd.to_numeric(non_nulls, errors='coerce').notnull().mean()
                if 0.60 <= num_pct < 0.99:
                    proposals.append({
                        "type": "Type Cast", "column": col, "reason": f"Mixed types ({num_pct:.1%} numeric)",
                        "rule_data": {"action": "cast_type", "column": col, "dtype": "float64"}
                    })
    return proposals

# --- STATE INITIALIZATION ---
@st.cache_data
def load_data(file_buffer):
    try:
        df = pd.read_csv(file_buffer)
        if len(df) > 10000:
            df = df.sample(10000, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = load_data("mock_data/train.csv")
if 'active_features' not in st.session_state:
    st.session_state.active_features = []
if 'rules' not in st.session_state:
    st.session_state.rules = []
if 'cleaning_recipe' not in st.session_state:
    st.session_state.cleaning_recipe = []
if 'proposals' not in st.session_state:
    st.session_state.proposals = []
if 'scanned_columns' not in st.session_state:
    st.session_state.scanned_columns = set()
if 'last_file_hash' not in st.session_state:
    st.session_state.last_file_hash = None

# One-time scout trigger for initial data
if not st.session_state.proposals and not st.session_state.raw_data.empty:
    st.session_state.proposals = generate_proposals(st.session_state.raw_data)

# --- PIPELINE ENGINE ---
def apply_recipe(df, recipe):
    df_clean = df.copy()
    for step in recipe:
        try:
            action, col = step['action'], step.get('column')
            if action == "drop_column": df_clean = df_clean.drop(columns=[col])
            elif action == "drop_nulls": df_clean = df_clean.dropna(subset=[col])
            elif action == "fill_null":
                val = step['value']
                if val == "mean": fill = df_clean[col].mean()
                elif val == "median": fill = df_clean[col].median()
                elif val == "mode": fill = df_clean[col].mode()[0]
                else: fill = val
                df_clean[col] = df_clean[col].fillna(fill)
            elif action == "cap_range":
                df_clean.loc[df_clean[col] < step['min'], col] = step['min']
                df_clean.loc[df_clean[col] > step['max'], col] = step['max']
            elif action == "cast_type":
                if step['dtype'] == "datetime64[ns]": df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                else: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(step['dtype'], errors='ignore')
            elif action == "drop_violated":
                r = step['rule']
                if r['type'] == "Null Check": df_clean = df_clean.dropna(subset=[r['col']])
                elif r['type'] == "Range Check": df_clean = df_clean[(df_clean[r['col']] >= r['min']) & (df_clean[r['col']] <= r['max'])]
                elif r['type'] == "Relational Check":
                    a, b = df_clean[r['col_a']], df_clean[r['col_b']] if r.get('target_type') == 'Feature' else r['value']
                    op = r['op']
                    if op == ">": mask = a > b
                    elif op == "<": mask = a < b
                    elif op == "==": mask = a == b
                    elif op == "!=": mask = a != b
                    elif op == ">=": mask = a >= b
                    elif op == "<=": mask = a <= b
                    df_clean = df_clean[mask]
                elif r['type'] == "Custom Expression": df_clean = df_clean.query(r['query'])
            elif action == "replace":
                f, r = step['find'], step['replace']
                if col == "All": df_clean = df_clean.replace(f, r)
                else: df_clean[col] = df_clean[col].replace(f, r)
        except: continue
    return df_clean

def generate_pipeline_code(recipe):
    code = ["import pandas as pd\nimport numpy as np\n", "def clean_data(df):"]
    if not recipe: code.append("    # No cleaning steps applied")
    else:
        for step in recipe:
            action, col = step['action'], step.get('column')
            if action == "drop_column": code.append(f"    df = df.drop(columns=['{col}'])")
            elif action == "drop_nulls": code.append(f"    df = df.dropna(subset=['{col}'])")
            elif action == "fill_null":
                v = step['value']
                if v in ["mean", "median", "mode"]: code.append(f"    df['{col}'] = df['{col}'].fillna(df['{col}'].{v + ('()[0]' if v=='mode' else '()')})")
                else: code.append(f"    df['{col}'] = df['{col}'].fillna({f'\"{v}\"' if isinstance(v, str) else v})")
            elif action == "cap_range":
                code.append(f"    df.loc[df['{col}'] < {step['min']}, '{col}'] = {step['min']}")
                code.append(f"    df.loc[df['{col}'] > {step['max']}, '{col}'] = {step['max']}")
            elif action == "cast_type":
                if step['dtype'] == "datetime64[ns]": code.append(f"    df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
                else: code.append(f"    df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('{step['dtype']}')")
            elif action == "drop_violated":
                r = step['rule']
                if r['type'] == "Null Check": code.append(f"    df = df.dropna(subset=['{r['col']}'])")
                elif r['type'] == "Range Check": code.append(f"    df = df[(df['{r['col']}'] >= {r['min']}) & (df['{r['col']}'] <= {r['max']})]")
                elif r['type'] == "Custom Expression": code.append(f"    df = df.query('{r['query']}')")
            elif action == "replace":
                if col == "All": code.append(f"    df = df.replace('{step['find']}', '{step['replace']}')")
                else: code.append(f"    df['{col}'] = df['{col}'].replace('{step['find']}', '{step['replace']}')")
    code.append("    return df")
    return "\n".join(code)

# --- CALLBACKS ---
def add_step(step):
    st.session_state.cleaning_recipe.append(step)
    st.toast(f"Step Added: {step['action']}")

# --- HEADER ---
h_col1, h_col2, h_col3 = st.columns([2, 3, 1])
h_col1.title("LUMI")
with h_col2:
    uploaded_file = st.file_uploader("Upload Source", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.last_file_hash != file_hash:
            st.session_state.raw_data = load_data(uploaded_file)
            st.session_state.last_file_hash = file_hash
            st.session_state.scanned_columns, st.session_state.cleaning_recipe = set(), []
            st.session_state.proposals = generate_proposals(st.session_state.raw_data)
            st.toast("Dataset Analyzed")
with h_col3:
    if st.button("SYNC", width="stretch"):
        st.session_state.active_features, st.session_state.rules, st.session_state.cleaning_recipe = [], [], []
        st.session_state.scanned_columns = set()
        st.session_state.raw_data = load_data("mock_data/train.csv")
        st.session_state.proposals = generate_proposals(st.session_state.raw_data)
        st.rerun()

st.divider()

# --- DATA PROCESSING ---
df_raw = st.session_state.raw_data
df = apply_recipe(df_raw, st.session_state.cleaning_recipe)
all_cols = df.columns.tolist()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Diagnostics", "Rulebook", "Find and Replace", "Audit Log", "Pipeline Preview"])

with tab1:
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    health = int((1 - (null_cells / total_cells)) * 100) if total_cells > 0 else 0
    active_rules_list = [r for r in st.session_state.rules if r.get('enabled', True)]
    total_violations = 0
    for rule in active_rules_list:
        try:
            mask = pd.Series(False, index=df.index)
            if rule['type'] == "Null Check": mask = df[rule['col']].isnull()
            elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max'])
            elif rule['type'] == "Relational Check":
                a, b = df[rule['col_a']], df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                if rule['op'] == ">": mask = a <= b
                elif rule['op'] == "<": mask = a >= b
                elif rule['op'] == "==": mask = a != b
                elif rule['op'] == "!=": mask = a == b
                elif rule['op'] == ">=": mask = a < b
                elif rule['op'] == "<=": mask = a > b
            elif rule['type'] == "Custom Expression": mask = ~df.index.isin(df.query(rule['query']).index)
            total_violations += mask.sum()
        except: pass
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
    st.divider()
    st.subheader("Feature Correlation")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r'), width="stretch", theme="streamlit")
    else: st.caption("Not enough numeric columns for correlation matrix.")

with tab2:
    selected_features = st.multiselect("Analyze Columns", all_cols, default=st.session_state.active_features)
    st.session_state.active_features = selected_features
    if not selected_features: st.info("Select one or more columns above to begin analysis")
    else:
        grid_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with grid_cols[idx % 2]:
                st.subheader(col_name)
                s1, s2, s3 = st.columns(3)
                s1.metric("Nulls", f"{df[col_name].isnull().sum()}")
                s2.metric("Unique", f"{df[col_name].nunique()}")
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    s3.metric("Skew", f"{df[col_name].skew():.2f}")
                    st.plotly_chart(px.box(df, y=col_name, height=220), width="stretch")
                else:
                    s3.metric("Type", "Obj")
                    st.plotly_chart(px.bar(x=df[col_name].value_counts().head(5).index, y=df[col_name].value_counts().head(5).values, height=220), width="stretch")

with tab3:
    if st.session_state.proposals:
        st.subheader("Recommended Rules")
        p_cols = st.columns(2)
        for p_idx, p in enumerate(st.session_state.proposals):
            with p_cols[p_idx % 2]:
                st.markdown(f'<div class="proposal-box"><strong>{p["type"]} on {p["column"]}</strong><br/><small>{p["reason"]}</small></div>', unsafe_allow_html=True)
                acc, dis = st.columns(2)
                if acc.button("Accept", key=f"p_acc_{p_idx}", width="stretch"):
                    st.session_state.scanned_columns.add(p['column'])
                    if 'action' in p['rule_data']: add_step(p['rule_data'])
                    else:
                        rule = p['rule_data'].copy()
                        rule.update({'enabled': True, 'color': f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                        st.session_state.rules.append(rule)
                    st.session_state.proposals.pop(p_idx); st.rerun()
                if dis.button("Dismiss", key=f"p_dis_{p_idx}", width="stretch"):
                    st.session_state.scanned_columns.add(p['column']); st.session_state.proposals.pop(p_idx); st.rerun()
        st.divider()
    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("New Rule")
        rtype = st.selectbox("Type", ["Null Check", "Range Check", "Relational Check", "Custom Expression"])
        if rtype == "Custom Expression":
            q_str = st.text_input("Pandas Query String", placeholder="Age > 30 & Sex == 'male'")
            if st.button("Add Rule"): st.session_state.rules.append({"type": "Custom Expression", "query": q_str, "desc": f"Matches: {q_str}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
        elif rtype == "Relational Check":
            tcol, op = st.selectbox("Feature A", all_cols), st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="])
            target_type = st.radio("Compare with", ["Another Feature", "Constant Value"], horizontal=True)
            if target_type == "Another Feature":
                col_b = st.selectbox("Feature B", all_cols)
                if st.button("Add Rule"): st.session_state.rules.append({"type": "Relational Check", "col_a": tcol, "op": op, "col_b": col_b, "target_type": "Feature", "desc": f"{tcol} {op} {col_b}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
            else:
                val = st.text_input("Constant Value")
                if st.button("Add Rule"):
                    try: final_val = float(val)
                    except: final_val = val
                    st.session_state.rules.append({"type": "Relational Check", "col_a": tcol, "op": op, "value": final_val, "target_type": "Value", "desc": f"{tcol} {op} {val}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
        else:
            tcol = st.selectbox("Target Column", all_cols)
            if rtype == "Range Check" and pd.api.types.is_numeric_dtype(df[tcol]):
                num_col1, num_col2 = st.columns(2)
                v_min, v_max = num_col1.number_input("Min", value=float(df[tcol].min())), num_col2.number_input("Max", value=float(df[tcol].max()))
                if st.button("Add Rule"): st.session_state.rules.append({"type": "Range Check", "col": tcol, "min": v_min, "max": v_max, "desc": f"{tcol} in [{v_min}, {v_max}]", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
            elif rtype == "Null Check":
                if st.button("Add Rule"): st.session_state.rules.append({"type": "Null Check", "col": tcol, "desc": f"{tcol} is NOT NULL", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
    with r2:
        rh1, rh2 = st.columns([2, 1])
        rh1.subheader("Active Rules")
        if st.session_state.rules and rh2.button("Clear All", width="stretch"): st.session_state.rules, st.session_state.cleaning_recipe = [], []; st.rerun()
        if not st.session_state.rules: st.info("Add a rule from the left panel")
        else:
            with st.container(height=600, border=False):
                for idx, rule in enumerate(st.session_state.rules):
                    mask = pd.Series(False, index=df.index)
                    if rule['enabled']:
                        try:
                            if rule['type'] == "Null Check": mask = df[rule['col']].isnull()
                            elif rule['type'] == "Range Check": mask = (df[rule['col']] < rule['min']) | (df[rule['col']] > rule['max'])
                            elif rule['type'] == "Relational Check":
                                a, b = df[rule['col_a']], df[rule['col_b']] if rule.get('target_type') == 'Feature' else rule['value']
                                op = rule['op']
                                if op == ">": mask = a <= b
                                elif op == "<": mask = a >= b
                                elif op == "==": mask = a != b
                                elif op == "!=": mask = a == b
                                elif op == ">=": mask = a < b
                                elif op == "<=": mask = a > b
                            elif rule['type'] == "Custom Expression": mask = ~df.index.isin(df.query(rule['query']).index)
                        except: pass
                    v_count, status_color, resolved = mask.sum(), (rule['color'] if rule['enabled'] else "rgba(100,100,100,0.2)"), rule.get('resolved', False)
                    desc_style = "text-decoration: line-through; opacity: 0.5;" if not rule['enabled'] else ""
                    st.markdown(f'<div class="violation-card"><div style="border-left: 8px solid {status_color}; padding-left: 15px;"><strong style="{desc_style}">{rule["type"]}</strong><br/><code style="color: #4F8BF9; {desc_style}">{rule["desc"]}</code><br/><span style="font-size: 0.85em; opacity: 0.7;">Violations: {v_count}</span>{f"<br/><span style=\"color: #2ecc71; font-size: 0.85em; font-weight: bold;\">Status: Resolved</span>" if resolved else ""}</div></div>', unsafe_allow_html=True)
                    if v_count > 0 and not resolved:
                        if rule['type'] == "Null Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Resolution", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"], key=f"res_{idx}", label_visibility="collapsed")
                            if res != "None" and res_cols[1].button("Apply", key=f"btn_{idx}", width="stretch"):
                                if res == "Drop Rows": add_step({"action": "drop_nulls", "column": rule['col']})
                                else: add_step({"action": "fill_null", "column": rule['col'], "value": res.split()[-1].lower()})
                                st.session_state.rules[idx]['resolved'] = True; st.rerun()
                        elif rule['type'] == "Range Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Res", ["None", "Drop Rows", "Cap at Bounds"], key=f"res_{idx}", label_visibility="collapsed")
                            if res != "None" and res_cols[1].button("Apply", key=f"btn_{idx}", width="stretch"):
                                if res == "Drop Rows": add_step({"action": "drop_violated", "rule": rule})
                                else: add_step({"action": "cap_range", "column": rule['col'], "min": rule['min'], "max": rule['max']})
                                st.session_state.rules[idx]['resolved'] = True; st.rerun()
                        else:
                            if st.button("Drop Violated Rows", key=f"res_{idx}", width="stretch"): add_step({"action": "drop_violated", "rule": rule}); st.session_state.rules[idx]['resolved'] = True; st.rerun()
                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{idx}", width="stretch"): st.session_state.rules[idx]['enabled'] = not rule['enabled']; st.rerun()
                    if btn_c2.button("Remove", key=f"del_{idx}", width="stretch"): st.session_state.rules.pop(idx); st.rerun()

with tab4:
    st.subheader("Manual Transformations")
    t_type = st.selectbox("Type", ["Find and Replace", "Cast Data Type", "Drop Column"])
    if t_type == "Find and Replace":
        c1, c2, c3 = st.columns(3)
        sf, sr, target = c1.text_input("Find"), c2.text_input("Replace"), c3.selectbox("Columns", ["All"] + all_cols)
        if st.button("Add Step", key="btn_fr"): add_step({"action": "replace", "column": target, "find": sf, "replace": sr}); st.rerun()
    elif t_type == "Cast Data Type":
        c1, c2 = st.columns(2)
        target, dtype_t = c1.selectbox("Column", all_cols), c2.selectbox("Cast To", ["string", "float64", "int64", "datetime64[ns]"])
        if st.button("Add Step", key="btn_cast"): add_step({"action": "cast_type", "column": target, "dtype": dtype_t}); st.rerun()
    elif t_type == "Drop Column":
        target = st.selectbox("Target Column", all_cols)
        if st.button("Add Step", key="btn_drop"): add_step({"action": "drop_column", "column": target}); st.rerun()

with tab5:
    st.subheader("Data Lineage")
    if not st.session_state.cleaning_recipe: st.caption("No transformations applied yet")
    else:
        bh = int((1 - (df_raw.isnull().sum().sum() / df_raw.size)) * 100) if df_raw.size > 0 else 0
        st.markdown(f'<div class="recipe-step"><strong>0. Original Data</strong> | Health: {bh}% | Rows: {len(df_raw):,}</div>', unsafe_allow_html=True)
        for i, step in enumerate(st.session_state.cleaning_recipe):
            c1, c2 = st.columns([4, 1])
            tdf = apply_recipe(df_raw, st.session_state.cleaning_recipe[:i+1])
            th = int((1 - (tdf.isnull().sum().sum() / tdf.size)) * 100) if tdf.size > 0 else 0
            c1.markdown(f'<div class="recipe-step">{i+1}. {step["action"]} on {step.get("column", "dataset")} | Health: {th}% | Rows: {len(tdf):,}</div>', unsafe_allow_html=True)
            if c2.button("Remove", key=f"rm_step_{i}", width="stretch"): st.session_state.cleaning_recipe.pop(i); st.rerun()

with tab6:
    v = st.radio("Mode", ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"], horizontal=True, key="p_mode")
    if v == "Raw Data (Before)": st.dataframe(df_raw, width="stretch")
    elif v == "Cleaned Data (After)": st.dataframe(df, width="stretch")
    else:
        st.code(generate_pipeline_code(st.session_state.cleaning_recipe), language="python")
        st.download_button("Download clean_data.py", generate_pipeline_code(st.session_state.cleaning_recipe), "clean_data.py", "text/x-python", width="stretch")

st.divider(); st.subheader("Validation Heatmap")
def get_styles(df_d):
    sdf = pd.DataFrame('', index=df_d.index, columns=df_d.columns)
    for r in st.session_state.rules:
        if not r['enabled']: continue
        try:
            if r['type'] == "Null Check": sdf.loc[df_d[r['col']].isnull(), r['col']] = f"background-color: {r['color']};"
            elif r['type'] == "Range Check": sdf.loc[(df_d[r['col']] < r['min']) | (df_d[r['col']] > r['max']), r['col']] = f"background-color: {r['color']};"
            elif r['type'] == "Custom Expression": sdf.loc[~df_d.index.isin(df_d.query(r['query']).index), :] = f"background-color: {r['color']};"
        except: continue
    return sdf
st.dataframe(df.style.apply(lambda _: get_styles(df), axis=None), width="stretch")
