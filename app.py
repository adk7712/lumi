import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from pathlib import Path

from engine import apply_recipe, generate_pipeline_code
from scout import generate_proposals
from ui_utils import inject_custom_css, get_safe_hue, get_heatmap_styles
from rule_utils import evaluate_rule

# Define constants
MAX_SAMPLE_ROWS = 10000

# Set page config
st.set_page_config(page_title="Lumi", layout="wide")

# Inject Custom CSS
inject_custom_css(st)

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path_or_buffer):
    """Loads data from a file path or buffer, supporting CSV and Excel."""
    try:
        # Streamlit's UploadedFile object has a 'type' attribute
        if hasattr(file_path_or_buffer, 'type'):
            file_type = file_path_or_buffer.type
            if file_type == "text/csv":
                return pd.read_csv(file_path_or_buffer)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(file_path_or_buffer)
        # For local file paths
        elif isinstance(file_path_or_buffer, str):
            suffix = Path(file_path_or_buffer).suffix.lower()
            if suffix == '.csv':
                return pd.read_csv(file_path_or_buffer)
            elif suffix == '.xlsx':
                return pd.read_excel(file_path_or_buffer)

        # Fallback for buffers without a clear type
        st.warning("Could not determine file type, attempting to read as CSV. May fail for other formats.")
        return pd.read_csv(file_path_or_buffer)

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.error(f"Error loading data: {type(e).__name__} - {e}. Please check the file format and content.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
        return pd.DataFrame()
# --- STATE INITIALIZATION ---
def initialize_state(from_reset=False):
    """Initializes all required session state variables."""
    # Streamlit's session state is crucial for maintaining application state across user interactions.
    # Define default values for the session state
    defaults = {
        'active_features': [],
        'rules': [],
        'cleaning_recipe': [],
        'proposals': [],
        'scanned_columns': set(),
        'last_file_hash': None,
        'raw_data': None,
        'original_full_data': None,
    }

    # Force reset or initialize for the first time
    # Persist state across reruns, or reset if 'from_reset' is True or key is new.
    for key, value in defaults.items():
        if from_reset or key not in st.session_state:
            st.session_state[key] = value

    # Load initial data if it's not already loaded
    if st.session_state.raw_data is None or from_reset:
        raw_df = load_data("mock_data/train.csv")
        st.session_state.original_full_data = raw_df
        # Sample large datasets for interactive use to improve performance.
        # A fixed random_state ensures reproducibility of the sample.
        if len(raw_df) > MAX_SAMPLE_ROWS:
            st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        else:
            st.session_state.raw_data = raw_df

        st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)

initialize_state()

# --- CALLBACKS ---
def add_step(step):
    """Adds a cleaning step to the recipe and shows a toast."""
    st.session_state.cleaning_recipe.append(step)
    st.toast(f"Step Added: {step['action']}")

# --- HEADER ---
h_col1, h_col2, h_col3 = st.columns([8, 1.5, 1], vertical_alignment="bottom")
with h_col1: # Main column for title
    st.subheader("LUMI")
with h_col2: # Column for file uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], label_visibility="collapsed", key="global_uploader")
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.last_file_hash != file_hash:
            raw_df = load_data(uploaded_file)
            st.session_state.original_full_data = raw_df
            if len(raw_df) > MAX_SAMPLE_ROWS:
                st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
            else:
                st.session_state.raw_data = raw_df

            st.session_state.last_file_hash = file_hash
            # Reset dependent state
            st.session_state.scanned_columns, st.session_state.cleaning_recipe, st.session_state.rules = set(), [], []
            st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
            st.toast("Dataset Analyzed")
            st.rerun()

with h_col3: # Column for reset button
    if st.button("Reset", key="reset_all", use_container_width=True):
        initialize_state(from_reset=True)
        st.rerun()

st.divider()

# --- DATA PROCESSING ---
df_raw = st.session_state.raw_data
df, apply_messages = apply_recipe(df_raw, st.session_state.cleaning_recipe)
for msg in apply_messages:
    st.toast(msg)
all_cols = df.columns.tolist()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Diagnostics", "Rulebook", "Find and Replace", "Audit Log", "Pipeline Preview"])

with tab1:
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
        fig = px.pie(type_counts, names='Data Type', values='Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel) # Using a pleasant, qualitative color palette for distinct categories.
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with o_col2:
        st.subheader("Workspace Status")
        st.markdown(f"**Recipe Steps:** {len(st.session_state.cleaning_recipe)}  \n**Tracked Features:** {len(st.session_state.active_features)}  \n**Active Rules:** {len(active_rules_list)}")
    st.divider()

    st.subheader("Feature Correlation")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r'), use_container_width=True, theme="streamlit")
    else:
        st.caption("Not enough numeric columns for correlation matrix.")

with tab2:
    st.multiselect("Analyze Columns", all_cols, key="active_features")
    selected_features = st.session_state.active_features

    if not selected_features:
        st.info("Select one or more columns above to begin analysis")
    else:
        grid_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with grid_cols[idx % 2]:
                st.subheader(col_name)
                s1, s2, s3 = st.columns(3)
                s1.metric("Nulls", f"{df[col_name].isnull().sum()}")
                s2.metric("Unique", f"{df[col_name].nunique()}")
                # Differentiate plotting and metric display based on data type for relevant insights.
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    s3.metric("Skew", f"{df[col_name].skew():.2f}")
                    st.plotly_chart(px.box(df, y=col_name, height=220), use_container_width=True)
                else:
                    s3.metric("Type", "Obj")
                    st.plotly_chart(px.bar(x=df[col_name].value_counts().head(5).index, y=df[col_name].value_counts().head(5).values, height=220), use_container_width=True)

with tab3:
    if st.session_state.proposals:
        with st.expander(f"Recommended Rules ({len(st.session_state.proposals)})", expanded=False):
            p_cols = st.columns(2)
            for p_idx, p in enumerate(st.session_state.proposals):
                with p_cols[p_idx % 2]:
                    st.markdown(f'<div class="proposal-box"><strong>{p["type"]} on {p["column"]}</strong><br/><small>{p["reason"]}</small></div>', unsafe_allow_html=True)
                    acc, dis = st.columns(2)
                    if acc.button("Accept", key=f"p_acc_{p_idx}", use_container_width=True):
                        st.session_state.scanned_columns.add(p['column'])
                        if 'action' in p['rule_data']:
                            add_step(p['rule_data'])
                        else:
                            rule = p['rule_data'].copy()
                            rule.update({'enabled': True, 'color': f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                            st.session_state.rules.append(rule)
                        st.session_state.proposals.pop(p_idx)
                        st.rerun()
                    if dis.button("Dismiss", key=f"p_dis_{p_idx}", use_container_width=True):
                        st.session_state.scanned_columns.add(p['column'])
                        st.session_state.proposals.pop(p_idx)
                        st.rerun()
        st.divider()

    r1, r2 = st.columns([1, 1])
    with r1:
        st.subheader("New Rule")
        rtype = st.selectbox("Type", ["Null Check", "Range Check", "Relational Check", "Custom Expression", "Informational"], key="rule_type_select")
        if rtype == "Informational":
            note = st.text_area("Note/Warning", placeholder="e.g., This column contains high cardinality data.", key="info_note_input")
            if st.button("Add Rule", key="btn_add_info"):
                st.session_state.rules.append({"type": "Informational", "desc": note, "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
        elif rtype == "Custom Expression":
            q_str = st.text_input("Pandas Query String", placeholder="Age > 30 & Sex == 'male'", key="custom_query_input")
            if st.button("Add Rule", key="btn_add_custom"):
                st.session_state.rules.append({"type": "Custom Expression", "query": q_str, "desc": f"Matches: {q_str}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
        elif rtype == "Relational Check":
            tcol, op = st.selectbox("Feature A", all_cols, key="rel_feature_a"), st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="], key="rel_op")
            target_type = st.radio("Compare with", ["Another Feature", "Constant Value"], horizontal=True, key="rel_target_type_radio")
            if target_type == "Another Feature":
                col_b = st.selectbox("Feature B", all_cols, key="rel_feature_b")
                if st.button("Add Rule", key="btn_add_rel_feat"):
                    st.session_state.rules.append({"type": "Relational Check", "col_a": tcol, "op": op, "col_b": col_b, "target_type": "Feature", "desc": f"{tcol} {op} {col_b}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
            else:
                val = st.text_input("Constant Value", key="rel_val_input")
                if st.button("Add Rule", key="btn_add_rel_val"):
                    try: final_val = float(val)
                    except: final_val = val
                    st.session_state.rules.append({"type": "Relational Check", "col_a": tcol, "op": op, "value": final_val, "target_type": "Value", "desc": f"{tcol} {op} {val}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
        else:
            tcol = st.selectbox("Target Column", all_cols, key="rule_target_col")
            if rtype == "Range Check" and pd.api.types.is_numeric_dtype(df[tcol]):
                num_col1, num_col2 = st.columns(2)
                v_min, v_max = num_col1.number_input("Min", value=float(df[tcol].min()), key="range_min_input"), num_col2.number_input("Max", value=float(df[tcol].max()), key="range_max_input")
                if st.button("Add Rule", key="btn_add_range"):
                    st.session_state.rules.append({"type": "Range Check", "col": tcol, "min": v_min, "max": v_max, "desc": f"{tcol} in [{v_min}, {v_max}]", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
            elif rtype == "Null Check":
                if st.button("Add Rule", key="btn_add_null"):
                    st.session_state.rules.append({"type": "Null Check", "col": tcol, "desc": f"{tcol} is NOT NULL", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
    with r2:
        rh1, rh2 = st.columns([2, 1], vertical_alignment="bottom")
        rh1.subheader("Active Rules")
        if st.session_state.rules and rh2.button("Clear All", use_container_width=True, key="clear_all_rules_btn"):
            st.session_state.rules, st.session_state.cleaning_recipe = [], []
            st.rerun()

        if not st.session_state.rules:
            st.info("Add a rule from the left panel")
        else:
            with st.container(height=600, border=False):
                for idx, rule in enumerate(st.session_state.rules):
                    v_count = 0
                    if rule['enabled']:
                        try:
                            mask = evaluate_rule(df, rule)
                            v_count = mask.sum()
                            rule.pop('error', None)
                        except (ValueError, KeyError, TypeError) as e:
                            rule['error'] = str(e)
                            st.toast(f"Rulebook Error ({rule.get('desc', 'N/A')}): {type(e).__name__} - {str(e)}", icon="🚨")

                    status_color, resolved = (rule['color'] if rule['enabled'] else "rgba(100,100,100,0.2)"), rule.get('resolved', False)
                    desc_style = "text-decoration: line-through; opacity: 0.5;" if not rule['enabled'] else ""
                    error_html = f"<br/><span style='color: #e74c3c; font-size: 0.85em; font-weight: bold;'>⚠️ Error: {rule.get('error')}</span>" if 'error' in rule else ""

                    # For Informational rules, don't show violation count
                    v_text = f"Violations: {v_count}" if rule['type'] != "Informational" else "Type: Info"
                    st.markdown(f'<div class="violation-card"><div style="border-left: 8px solid {status_color}; padding-left: 15px;"><strong style="{desc_style}">{rule["type"]}</strong><br/><code style="color: #4F8BF9; {desc_style}">{rule["desc"]}</code><br/><span style="font-size: 0.85em; opacity: 0.7;">{v_text}</span>{f"<br/><span style=\"color: #2ecc71; font-size: 0.85em; font-weight: bold;\">Status: Resolved</span>" if resolved else ""}{error_html}</div></div>', unsafe_allow_html=True)

                    if v_count > 0 and not resolved and rule['type'] != "Informational":
                        if rule['type'] == "Null Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Resolution", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"], key=f"res_{idx}", label_visibility="collapsed")
                            if res != "None" and res_cols[1].button("Apply", key=f"btn_res_{idx}", use_container_width=True):
                                if res == "Drop Rows": add_step({"action": "drop_nulls", "column": rule['col']})
                                else: add_step({"action": "fill_null", "column": rule['col'], "value": res.split()[-1].lower()})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        elif rule['type'] == "Range Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Res", ["None", "Drop Rows", "Cap at Bounds"], key=f"range_res_{idx}", label_visibility="collapsed")
                            if res != "None" and res_cols[1].button("Apply", key=f"btn_range_res_{idx}", use_container_width=True):
                                if res == "Drop Rows": add_step({"action": "drop_violated", "rule": rule})
                                else: add_step({"action": "cap_range", "column": rule['col'], "min": rule['min'], "max": rule['max']})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        else:
                            if st.button("Drop Violated Rows", key=f"gen_res_{idx}", use_container_width=True):
                                add_step({"action": "drop_violated", "rule": rule})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()

                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{idx}", use_container_width=True):
                        st.session_state.rules[idx]['enabled'] = not rule['enabled']
                        st.rerun()
                    if btn_c2.button("Remove", key=f"del_{idx}", use_container_width=True):
                        st.session_state.rules.pop(idx)
                        st.rerun()

with tab4:
    st.subheader("Manual Transformations")
    t_type = st.selectbox("Type", ["Find and Replace", "Cast Data Type", "Drop Column"], key="trans_type_select")
    if t_type == "Find and Replace":
        c1, c2, c3 = st.columns(3)
        sf, sr, target = c1.text_input("Find", key="find_input"), c2.text_input("Replace", key="replace_input"), c3.selectbox("Columns", ["All"] + all_cols, key="replace_target_col")
        if st.button("Add Step", key="btn_fr"):
            add_step({"action": "replace", "column": target, "find": sf, "replace": sr})
            st.rerun()
    elif t_type == "Cast Data Type":
        c1, c2 = st.columns(2)
        target, dtype_t = c1.selectbox("Column", all_cols, key="cast_target_col"), c2.selectbox("Cast To", ["string", "float64", "int64", "datetime64[ns]"], key="cast_dtype_select")
        if st.button("Add Step", key="btn_cast"):
            add_step({"action": "cast_type", "column": target, "dtype": dtype_t})
            st.rerun()
    elif t_type == "Drop Column":
        target = st.selectbox("Target Column", all_cols, key="drop_target_col")
        if st.button("Add Step", key="btn_drop"):
            add_step({"action": "drop_column", "column": target})
            st.rerun()

with tab5:
    st.subheader("Data Lineage")
    if not st.session_state.cleaning_recipe:
        st.caption("No transformations applied yet")
    else:
        bh = int((1 - (df_raw.isnull().sum().sum() / df_raw.size)) * 100) if df_raw.size > 0 else 0
        st.markdown(f'<div class="recipe-step"><strong>0. Original Data</strong> | Health: {bh}% | Rows: {len(df_raw):,}</div>', unsafe_allow_html=True)

        # Efficiently generate audit trail
        # Create a copy of the raw DataFrame to apply steps iteratively for lineage visualization.
        temp_df = df_raw.copy()
        for i, step in enumerate(st.session_state.cleaning_recipe):
            c1, c2 = st.columns([4, 1])
            # Apply only the current step to the intermediate dataframe
            # Apply each cleaning step sequentially to show intermediate states.
            temp_df = apply_recipe(temp_df, [step])
            th = int((1 - (temp_df.isnull().sum().sum() / temp_df.size)) * 100) if temp_df.size > 0 else 0
            c1.markdown(f'<div class="recipe-step">{i+1}. {step["action"]} on {step.get("column", "dataset")} | Health: {th}% | Rows: {len(temp_df):,}</div>', unsafe_allow_html=True)
            if c2.button("Remove", key=f"rm_step_{i}", use_container_width=True):
                st.session_state.cleaning_recipe.pop(i)
                st.rerun()

with tab6:
    v = st.radio("Mode", ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"], horizontal=True, key="p_mode")
    if v == "Raw Data (Before)": st.dataframe(df_raw, use_container_width=True)
    elif v == "Cleaned Data (After)": st.dataframe(df, use_container_width=True)
    else:
        code_output = generate_pipeline_code(st.session_state.cleaning_recipe)
        st.code(code_output, language="python")
        st.download_button("Download clean_data.py", code_output, "clean_data.py", "text/x-python", use_container_width=True, key="download_pipeline_btn")

st.divider()
st.subheader("Validation Heatmap")
# Apply custom styles to highlight cells violating active rules.
heatmap_sdf, heatmap_messages = get_heatmap_styles(df, st.session_state.rules)
for msg in heatmap_messages:
    st.toast(msg)
st.dataframe(df.style.apply(lambda _: heatmap_sdf, axis=None), use_container_width=True)
