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
from streamlit_sortables import sort_items

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
    # 1. Load initial data first to get the columns list for default dropdown selections
    if 'raw_data' not in st.session_state or from_reset:
        raw_df = load_data("mock_data/train.csv")
        st.session_state.original_full_data = raw_df
        if len(raw_df) > MAX_SAMPLE_ROWS:
            st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        else:
            st.session_state.raw_data = raw_df

        base_df = st.session_state.raw_data
        bh = int((1 - (base_df.isnull().sum().sum() / base_df.size)) * 100) if base_df.size > 0 else 0
        st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]
        st.session_state.proposals = generate_proposals(st.session_state.raw_data, set())
        st.session_state.scanned_columns = set()

    df = st.session_state.raw_data
    all_cols = df.columns.tolist()
    first_col = all_cols[0] if all_cols else ""

    # 2. Define default values for the session state
    defaults = {
        'active_features': [],
        'rules': [],
        'cleaning_recipe': [],
        'intermediate_states': st.session_state.intermediate_states,
        'proposals': st.session_state.proposals,
        'scanned_columns': st.session_state.scanned_columns,
        'last_file_hash': None,
        'raw_data': st.session_state.raw_data,
        'original_full_data': st.session_state.original_full_data,

        # Transient UI Widget state initializers (prevents AppTest KeyErrors)
        'find_input': "",
        'replace_input': "",
        'replace_target_col': "All",
        'replace_use_regex': False,
        'rename_new_name_input': "",
        'rename_target_col': first_col,
        'norm_target_col': "All",
        'norm_method_select': "lowercase",
        'cast_target_col': first_col,
        'cast_dtype_select': "string",
        'drop_target_col': first_col,
        'strip_target_col': "All",
        'rule_target_col': first_col,
        'rule_type_select': "Null Check",
        'trans_type_select': "Find and Replace",
        'rel_feature_a': first_col,
        'rel_feature_b': first_col,
        'rel_op': ">",
        'rel_target_type_radio': "Another Feature",
        'rel_val_input': "",
        'info_note_input': "",
        'show_reorder_success': False,
    }

    # Force reset or initialize for the first time
    for key, value in defaults.items():
        if from_reset or key not in st.session_state:
            st.session_state[key] = value

initialize_state()

# --- CALLBACKS ---
def add_step(step):
    """Adds a cleaning step to the recipe, updates the cached state, and shows a toast."""
    st.session_state.cleaning_recipe.append(step)

    # Calculate delta state and cache it
    last_df = st.session_state.intermediate_states[-1][3]
    new_df, messages = apply_recipe(last_df, [step])
    for msg in messages:
        st.toast(msg)

    th = int((1 - (new_df.isnull().sum().sum() / new_df.size)) * 100) if new_df.size > 0 else 0
    step_desc = f"{step['action']} on {step.get('column', 'dataset')}"
    st.session_state.intermediate_states.append((step_desc, th, len(new_df), new_df))

    st.toast(f"Step Added: {step['action']}")

# --- HEADER ---
h_col1, h_col2, h_col3 = st.columns([6, 2, 2], vertical_alignment="bottom")
with h_col1: # Main column for title
    st.subheader("LUMI")
with h_col2: # Column for file uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], label_visibility="collapsed", key="global_uploader")
    if uploaded_file:
        # Optimization: Use file attributes for a lightweight identifier instead of full-file hashing
        # to prevent memory bottlenecks on large datasets.
        file_id = f"{uploaded_file.file_id}_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_file_hash != file_id:
            raw_df = load_data(uploaded_file)
            st.session_state.original_full_data = raw_df
            if len(raw_df) > MAX_SAMPLE_ROWS:
                st.session_state.raw_data = raw_df.sample(MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
            else:
                st.session_state.raw_data = raw_df

            st.session_state.last_file_hash = file_id
            # Reset dependent state
            st.session_state.active_features = []
            st.session_state.scanned_columns, st.session_state.cleaning_recipe, st.session_state.rules = set(), [], []

            # Reset intermediate states
            base_df = st.session_state.raw_data
            bh = int((1 - (base_df.isnull().sum().sum() / base_df.size)) * 100) if base_df.size > 0 else 0
            st.session_state.intermediate_states = [("Original Data", bh, len(base_df), base_df.copy())]

            st.session_state.proposals = generate_proposals(st.session_state.raw_data, st.session_state.scanned_columns)
            st.toast("Dataset Analyzed")
            st.rerun()

with h_col3: # Column for reset button
    u_c1, u_c2 = st.columns(2)
    if u_c1.button("Undo", key="undo_btn", width="stretch", disabled=len(st.session_state.cleaning_recipe) == 0):
        st.session_state.cleaning_recipe.pop()
        st.session_state.intermediate_states.pop()
        st.toast("Last step undone")
        st.rerun()
    if u_c2.button("Reset", key="reset_all", width="stretch"):
        initialize_state(from_reset=True)
        st.rerun()

st.divider()

# --- DATA PROCESSING ---
df = st.session_state.intermediate_states[-1][3]
all_cols = df.columns.tolist()

# --- TABS ---
tab1, tab2, tab_insights, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Diagnostics", "Visual Insights", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"])

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
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    with o_col2:
        st.subheader("Workspace Status")
        st.markdown(f"**Recipe Steps:** {len(st.session_state.cleaning_recipe)}  \n**Tracked Features:** {len(st.session_state.active_features)}  \n**Active Rules:** {len(active_rules_list)}")

with tab2:
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
                    from ui_utils import render_diagnostic_metric  # Add this import at top if not there
                    render_diagnostic_metric(s1, "Type", str(df[col_name].dtype))
                    render_diagnostic_metric(s2, "Nulls", f"{df[col_name].isnull().sum()}")
                    render_diagnostic_metric(s3, "Unique", f"{df[col_name].nunique()}")
                    # Differentiate plotting and metric display based on data type for relevant insights.
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        render_diagnostic_metric(s4, "Skew", f"{df[col_name].skew():.2f}")
                        fig = px.box(df, x=col_name, height=220)
                    else:
                        top_val = df[col_name].mode()[0] if not df[col_name].mode().empty else "N/A"
                        render_diagnostic_metric(s4, "Top", str(top_val)[:10])
                        counts = df[col_name].value_counts()
                        num_uniques = len(counts)
                        if num_uniques <= 10:
                            chart_data = counts
                        else:
                            top_n = counts.head(9)
                            other_sum = counts.iloc[9:].sum()
                            chart_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
                        fig = px.bar(x=chart_data.index, y=chart_data.values, height=220)

                    # Cleanup chart aesthetics by removing redundant axis labels and disabling hover
                    fig.update_layout(xaxis_title=None, yaxis_title=None, hovermode=False)
                    st.plotly_chart(fig, width="stretch", theme="streamlit")

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

with tab_insights:
    st.subheader("Visual Insights")

    # 1. Filtered Correlation Heatmap
    st.markdown("### Feature Correlation")
    st.markdown("Shows relationships between numeric columns. Features without any correlation within the selected range are filtered out.")
    corr_range = st.slider("Correlation Range", -1.0, 1.0, (-1.0, 1.0), 0.05, key="corr_range_val")

    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        corr_matrix_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_matrix_no_diag.values, np.nan)
        in_range_mask = (corr_matrix_no_diag >= corr_range[0]) & (corr_matrix_no_diag <= corr_range[1])
        correlated_cols = corr_matrix_no_diag.columns[in_range_mask.any()].tolist()

        if len(correlated_cols) > 1:
            filtered_corr = corr_matrix.loc[correlated_cols, correlated_cols]
            fig_corr = px.imshow(
                filtered_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_corr, width="stretch", theme="streamlit")
        else:
            st.info("No numeric columns have correlation within the selected range.")
    else:
        st.caption("Not enough numeric columns for correlation matrix.")

    st.divider()

    # 2. Missingness Map
    st.markdown("### Missingness Pattern Map")
    st.markdown("Visualizes where missing values occur across the rows of the dataset.")

    if df.size > 0:
        null_mask = df.isnull().astype(int)

        if null_mask.sum().sum() > 0:
            vis_df = null_mask
            if len(vis_df) > 1000:
                st.caption("Showing a representative sample of 1,000 rows for rendering performance.")
                vis_df = vis_df.sample(1000, random_state=42).sort_index()

            fig_null = px.imshow(
                vis_df,
                aspect="auto",
                color_continuous_scale=[[0, "#2c3e50"], [0.5, "#2c3e50"], [0.5, "#e74c3c"], [1, "#e74c3c"]],
                labels=dict(x="Columns", y="Row Index", color="Status")
            )
            fig_null.update_layout(
                coloraxis_colorbar=dict(
                    title="Status",
                    tickvals=[0.25, 0.75],
                    ticktext=["Present", "Missing"]
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis_title="Row Index"
            )
            st.plotly_chart(fig_null, width="stretch", theme="streamlit")
        else:
            st.success("🎉 No missing values found in the dataset!")
    else:
        st.caption("Dataset is empty.")

    st.divider()

    # 3. Outliers Grid
    st.markdown("### Global Outlier Distribution")
    st.markdown("Compares distributions of all numeric features on a single box plot visualization to highlight outliers.")

    if len(numeric_df.columns) > 0:
        st.markdown("*Note: Features are standardized to Z-scores (mean=0, std=1) to allow direct visual comparison across different scales.*")
        z_scored_df = pd.DataFrame()
        for col in numeric_df.columns:
            col_std = numeric_df[col].std()
            if col_std > 0:
                z_scored_df[col] = (numeric_df[col] - numeric_df[col].mean()) / col_std
            else:
                z_scored_df[col] = 0.0

        melted_z = z_scored_df.melt(var_name="Feature", value_name="Standardized Value")

        fig_outliers = px.box(
            melted_z,
            x="Standardized Value",
            y="Feature",
            color="Feature",
            height=max(200, 50 * len(numeric_df.columns)),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_outliers.update_layout(
            hovermode="closest",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Standardized Value (Z-Score)"
        )
        st.plotly_chart(fig_outliers, width="stretch", theme="streamlit")
    else:
        st.caption("No numeric features to display outliers.")

with tab3:
    if st.session_state.proposals:
        with st.expander(f"Recommended Rules ({len(st.session_state.proposals)})", expanded=False):
            if st.button("Accept All Recommendations", key="accept_all_proposals", width="stretch"):
                for p in st.session_state.proposals:
                    st.session_state.scanned_columns.add(p['column'])
                    if 'action' in p['rule_data']:
                        add_step(p['rule_data'])
                    else:
                        rule = p['rule_data'].copy()
                        rule.update({'enabled': True, 'color': f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                        st.session_state.rules.append(rule)
                st.session_state.proposals = []
                st.toast("All recommendations accepted")
                st.rerun()

            p_cols = st.columns(2)
            for p_idx, p in enumerate(st.session_state.proposals):
                with p_cols[p_idx % 2]:
                    st.markdown(f'<div class="proposal-box"><strong>{p["type"]} on {p["column"]}</strong><br/><small>{p["reason"]}</small></div>', unsafe_allow_html=True)
                    acc, dis = st.columns(2)
                    if acc.button("Accept", key=f"p_acc_{p_idx}", width="stretch"):
                        st.session_state.scanned_columns.add(p['column'])
                        if 'action' in p['rule_data']:
                            add_step(p['rule_data'])
                        else:
                            rule = p['rule_data'].copy()
                            rule.update({'enabled': True, 'color': f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                            st.session_state.rules.insert(0, rule)
                        st.session_state.proposals.pop(p_idx)
                        st.rerun()
                    if dis.button("Dismiss", key=f"p_dis_{p_idx}", width="stretch"):
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
                st.session_state.rules.insert(0, {"type": "Informational", "desc": note, "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                st.rerun()
        elif rtype == "Custom Expression":
            with st.form(key="custom_expr_form", clear_on_submit=True):
                q_str = st.text_input("Pandas Query String", placeholder="Age > 30 & Sex == 'male'", key="custom_query_input")
                submit_btn = st.form_submit_button("Add Rule")

            if submit_btn and q_str:
                try:
                    test_result = df.query(q_str)

                    if len(df) > 0 and len(test_result) == 0:
                        st.error("⚠️ This query returned no matches on the dataset. Please check for typos or type mismatches (e.g., comparing a number to a string). Rule not added.")
                    else:
                        st.session_state.rules.insert(0, {"type": "Custom Expression", "query": q_str, "desc": f"Matches: {q_str}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                        st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    # Categorize common errors for user-friendliness
                    if "invalid syntax" in err_msg.lower():
                        friendly_err = "Syntax Error: The expression has invalid characters or structure (e.g., using '>>' instead of '>')."
                    elif "is not defined" in err_msg.lower() or "not found" in err_msg.lower():
                        friendly_err = "Column Error: One of the columns mentioned in your query doesn't exist in the dataset."
                    elif "cannot compare" in err_msg.lower() or "not supported between instances" in err_msg.lower() or "typeerror" in err_msg.lower():
                        friendly_err = "Type Error: You are trying to compare incompatible types (e.g., a number with a string)."
                    else:
                        friendly_err = f"Pandas Error: {err_msg}"
                    st.error(friendly_err)
        elif rtype == "Relational Check":
            tcol, op = st.selectbox("Feature A", all_cols, key="rel_feature_a"), st.selectbox("Operator", [">", "<", "==", "!=", ">=", "<="], key="rel_op")
            target_type = st.radio("Compare with", ["Another Feature", "Constant Value"], horizontal=True, key="rel_target_type_radio")
            if target_type == "Another Feature":
                col_b = st.selectbox("Feature B", all_cols, key="rel_feature_b")
                if st.button("Add Rule", key="btn_add_rel_feat"):
                    st.session_state.rules.insert(0, {"type": "Relational Check", "col_a": tcol, "op": op, "col_b": col_b, "target_type": "Feature", "desc": f"{tcol} {op} {col_b}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                    st.rerun()
            else:
                val = st.text_input("Constant Value", key="rel_val_input")
                if st.button("Add Rule", key="btn_add_rel_val"):
                    try: final_val = float(val)
                    except: final_val = val
                    st.session_state.rules.insert(0, {"type": "Relational Check", "col_a": tcol, "op": op, "value": final_val, "target_type": "Value", "desc": f"{tcol} {op} {val}", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                    st.rerun()
        else:
            tcol = st.selectbox("Target Column", all_cols, key="rule_target_col")
            if rtype == "Range Check":
                if pd.api.types.is_numeric_dtype(df[tcol]):
                    num_col1, num_col2 = st.columns(2)
                    v_min, v_max = num_col1.number_input("Min", value=float(df[tcol].min()), key="range_min_input"), num_col2.number_input("Max", value=float(df[tcol].max()), key="range_max_input")
                    if st.button("Add Rule", key="btn_add_range"):
                        st.session_state.rules.insert(0, {"type": "Range Check", "col": tcol, "min": v_min, "max": v_max, "desc": f"{tcol} in [{v_min}, {v_max}]", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                        st.rerun()
                else:
                    st.warning(f"Range Checks are only applicable to numeric columns. '{tcol}' is {df[tcol].dtype}.")
            elif rtype == "Null Check":
                if st.button("Add Rule", key="btn_add_null"):
                    st.session_state.rules.insert(0, {"type": "Null Check", "col": tcol, "desc": f"{tcol} is NOT NULL", "enabled": True, "color": f"hsla({get_safe_hue(len(st.session_state.rules))}, 70%, 50%, 0.4)"})
                    st.rerun()
    with r2:
        rh1, rh2 = st.columns([2, 1], vertical_alignment="bottom")
        rh1.subheader("Active Rules")
        if st.session_state.rules and rh2.button("Clear All", width="stretch", key="clear_all_rules_btn"):
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

                    status_color, resolved = (rule['color'] if rule['enabled'] else "rgba(100,100,100,0.2)"), rule.get('resolved', False)
                    desc_style = "text-decoration: line-through; opacity: 0.5;" if not rule['enabled'] else ""
                    error_html = f"<br/><span style='color: #e74c3c; font-size: 0.85em; font-weight: bold;'>⚠️ Error: {rule.get('error')}</span>" if 'error' in rule else ""

                    # For Informational rules, don't show violation count
                    v_text = f"Violations: {v_count}" if rule['type'] != "Informational" else "Type: Info"
                    st.markdown(f'<div class="violation-card"><div style="border-left: 8px solid {status_color}; padding-left: 15px;"><strong style="{desc_style}">{rule["type"]}</strong><br/><code style="color: #4F8BF9; {desc_style}">{rule["desc"]}</code><br/><span style="font-size: 0.85em; opacity: 0.7;">{v_text}</span>{f"<br/><span style=\"color: #2ecc71; font-size: 0.85em; font-weight: bold;\">Status: Resolved</span>" if resolved else ""}{error_html}</div></div>', unsafe_allow_html=True)

                    if v_count > 0 and not resolved and rule['type'] != "Informational":
                        if rule['type'] == "Null Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Resolution", ["Select resolution method...", "Drop Rows", "Fill with Mean", "Fill with Median", "KNN Imputer", "Iterative Imputer"], key=f"res_{idx}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_res_{idx}", width="stretch"):
                                if res == "Drop Rows": add_step({"action": "drop_nulls", "column": rule['col']})
                                elif "Imputer" in res: add_step({"action": "fill_null", "column": rule['col'], "value": res.split()[0].lower()})
                                else: add_step({"action": "fill_null", "column": rule['col'], "value": res.split()[-1].lower()})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        elif rule['type'] == "Range Check":
                            res_cols = st.columns([3, 1])
                            res = res_cols[0].selectbox("Res", ["Select resolution method...", "Drop Rows", "Cap at Bounds", "Log Transform"], key=f"range_res_{idx}", label_visibility="collapsed")
                            if res != "Select resolution method..." and res_cols[1].button("Apply", key=f"btn_range_res_{idx}", width="stretch"):
                                if res == "Drop Rows": add_step({"action": "drop_violated", "rule": rule})
                                elif res == "Log Transform": add_step({"action": "log_transform", "column": rule['col']})
                                else: add_step({"action": "cap_range", "column": rule['col'], "min": rule['min'], "max": rule['max']})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()
                        else:
                            if st.button("Drop Violated Rows", key=f"gen_res_{idx}", width="stretch"):
                                add_step({"action": "drop_violated", "rule": rule})
                                st.session_state.rules[idx]['resolved'] = True
                                st.rerun()

                    btn_c1, btn_c2 = st.columns(2)
                    if btn_c1.button("Ignore" if rule['enabled'] else "Enable", key=f"tg_{idx}", width="stretch"):
                        st.session_state.rules[idx]['enabled'] = not rule['enabled']
                        st.rerun()
                    if btn_c2.button("Remove", key=f"del_{idx}", width="stretch"):
                        st.session_state.rules.pop(idx)
                        st.rerun()

with tab4:
    st.subheader("Transformations")
    t_type = st.selectbox("Type", ["Find and Replace", "Normalize Text", "Cast Data Type", "Drop Column", "Strip Whitespace", "Rename Column", "Reorder Columns"], key="trans_type_select")
    if t_type == "Find and Replace":
        c1, c2, c3 = st.columns(3)
        sf, sr, target = c1.text_input("Find", key="find_input"), c2.text_input("Replace", key="replace_input"), c3.selectbox("Columns", ["All"] + all_cols, key="replace_target_col")
        use_regex = st.toggle("Use Regular Expressions", key="replace_use_regex")
        if st.button("Add Step", key="btn_fr"):
            add_step({"action": "replace", "column": target, "find": sf, "replace": sr, "regex": use_regex})
            st.rerun()
    elif t_type == "Normalize Text":
        c1, c2 = st.columns(2)
        target, method = c1.selectbox("Columns", ["All"] + all_cols, key="norm_target_col"), c2.selectbox("Method", ["lowercase", "uppercase", "titlecase", "remove_punctuation", "fuzzy_dedupe"], key="norm_method_select")
        if st.button("Add Step", key="btn_norm"):
            add_step({"action": "normalize_text", "column": target, "value": method})
            st.rerun()
    elif t_type == "Cast Data Type":
        c1, c2 = st.columns(2)
        target, dtype_t = c1.selectbox("Column", all_cols, key="cast_target_col"), c2.selectbox("Cast To", ["string", "float64", "int64", "datetime64[ns]"], key="cast_dtype_select")
        if st.button("Add Step", key="btn_cast"):
            add_step({"action": "cast_type", "column": target, "dtype": dtype_t})
            st.rerun()
    elif t_type == "Drop Column":
        target = st.selectbox("Target Column", all_cols, key="drop_target_col")

        # Collision Detection
        dependent_rules = []
        for r in st.session_state.rules:
            if r.get('col') == target or r.get('col_a') == target or r.get('col_b') == target:
                dependent_rules.append(r['desc'])
            elif r.get('type') == "Custom Expression" and target in r.get('query', ''):
                dependent_rules.append(r['desc'])

        if dependent_rules:
            st.warning(f"⚠️ Column '{target}' is used in the following rules: {', '.join(dependent_rules)}. Dropping it may break these rules.")

        if st.button("Add Step", key="btn_drop"):
            add_step({"action": "drop_column", "column": target})
            st.rerun()
    elif t_type == "Strip Whitespace":
        target = st.selectbox("Columns", ["All"] + all_cols, key="strip_target_col")
        if st.button("Add Step", key="btn_strip"):
            add_step({"action": "strip_whitespace", "column": target})
            st.rerun()
    elif t_type == "Rename Column":
        target = st.selectbox("Target Column", all_cols, key="rename_target_col")
        new_name = st.text_input("New Column Name", key="rename_new_name_input")
        if st.button("Add Step", key="btn_rename"):
            if not new_name.strip():
                st.error("Column name cannot be empty")
            elif new_name in all_cols:
                st.error(f"A column named '{new_name}' already exists.")
            else:
                add_step({"action": "rename_column", "column": target, "value": new_name})
                # Sync validation rules
                for rule in st.session_state.rules:
                    if rule.get('col') == target:
                        rule['col'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                    if rule.get('col_a') == target:
                        rule['col_a'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                    if rule.get('col_b') == target:
                        rule['col_b'] = new_name
                        rule['desc'] = rule['desc'].replace(target, new_name)
                # Sync active features in diagnostics tab
                if target in st.session_state.active_features:
                    idx = st.session_state.active_features.index(target)
                    st.session_state.active_features[idx] = new_name
                st.rerun()
    elif t_type == "Reorder Columns":
        if 'temp_col_order' not in st.session_state or set(st.session_state.temp_col_order) != set(all_cols):
            st.session_state.temp_col_order = list(all_cols)

        temp_cols = st.session_state.temp_col_order

        st.markdown("### Drag-and-Drop to Reorder Columns")
        st.info("Grab any column name card and drag to rearrange the column order. Click \"Apply Column Order\" once done.")

        # Display mini toast notification if reorder was applied successfully
        if st.session_state.get('show_reorder_success'):
            st.toast("Column order applied successfully!", icon="✅")
            st.session_state.show_reorder_success = False

        # Custom styles matching Lumi's global dark/gray card color themes and typography
        sortable_style = """
        .sortable-component {
            background-color: transparent !important;
            font-family: 'JetBrains Mono', monospace !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
            padding: 10px 0 !important;
        }
        .sortable-item {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            color: inherit !important;
            background-color: rgba(128, 128, 128, 0.05) !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 6px !important;
            padding: 6px 12px !important;
            margin: 4px !important;
            cursor: grab !important;
            transition: all 0.2s ease-in-out !important;
            display: inline-block !important;
            box-shadow: none !important;
        }
        .sortable-item:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(79, 139, 249, 0.4) !important;
        }
        .sortable-item:active {
            cursor: grabbing !important;
            background-color: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(79, 139, 249, 0.6) !important;
        }
        """

        # Render sortable items list using streamlit-sortables in horizontal mode
        sorted_cols = sort_items(temp_cols, direction="horizontal", custom_style=sortable_style, key="col_reorder_widget")

        if sorted_cols != temp_cols:
            st.session_state.temp_col_order = sorted_cols
            st.rerun()

        btn_placeholder = st.empty()
        if btn_placeholder.button("Apply Column Order", key="btn_apply_reorder"):
            loading_html = """
            <div style="display: inline-flex; align-items: center; gap: 12px; height: 38px; margin-bottom: 1rem;">
                <button disabled style="
                    border-radius: 6px;
                    border: 1px solid rgba(128, 128, 128, 0.2);
                    background-color: transparent;
                    color: inherit;
                    opacity: 0.4;
                    padding: 0px 16px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 14px;
                    height: 38px;
                    cursor: not-allowed;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    box-sizing: border-box;
                ">Apply Column Order</button>
                <div class="spinner-circle"></div>
            </div>
            """
            btn_placeholder.markdown(loading_html, unsafe_allow_html=True)
            import time
            time.sleep(0.8)  # perceived loading delay to show spinner
            add_step({"action": "reorder_columns", "value": list(st.session_state.temp_col_order)})
            st.session_state.show_reorder_success = True
            st.rerun()

with tab5:
    st.subheader("Data Lineage")
    if not st.session_state.cleaning_recipe:
        st.caption("No transformations applied yet")
    else:
        # Use cached intermediate states for high performance
        for i, (desc, health, rows, _) in enumerate(st.session_state.intermediate_states):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f'<div class="recipe-step"><strong>{i + 1}. {desc}</strong> | Health: {health}% | Rows: {rows:,}</div>', unsafe_allow_html=True)

            # Original data cannot be removed
            if i > 0:
                if c2.button("Remove", key=f"rm_step_{i}", width="stretch"):
                    st.session_state.cleaning_recipe.pop(i-1)
                    # When a step is removed, we truncate the intermediate states and rebuild the cache
                    remaining_recipe = st.session_state.cleaning_recipe[i-1:]
                    st.session_state.cleaning_recipe = st.session_state.cleaning_recipe[:i-1]
                    st.session_state.intermediate_states = st.session_state.intermediate_states[:i]
                    for r_step in remaining_recipe:
                        add_step(r_step)
                    st.rerun()

with tab6:
    v = st.radio("Mode", ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"], horizontal=True, key="p_mode")
    if v == "Raw Data (Before)": st.dataframe(st.session_state.raw_data, width="stretch")
    elif v == "Cleaned Data (After)": st.dataframe(df, width="stretch")
    else:
        code_output = generate_pipeline_code(st.session_state.cleaning_recipe)
        st.code(code_output, language="python")
        st.download_button("Download clean_data.py", code_output, "clean_data.py", "text/x-python", width="stretch", key="download_pipeline_btn")

st.divider()
st.subheader("Violation Browser")

# Optimization: Instead of styling the entire dataframe (which is slow),
# filter to only the rows that violate at least one rule.
active_rules_for_heatmap = [r for r in st.session_state.rules if r.get('enabled', True) and r.get('type') != "Informational"]

if not active_rules_for_heatmap:
    st.info("No active rules to check for violations.")
else:
    # Build a combined mask for all violations
    combined_mask = pd.Series(False, index=df.index)
    for rule in active_rules_for_heatmap:
        try:
            combined_mask |= evaluate_rule(df, rule)
        except Exception:
            continue

    violation_df = df[combined_mask]

    if violation_df.empty:
        st.success("🎉 No violations found in the current dataset!")
    else:
        st.warning(f"Found {len(violation_df):,} rows with violations. Showing top 100.")
        # Apply styling only to the small subset of rows for high performance
        heatmap_sdf, _ = get_heatmap_styles(violation_df, active_rules_for_heatmap)
        st.dataframe(violation_df.head(100).style.apply(lambda _: heatmap_sdf.head(100), axis=None), width="stretch")
