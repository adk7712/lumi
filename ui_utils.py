import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
from rule_utils import evaluate_rule

# Constants for UI plots and displays
PLOT_SAMPLE_SIZE = 1000
DIAGNOSTIC_CHART_HEIGHT = 220
MAX_CATEGORIES_DISPLAY = 10
VIOLATION_PREVIEW_LIMIT = 100

def downsample_for_plot(df_or_series, sample_size: int = PLOT_SAMPLE_SIZE):
    """Downsamples a DataFrame or Series if it exceeds sample_size, returning the sample and an is_sampled bool."""
    if len(df_or_series) > sample_size:
        return df_or_series.sample(sample_size, random_state=42).sort_index(), True
    return df_or_series, False

def render_diagnostic_metric(container, label: str, value: str):
    """
    Render a diagnostic metric with small label text using custom HTML.
    This bypasses Streamlit's metric styling limitations.

    Args:
        container: A streamlit column/container object
        label: The metric label (Type, Nulls, Unique, etc.)
        value: The metric value (int64, 0, 3, etc.)
    """
    container.markdown(f"""
    <div class="diagnostic-metric">
        <div class="diagnostic-metric-label">
            {label}
        </div>
        <div class="diagnostic-metric-value">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

def inject_custom_css(st_object):
    """Injects the global CSS for the Lumi workspace."""
    # Read the CSS from the external file using absolute path relative to this script
    css_path = Path(__file__).parent / "styles" / "global.css"

    # Fallback to the current working directory if file not found
    if not css_path.exists():
        css_path = Path("styles") / "global.css"

    try:
        with open(css_path, "r") as f:
            css_content = f.read()
        st_object.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st_object.error(f"CSS file not found at {css_path}. Please ensure styles/global.css is in the app directory.")

def load_style(name: str) -> str:
    """Reads a stylesheet from the styles directory and returns it as a string."""
    css_path = Path(__file__).parent / "styles" / name
    if not css_path.exists():
        css_path = Path("styles") / name
    try:
        with open(css_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_safe_hue(n: int) -> int:
    """Returns a high-contrast hue that avoids red and green."""
    # Select hues that avoid red/green to improve accessibility and visual distinction.
    safe_hues = [
        200, 240, 280, 310, 185, 220, 260, 300, 330,
        170, 210, 250, 290, 320, 195, 230, 270, 305, 340
    ]
    return safe_hues[n % len(safe_hues)]

def get_heatmap_styles(df_d: pd.DataFrame, rules: list) -> tuple[pd.DataFrame, list[str]]:
    """Generates the styling dataframe for the validation heatmap."""
    # Create a DataFrame to hold style information, highlighting cells with rule violations.
    sdf = pd.DataFrame('', index=df_d.index, columns=df_d.columns)
    messages = []
    for r in rules:
        if not r.get('enabled', True):
            continue
        try:
            violation_mask = evaluate_rule(df_d, r)
            style = f"background-color: {r['color']};"

            rule_type = r.get('type')
            if rule_type == "Custom Expression":
                sdf.loc[violation_mask, :] = style
            elif rule_type == "Relational Check":
                sdf.loc[violation_mask, r['col_a']] = style
                if r.get('target_type') == 'Feature':
                    sdf.loc[violation_mask, r['col_b']] = style
            else:
                if 'col' in r:
                    sdf.loc[violation_mask, r['col']] = style

        except Exception as e:
            messages.append(f"Heatmap Style Error ({r.get('desc', 'N/A')}): {str(e)}")
            continue
    return sdf, messages

def apply_lumi_layout(fig: go.Figure) -> go.Figure:
    """Applies a consistent theme and layout configuration to a Plotly figure."""
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        font_family="JetBrains Mono, Courier New, monospace",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

@st.cache_data
def plot_correlation_matrix(df: pd.DataFrame, corr_range: tuple) -> go.Figure:
    """Computes features' correlation matrix and plots a filtered heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        corr_matrix_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_matrix_no_diag.values, np.nan)
        in_range_mask = (corr_matrix_no_diag >= corr_range[0]) & (corr_matrix_no_diag <= corr_range[1])
        correlated_cols = corr_matrix_no_diag.columns[in_range_mask.any()].tolist()

        if len(correlated_cols) > 1:
            filtered_corr = corr_matrix.loc[correlated_cols, correlated_cols]
            fig = px.imshow(
                filtered_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            return apply_lumi_layout(fig)
    return None

@st.cache_data
def plot_missingness_map(df: pd.DataFrame) -> tuple:
    """Generates a binary missingness pattern map representation."""
    if df.size > 0:
        null_mask = df.isnull().astype(int)
        if null_mask.sum().sum() > 0:
            vis_df, is_sampled = downsample_for_plot(null_mask)

            fig = px.imshow(
                vis_df,
                aspect="auto",
                color_continuous_scale=[[0, "#2c3e50"], [0.5, "#2c3e50"], [0.5, "#e74c3c"], [1, "#e74c3c"]],
                labels=dict(x="Columns", y="Row Index", color="Status")
            )
            apply_lumi_layout(fig)
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Status",
                    tickvals=[0.25, 0.75],
                    ticktext=["Present", "Missing"]
                ),
                yaxis_title="Row Index"
            )
            return fig, is_sampled
    return None, False


def get_loading_spinner_html(text: str = "Apply Column Order") -> str:
    """Returns HTML for a disabled button accompanied by a CSS loading spinner."""
    return f"""
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
        ">{text}</button>
        <div class="spinner-circle"></div>
    </div>
    """

@st.cache_data
def plot_outlier_distribution(df: pd.DataFrame) -> tuple:
    """Computes column Z-scores and plots comparative outlier box plots."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        z_scored_df = pd.DataFrame()
        for col in numeric_df.columns:
            col_std = numeric_df[col].std()
            if col_std > 0:
                z_scored_df[col] = (numeric_df[col] - numeric_df[col].mean()) / col_std
            else:
                z_scored_df[col] = 0.0

        plot_z_df, is_sampled = downsample_for_plot(z_scored_df)

        melted_z = plot_z_df.melt(var_name="Feature", value_name="Standardized Value")

        fig = px.box(
            melted_z,
            x="Standardized Value",
            y="Feature",
            color="Feature",
            height=max(200, 50 * len(numeric_df.columns)),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        apply_lumi_layout(fig)
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            xaxis_title="Standardized Value (Z-Score)"
        )
        return fig, is_sampled
    return None, False