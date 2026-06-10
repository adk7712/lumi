import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from rule_utils import evaluate_rule

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
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            return fig
    return None

def plot_missingness_map(df: pd.DataFrame) -> tuple:
    """Generates a binary missingness pattern map representation."""
    if df.size > 0:
        null_mask = df.isnull().astype(int)
        if null_mask.sum().sum() > 0:
            vis_df = null_mask
            is_sampled = len(vis_df) > 1000
            if is_sampled:
                vis_df = vis_df.sample(1000, random_state=42).sort_index()

            fig = px.imshow(
                vis_df,
                aspect="auto",
                color_continuous_scale=[[0, "#2c3e50"], [0.5, "#2c3e50"], [0.5, "#e74c3c"], [1, "#e74c3c"]],
                labels=dict(x="Columns", y="Row Index", color="Status")
            )
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Status",
                    tickvals=[0.25, 0.75],
                    ticktext=["Present", "Missing"]
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis_title="Row Index"
            )
            return fig, is_sampled
    return None, False

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

        is_sampled = len(z_scored_df) > 1000
        plot_z_df = z_scored_df
        if is_sampled:
            plot_z_df = plot_z_df.sample(1000, random_state=42)

        melted_z = plot_z_df.melt(var_name="Feature", value_name="Standardized Value")

        fig = px.box(
            melted_z,
            x="Standardized Value",
            y="Feature",
            color="Feature",
            height=max(200, 50 * len(numeric_df.columns)),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Standardized Value (Z-Score)"
        )
        return fig, is_sampled
    return None, False