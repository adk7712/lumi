import pandas as pd
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
    <div style="border-left: 3px solid rgba(28, 131, 225, 0.5); padding: 2px 8px; margin-bottom: 10px;">
        <div style="font-size: 1.25rem; line-height: 2; margin-bottom: 3px; font-weight: 400;">
            {label}
        </div>
        <div style="font-size: 1.0rem; font-weight: 500; line-height: 1; opacity: 0.75;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

def inject_custom_css(st_object):
    """Injects the global CSS for the Lumi workspace."""
    # Read the CSS from the external file using absolute path relative to this script
    css_path = Path(__file__).parent / "style.css"

    # Fallback to the current working directory if file not found
    if not css_path.exists():
        css_path = Path("style.css")

    try:
        with open(css_path, "r") as f:
            css_content = f.read()
        st_object.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st_object.error(f"CSS file not found at {css_path}. Please ensure style.css is in the app directory.")

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