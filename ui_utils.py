import pandas as pd
from rule_utils import evaluate_rule
def inject_custom_css(st_object):
    """Injects the global CSS for the Lumi workspace."""
    # Read the CSS from the external file
    with open("style.css", "r") as f:
        css_content = f.read()
    st_object.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

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
