import streamlit as st
import pandas as pd
from rule_utils import evaluate_rule

def inject_custom_css():
    """Injects the global CSS for the Lumi workspace."""
    # Main block to inject custom CSS for consistent UI styling.
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

        /* --- Hidden Streamlit Elements --- */
        header,
        [data-testid="stHeaderActionElements"],
        .stMarkdown a,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploader"] small {
            display: none !important;
        }

        /* Push content up to the very top */
        .block-container {
            padding-top: 0rem;
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
            padding-right: 20px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.1);
            margin-bottom: 10px;
            position: relative;
        }

        /* Tab Label Spacing */
        .stTabs [data-baseweb="tab-list"] button {
            margin-right: 15px;
        }

        /* Bigger Tab Labels */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p,
        .stTabs button[data-baseweb="tab"] p,
        .stTabs button[role="tab"] p,
        .stTabs button[role="tab"] span,
        .stTabs button[data-testid="stBaseButton-tab"] p {
            font-size: 1.1rem;
            font-weight: 600;
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

        /* --- UPLOAD BUTTON FIX --- */
        [data-testid="stFileUploader"] {
            padding: 0 !important;
            width: 100% !important;
        }
        /* Strip all wrappers of their default dropzone styling */
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileDropzone"] {
            padding: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            background-color: transparent !important;
            border: none !important;
            display: flex !important;
            align-items: stretch !important;
            justify-content: stretch !important;
            width: 100% !important;
        }
        [data-testid="stFileUploader"] button {
            width: 100% !important;
            height: 38px !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 6px !important;
        }
        /* Hide ALL internal Streamlit elements to prevent layout interference */
        [data-testid="stFileUploader"] button > * {
            display: none !important;
        }
        [data-testid="stFileUploader"] button::before {
            content: "Upload Dataset" !important;
            visibility: visible !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            color: inherit !important;
        }

        /* --- Custom Styling for Expander --- */
        [data-testid="stExpander"] {
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
            padding: 0 !important;
        }
        [data-testid="stExpander"] details {
            border: none !important;
            padding: 0 !important;
        }
        [data-testid="stExpander"] summary {
            padding: 0 !important;
            background-color: transparent !important;
            list-style: none !important;
        }
        [data-testid="stExpander"] summary::-webkit-details-marker {
            display: none !important;
        }
        [data-testid="stExpander"] summary p {
            font-size: 1.25rem !important; /* Match st.subheader */
            font-weight: 600 !important;
            margin: 0 !important;
            color: inherit !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            padding-top: 1rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        /* Font Consistency */
        html, body, .stMarkdown, p, span, label, button, code, strong {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* --- Clean Buttons --- */
        .stButton > button, 
        .stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover, 
        .stDownloadButton > button:hover {
            border-color: #4F8BF9;
            color: #4F8BF9;
            background-color: rgba(79, 139, 249, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

def get_safe_hue(n):
    """Returns a high-contrast hue that avoids red and green."""
    # Select hues that avoid red/green to improve accessibility and visual distinction.
    safe_hues = [
        200, 240, 280, 310, 185, 220, 260, 300, 330,
        170, 210, 250, 290, 320, 195, 230, 270, 305, 340
    ]
    return safe_hues[n % len(safe_hues)]

def get_heatmap_styles(df_d, rules):
    """Generates the styling dataframe for the validation heatmap."""
    # Create a DataFrame to hold style information, highlighting cells with rule violations.
    sdf = pd.DataFrame('', index=df_d.index, columns=df_d.columns)
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
            st.toast(f"Heatmap Style Error ({r.get('desc', 'N/A')}): {str(e)}", icon="🚨")
            continue
    return sdf
