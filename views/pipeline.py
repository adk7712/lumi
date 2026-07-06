import streamlit as st
import pandas as pd
from engine import generate_pipeline_code, generate_notebook_code

def render_pipeline_preview_tab(df):
    v = st.radio("Mode", ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"], horizontal=True, key="p_mode")
    
    if v in ["Raw Data (Before)", "Cleaned Data (After)"]:
        # Toolbar controls
        c1, c2 = st.columns(2)
        with c1:
            preview_mode = st.selectbox(
                "Preview Mode", 
                ["First Rows", "Last Rows", "Random Sample", "All Rows"], 
                index=0, 
                key="preview_mode_select"
            )
        with c2:
            max_val = max(1, len(st.session_state.raw_data) if v == "Raw Data (Before)" else len(df))
            row_count = st.number_input(
                "Rows to Show", 
                min_value=1, 
                max_value=max_val, 
                value=min(100, max_val), 
                step=10, 
                disabled=(preview_mode == "All Rows"),
                key="preview_row_count"
            )
            
        target_df = st.session_state.raw_data if v == "Raw Data (Before)" else df
        
        # Apply slice
        if preview_mode == "First Rows":
            display_df = target_df.head(row_count)
        elif preview_mode == "Last Rows":
            display_df = target_df.tail(row_count)
        elif preview_mode == "Random Sample":
            display_df = target_df.sample(n=min(row_count, len(target_df)), random_state=42) if len(target_df) > 0 else target_df
        else:  # All Rows
            display_df = target_df
            
        st.dataframe(display_df, width="stretch")
        
        if v == "Cleaned Data (After)":
            st.divider()
            d_col1, d_col2 = st.columns(2)
            csv_data = df.to_csv(index=False).encode('utf-8')
            d_col1.download_button("Download Cleaned CSV (.csv)", csv_data, "cleaned_data.csv", "text/csv", use_container_width=True, key="download_clean_csv")

            try:
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                excel_data = excel_buffer.getvalue()
                d_col2.download_button("Download Cleaned Excel (.xlsx)", excel_data, "cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="download_clean_excel")
            except Exception as e:
                d_col2.error(f"Error generating Excel file: {str(e)}")
    else:
        code_output = generate_pipeline_code(st.session_state.cleaning_recipe, st.session_state.rules)
        notebook_output = generate_notebook_code(st.session_state.cleaning_recipe, st.session_state.rules)
        st.code(code_output, language="python")
        
        c1, c2 = st.columns(2)
        c1.download_button("Download clean_data.py", code_output, "clean_data.py", "text/x-python", use_container_width=True, key="download_pipeline_btn")
        c2.download_button("Download clean_data.ipynb", notebook_output, "clean_data.ipynb", "application/x-ipynb+json", use_container_width=True, key="download_notebook_btn")
