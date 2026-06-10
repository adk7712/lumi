import streamlit as st
import pandas as pd
from engine import generate_pipeline_code

def render_pipeline_preview_tab(df):
    v = st.radio("Mode", ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"], horizontal=True, key="p_mode")
    if v == "Raw Data (Before)":
        st.dataframe(st.session_state.raw_data, width="stretch")
    elif v == "Cleaned Data (After)":
        st.dataframe(df, width="stretch")

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
        st.code(code_output, language="python")
        st.download_button("Download clean_data.py", code_output, "clean_data.py", "text/x-python", width="stretch", key="download_pipeline_btn")
