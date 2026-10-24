import streamlit as st
import pandas as pd
import io
import json
import hashlib
import logging
from engine import generate_pipeline_code, generate_notebook_code, apply_recipe

logger = logging.getLogger("lumi.pipeline")

EXCEL_MAX_ROWS = 1_048_576  # hard worksheet limit in .xlsx
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _recipe_signature(recipe) -> str:
    """Stable hash of the recipe metadata; changes whenever a step is added/removed/edited."""
    return hashlib.sha1(json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _build_exports(df, include_xlsx=True):
    """Returns {"csv": bytes, "xlsx": bytes|None, "xlsx_note": str|None, "rows": int}."""
    out = {"csv": df.to_csv(index=False).encode("utf-8"), "xlsx": None, "xlsx_note": None, "rows": len(df)}
    if not include_xlsx:
        return out
    if len(df) + 1 > EXCEL_MAX_ROWS:
        out["xlsx_note"] = (f"Excel is limited to {EXCEL_MAX_ROWS:,} rows per sheet; "
                            f"this result has {len(df):,}. Use the CSV download.")
        return out
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        out["xlsx"] = buf.getvalue()
    except Exception as e:
        out["xlsx_note"] = f"Error generating Excel file: {e}"
    return out


def _track(fmt):
    def _cb():
        from ui_utils import queue_event
        queue_event("file_downloaded", {"format": fmt})
    return _cb


def _show_buttons(exports, csv_name, xlsx_name, key_prefix):
    d1, d2 = st.columns(2)
    d1.download_button("Download Cleaned CSV (.csv)", exports["csv"], csv_name, "text/csv",
                       use_container_width=True, key=f"{key_prefix}_csv", on_click=_track("csv"))
    if exports.get("xlsx") is not None:
        d2.download_button("Download Cleaned Excel (.xlsx)", exports["xlsx"], xlsx_name, _XLSX_MIME,
                           use_container_width=True, key=f"{key_prefix}_excel", on_click=_track("xlsx"))
    elif exports.get("xlsx_note"):
        d2.error(exports["xlsx_note"]) if exports["xlsx_note"].startswith("Error") else d2.info(exports["xlsx_note"])


def _active_sample_info():
    info = st.session_state.get("sample_info")
    raw = st.session_state.get("raw_data")
    if info and raw is not None and info.get("shown") == len(raw):
        return info
    return None


def _render_downloads(df):
    sig = _recipe_signature(st.session_state.cleaning_recipe)
    key = (st.session_state.get("last_file_hash"), sig)
    info = _active_sample_info()

    # In-app (sample or whole-file) result, cached on the recipe signature.
    cached = st.session_state.get("sample_export")
    if not cached or cached["key"] != key:
        cached = {"key": key, **_build_exports(df)}
        st.session_state.sample_export = cached
    if info:
        st.caption(f"These downloads contain the {len(df):,}-row working sample only.")
    _show_buttons(cached, "cleaned_data.csv", "cleaned_data.xlsx", "download_clean")

    if not info:
        return

    st.markdown("**Full file**")
    total = info.get("total")
    xlsx_possible = not total or total <= EXCEL_MAX_ROWS
    want_xlsx = st.checkbox("Also build Excel (.xlsx)", value=xlsx_possible and (not total or total <= 200_000),
                            disabled=not xlsx_possible, key="full_include_xlsx",
                            help=None if xlsx_possible else f"Excel is limited to {EXCEL_MAX_ROWS:,} rows.")
    want_xlsx = bool(want_xlsx) and xlsx_possible
    full = st.session_state.get("full_export")
    if full and full["key"] != key:
        full = None
        st.session_state.full_export = None
    needs_prepare = full is None or (want_xlsx and full["xlsx"] is None and not full.get("xlsx_note"))
    if needs_prepare:
        if st.button("Prepare full-file download", key="prepare_full_download", use_container_width=True,
                     help="Re-reads the original file and applies the current cleaning steps to every row."):
            if _prepare_full(key, want_xlsx):
                st.session_state._needs_rerun = True
        return
    for m in full.get("messages", []):
        st.warning(m)
    st.success(f"Full result ready: {full['rows']:,} rows.")
    _show_buttons(full, "cleaned_data_full.csv", "cleaned_data_full.xlsx", "download_full")


def _prepare_full(key, include_xlsx):
    from state_manager import load_full_data
    try:
        with st.spinner("Applying the cleaning steps to the full file..."):
            full_df = load_full_data()
            result, messages = apply_recipe(full_df, st.session_state.cleaning_recipe)
            del full_df
            exports = _build_exports(result, include_xlsx)
            del result
        st.session_state.full_export = {"key": key, "messages": list(messages), **exports}
        return True
    except MemoryError:
        st.session_state.full_export = None
        st.error("The full file is too large to process in memory here. Export the Python script "
                 "(Python Code mode) and run it locally on the full file instead.")
    except Exception as e:
        logger.warning("Full-file export failed", exc_info=True)
        st.session_state.full_export = None
        st.error(f"Could not prepare the full-file download: {e}")
        return False

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
            _render_downloads(df)
    else:
        code_output = generate_pipeline_code(st.session_state.cleaning_recipe, st.session_state.rules)
        notebook_output = generate_notebook_code(st.session_state.cleaning_recipe, st.session_state.rules)
        st.code(code_output, language="python")
        
        c1, c2 = st.columns(2)
        
        def _track_py():
            from ui_utils import queue_event
            queue_event("script_exported", {"format": "python"})
            
        def _track_nb():
            from ui_utils import queue_event
            queue_event("script_exported", {"format": "jupyter"})
            
        c1.download_button("Download clean_data.py", code_output, "clean_data.py", "text/x-python", use_container_width=True, key="download_pipeline_btn", on_click=_track_py)
        c2.download_button("Download clean_data.ipynb", notebook_output, "clean_data.ipynb", "application/x-ipynb+json", use_container_width=True, key="download_notebook_btn", on_click=_track_nb)
