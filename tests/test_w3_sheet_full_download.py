"""Q13: the full-file download reads the same XLSX sheet the sample was loaded from."""
import io

import pandas as pd
import streamlit as st

from lumi_state import session_io


class _Buf(io.BytesIO):
    name = "book.xlsx"


def _two_sheet_bytes():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"b": [7, 8, 9]}).to_excel(w, sheet_name="Second", index=False)
    return buf.getvalue()


def test_full_download_uses_sheet_recorded_with_upload():
    up = _Buf(_two_sheet_bytes())
    up.lumi_sheet = "Second"
    st.session_state.pop("xlsx_sheet", None)  # e.g. after a resume: no landing-page choice in state
    st.session_state.full_source = session_io._capture_full_source(up)
    df = session_io.load_full_data()
    assert list(df.columns) == ["b"] and len(df) == 3


def test_stale_session_choice_does_not_change_full_download():
    up = _Buf(_two_sheet_bytes())  # no sheet chosen for this upload -> first sheet
    st.session_state["xlsx_sheet"] = "Second"  # left over from another workbook
    st.session_state.full_source = session_io._capture_full_source(up)
    df = session_io.load_full_data()
    assert list(df.columns) == ["a"]
