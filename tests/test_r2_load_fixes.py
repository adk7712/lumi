"""R2 load fixes: leading-zero text columns, trailing delimiters, float64 kept, failed uploads."""
import io
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import numpy as np
import pandas as pd
import pytest
import streamlit as st

import state_manager as sm
from codegen import generate_pipeline_code
from lumi_state import session_io
from lumi_state.core import CachedFileWrapper, downcast_dtypes, load_data

ZIP_CSV = b"id,zip,price\n1,01234,214.34\n2,00501,n/a\n3,90210,7.5\n4,n/a,1.25\n"


def _ld(raw: bytes, name="t.csv", nrows=None):
    return load_data(CachedFileWrapper(raw, name), nrows=nrows)


def _xlsx(rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    b = io.BytesIO()
    wb.save(b)
    return b.getvalue()


def _full(raw, name, monkeypatch):
    monkeypatch.setattr(session_io.st, "session_state", {"full_source": {"name": name, "bytes": raw}})
    return session_io.load_full_data()


# --- 1. leading zeros -------------------------------------------------------------------

def test_csv_leading_zero_column_stays_text(monkeypatch):
    df = _ld(ZIP_CSV)
    assert df["zip"].tolist()[:3] == ["01234", "00501", "90210"]
    assert pd.isna(df["zip"].iloc[3])  # default NA handling still applies
    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_float_dtype(df["price"])
    full = _full(ZIP_CSV, "t.csv", monkeypatch)
    pd.testing.assert_frame_equal(full, df)
    assert "01234" in full.to_csv(index=False)


def test_csv_plain_numbers_stay_numeric():
    df = _ld(b"a,b\n0,0.5\n10,-0.25\n0,n/a\n")
    assert pd.api.types.is_integer_dtype(df["a"])
    assert pd.api.types.is_float_dtype(df["b"])


def test_csv_long_integer_ids_with_missing_stay_text():
    df = _ld(b"card,x\n1234567890123456789,1\nn/a,2\n")
    assert df["card"].iloc[0] == "1234567890123456789"


def test_xlsx_text_cells_with_leading_zeros_stay_text(monkeypatch):
    raw = _xlsx([["zip", "n"], ["01234", 1], ["n/a", 2.5], ["90210", 3]])
    df = _ld(raw, "z.xlsx")
    assert df["zip"].tolist()[0] == "01234" and df["zip"].tolist()[2] == "90210"
    assert pd.isna(df["zip"].iloc[1])
    assert pd.api.types.is_float_dtype(df["n"])
    pd.testing.assert_frame_equal(_full(raw, "z.xlsx", monkeypatch), df)


# --- 2. trailing delimiter -------------------------------------------------------------

def test_trailing_delimiter_does_not_shift_columns(monkeypatch):
    raw = b"id,age,name\n1,,a,\n2,30,b,\n"
    df = _ld(raw)
    assert list(df.columns) == ["id", "age", "name"]
    assert df["id"].tolist() == [1, 2]
    assert df["name"].tolist() == ["a", "b"]
    assert isinstance(df.index, pd.RangeIndex)
    pd.testing.assert_frame_equal(_full(raw, "t.csv", monkeypatch), df)


# --- 6. floats are not downcast --------------------------------------------------------

def test_floats_stay_float64_on_load_and_downcast():
    df = _ld(b"v\n214.34\n1.1\n")
    assert df["v"].dtype == np.float64
    assert df["v"].iloc[0] == 214.34
    out = downcast_dtypes(pd.DataFrame({"f": [214.34, 1.5], "i": [1, 2]}))
    assert out["f"].dtype == np.float64
    assert out["i"].dtype == np.int8  # integer downcasting is kept


def test_exported_helper_keeps_float64():
    ns = {}
    code = generate_pipeline_code([])
    exec(code, ns)  # noqa: S102 - our own generated script
    out = ns["_downcast_dtypes"](pd.DataFrame({"f": [214.34, 1.5], "i": [1, 2]}))
    assert out["f"].dtype == np.float64 and out["i"].dtype == np.int8


# --- 3. failed uploads -----------------------------------------------------------------

def _Upload(content, name="data.csv"):
    return CachedFileWrapper(content, name)  # same hashable buffer type the app uses for uploads


@pytest.fixture
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path / "cache")
    saves = []
    monkeypatch.setattr(session_io, "save_db_session", lambda: saves.append(1))
    monkeypatch.setattr(session_io, "cache_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(session_io, "_set_session_cookie", lambda *a, **k: None)
    sm.initialize_state(from_reset=True)
    st.session_state.pop("load_error", None)
    yield saves
    st.session_state.pop("load_error", None)


@pytest.mark.parametrize("content", [
    b"",                                   # empty file
    b"a,b\n1,2\n3,4,5,6\n",                # ragged CSV
    b"a,b,c\n",                            # headers but no rows
])
def test_failed_upload_keeps_landing_and_stores_error(fresh_state, content):
    with mock.patch("streamlit.error"):
        ok = sm.process_uploaded_file(_Upload(content), "h-bad")
    assert ok is False
    assert st.session_state.raw_data is None
    assert st.session_state.get("load_error")
    assert fresh_state == []  # nothing saved for a failed load


def test_zip_bomb_xlsx_reported(fresh_state, monkeypatch):
    from lumi_state import core
    monkeypatch.setattr(core, "MAX_XLSX_UNCOMPRESSED_BYTES", 10)
    raw = _xlsx([["a"], [1]])
    with mock.patch("streamlit.error"):
        ok = sm.process_uploaded_file(_Upload(raw, "b.xlsx"), "h-bomb")
    assert ok is False and st.session_state.raw_data is None
    assert "limit" in st.session_state.load_error


def test_successful_upload_clears_previous_error(fresh_state):
    st.session_state.load_error = "old"
    assert sm.process_uploaded_file(_Upload(b"a,b\n1,2\n"), "h-good") is not False
    assert st.session_state.raw_data is not None
    assert not st.session_state.get("load_error")


def test_landing_shows_persistent_load_error():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30)
    at.session_state["_is_testing"] = True
    at.session_state["load_error"] = "Could not read this file: it is empty."
    at.run()
    assert not at.exception
    assert any("it is empty" in e.value for e in at.error)
    at.run()  # survives a rerun
    assert any("it is empty" in e.value for e in at.error)


def test_landing_does_not_reprocess_the_failed_upload():
    from views.landing import _should_process_upload
    st.session_state.load_error = "bad"
    st.session_state._load_error_hash = "h1"
    try:
        assert _should_process_upload("h1") is False
        assert st.session_state.load_error == "bad"
        assert _should_process_upload("h2") is True  # a new file is a new attempt
        assert not st.session_state.get("load_error")
    finally:
        st.session_state.pop("load_error", None)
        st.session_state.pop("_load_error_hash", None)
