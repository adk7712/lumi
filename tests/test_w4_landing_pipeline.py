"""B4 (XLSX sheet kept with the cached upload), B1 (escaped names on landing), Q8 (full-download guard)."""
import io

import pandas as pd
import streamlit as st

from lumi_state import session_io
from lumi_state.core import read_tabular


class _Buf(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def _two_sheet_xlsx() -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"b": [7, 8, 9]}).to_excel(w, sheet_name="Second", index=False)
    return bio.getvalue()


def _cache_env(monkeypatch, tmp_path):
    import ui_utils
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ui_utils, "is_authenticated_user", lambda: True)
    st.session_state["_is_testing"] = False


def test_cached_upload_keeps_chosen_sheet(monkeypatch, tmp_path):
    _cache_env(monkeypatch, tmp_path)
    up = _Buf(_two_sheet_xlsx(), "book.xlsx")
    up.lumi_sheet = "Second"
    session_io.cache_uploaded_file("sid_1", up)
    buf = session_io.load_cached_file("sid_1", "book.xlsx")
    assert buf.lumi_sheet == "Second"
    df, _ = read_tabular(buf, "book.xlsx", sheet=buf.lumi_sheet)
    assert list(df.columns) == ["b"] and len(df) == 3


def test_cached_upload_without_sheet_and_delete(monkeypatch, tmp_path):
    _cache_env(monkeypatch, tmp_path)
    up = _Buf(_two_sheet_xlsx(), "book.xlsx")
    up.lumi_sheet = "Second"
    session_io.cache_uploaded_file("sid_2", up)
    # Re-caching the same session without a sheet clears the stale sheet file
    session_io.cache_uploaded_file("sid_2", _Buf(b"a,b\n1,2\n", "data.csv"))
    assert getattr(session_io.load_cached_file("sid_2", "data.csv"), "lumi_sheet", None) is None
    up.seek(0)
    session_io.cache_uploaded_file("sid_2", up)
    session_io.delete_cached_file("sid_2")
    assert not list(tmp_path.iterdir())


def test_landing_imports_markdown_escape():
    from views import landing
    assert landing._md_escape("*bold*_[x](y)") == r"\*bold\*\_\[x\]\(y\)"


def _set_recipe(steps):
    st.session_state["cleaning_recipe"] = steps


def test_full_download_guard_only_for_knn_or_iterative_on_big_files():
    from views.pipeline import _too_heavy_for_full
    from engine_ops import KNN_MAX_ROWS
    knn = [{"action": "fill_null", "column": "a", "value": "knn"}]
    it = [{"action": "fill_null", "column": "a", "value": "iterative"}]
    mean = [{"action": "fill_null", "column": "a", "value": "mean"}]
    _set_recipe(knn)
    assert _too_heavy_for_full(KNN_MAX_ROWS + 1)
    assert _too_heavy_for_full(None)
    assert not _too_heavy_for_full(KNN_MAX_ROWS)
    _set_recipe(it)
    assert _too_heavy_for_full(500_000)
    _set_recipe(mean)
    assert not _too_heavy_for_full(500_000)
    _set_recipe([])
    assert not _too_heavy_for_full(None)
