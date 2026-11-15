"""Full-file cleaned download for sampled uploads (Pipeline tab)."""
import io
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
N = 15000
NULLS = len(range(0, N, 100))  # Age is blank every 100th row
STEP = {"action": "drop_nulls", "column": "Age"}


def _csv(n):
    rows = [f"{i},{'' if i % 100 == 0 else 20 + i % 50}" for i in range(n)]
    return ("Id,Age\n" + "\n".join(rows) + "\n").encode()


def _upload(n=N):
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("big.csv", _csv(n), "text/csv").run()
    assert not at.exception
    return at


def _add_step(at, step=STEP):
    from engine import apply_recipe
    recipe = list(at.session_state.cleaning_recipe) + [step]
    at.session_state["cleaning_recipe"] = recipe
    at.session_state["current_df"] = apply_recipe(at.session_state.raw_data, recipe)[0]
    return at


def _pipeline(at):
    at.radio(key="p_mode").set_value("Cleaned Data (After)").run()
    assert not at.exception
    return at


def _btn(at, key):
    return [b for b in at.button if b.key == key]


def _full_df(at):
    return pd.read_csv(io.BytesIO(at.session_state.full_export["csv"]))


def test_full_download_matches_engine_on_full_frame():
    from engine import apply_recipe
    from lumi_state.core import downcast_dtypes
    at = _add_step(_upload())
    _pipeline(at)
    assert _btn(at, "prepare_full_download")
    assert at.session_state.full_export is None
    _btn(at, "prepare_full_download")[0].click().run()
    assert not at.exception
    full = _full_df(at)
    assert len(full) == N - NULLS
    expected, _ = apply_recipe(downcast_dtypes(pd.read_csv(io.BytesIO(_csv(N)))), [STEP])
    pd.testing.assert_frame_equal(full.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    assert at.session_state.full_export["xlsx"] is not None
    assert len(at.session_state.sample_export["csv"]) > 0
    assert not _btn(at, "prepare_full_download")  # buttons shown instead
    assert any("Full result ready" in s.value for s in at.success)


def test_cache_invalidated_when_step_added():
    at = _add_step(_upload())
    _pipeline(at)
    _btn(at, "prepare_full_download")[0].click().run()
    assert at.session_state.full_export is not None
    _add_step(at, {"action": "drop_column", "column": "Id"}).run()
    assert at.session_state.full_export is None
    assert _btn(at, "prepare_full_download")


def test_sample_csv_cache_keyed_on_recipe():
    at = _upload()
    _pipeline(at)
    k1 = at.session_state.sample_export["key"]
    _add_step(at).run()
    assert at.session_state.sample_export["key"] != k1
    assert at.session_state.sample_export["rows"] == len(at.session_state.current_df)


def test_no_prepare_button_when_not_sampled():
    at = _upload(500)
    _pipeline(at)
    assert at.session_state.sample_info is None
    assert not _btn(at, "prepare_full_download")
    assert at.session_state.full_source is None


def test_head_path_full_download_has_all_rows(monkeypatch):
    import lumi_state.session_io as sio
    monkeypatch.setattr(sio, "LARGE_FILE_THRESHOLD_BYTES", 10)
    at = _add_step(_upload())
    assert at.session_state.sample_info["method"] == "head"
    _pipeline(at)
    _btn(at, "prepare_full_download")[0].click().run()
    assert not at.exception
    assert len(_full_df(at)) == N - NULLS


def test_xlsx_disabled_above_row_limit(monkeypatch):
    import views.pipeline as vp
    monkeypatch.setattr(vp, "EXCEL_MAX_ROWS", 12000)
    at = _upload()  # total 15000 > limit
    _pipeline(at)
    assert at.checkbox(key="full_include_xlsx").disabled
    _btn(at, "prepare_full_download")[0].click().run()
    full = at.session_state.full_export
    assert full["xlsx"] is None and len(_full_df(at)) == N


def test_xlsx_note_when_result_exceeds_limit(monkeypatch):
    import views.pipeline as vp
    monkeypatch.setattr(vp, "EXCEL_MAX_ROWS", 12000)
    full = vp._build_exports(pd.DataFrame({"a": range(12001)}))
    assert full["xlsx"] is None and "limit" in full["xlsx_note"]


def test_missing_source_shows_friendly_error():
    at = _upload()
    at.session_state["full_source"] = None
    _pipeline(at)
    _btn(at, "prepare_full_download")[0].click().run()
    assert not at.exception
    assert any("no longer available" in e.value for e in at.error)


def test_memory_error_is_friendly(monkeypatch):
    import state_manager
    def boom():
        raise MemoryError()
    monkeypatch.setattr(state_manager, "load_full_data", boom)
    at = _upload()
    _pipeline(at)
    _btn(at, "prepare_full_download")[0].click().run()
    assert not at.exception
    assert any("too large" in e.value for e in at.error)


def test_load_full_data_helper_and_no_frame_kept():
    at = _upload()
    assert at.session_state.original_full_data is None
    assert len(at.session_state.raw_data) == 10000
    src = at.session_state.full_source
    assert src["name"] == "big.csv" and src["bytes"] == _csv(N)
