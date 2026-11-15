"""Wave-3 UI fixes: export sanitising, heatmap hues, sheet picker, guest reset, a11y."""
import io
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import openpyxl
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def test_formula_injection_csv_and_xlsx():
    from views.pipeline import _build_exports
    df = pd.DataFrame({"t": ["=1+1", "+x", "-y", "@z", "\tq", "\rr", "ok"], "n": [-5, 1, 2, 3, 4, 5, 6]})
    out = _build_exports(df)
    csv = pd.read_csv(io.BytesIO(out["csv"]), dtype=str, keep_default_na=False)
    assert list(csv["t"][:4]) == ["'=1+1", "'+x", "'-y", "'@z"]
    assert csv["t"][6] == "ok" and csv["n"][0] == "-5"
    ws = openpyxl.load_workbook(io.BytesIO(out["xlsx"])).active
    assert ws["A2"].value == "'=1+1"
    assert ws["B2"].value == -5  # numeric stays a number
    assert df["t"][0] == "=1+1"  # input not mutated


def test_xlsx_strips_control_chars_csv_keeps_them():
    from views.pipeline import _build_exports
    df = pd.DataFrame({"t": ["a\x00b\x07c", "x\ty\nz"]})
    out = _build_exports(df)
    assert out["xlsx"] is not None and out["xlsx_note"] is None
    ws = openpyxl.load_workbook(io.BytesIO(out["xlsx"])).active
    assert ws["A2"].value == "abc"
    assert ws["A3"].value == "x\ty\nz"


def test_safe_hue_distinct_and_stable():
    from ui_utils import get_safe_hue
    assert [get_safe_hue(i) for i in range(3)] == [200, 240, 280]
    hues = [get_safe_hue(i) for i in range(100)]
    assert len(set(hues)) == 100
    assert all(170 <= h <= 340 for h in hues)


def _xlsx(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, index=False, sheet_name=name)
    return buf.getvalue()


def test_read_tabular_sheet_notice_and_choice():
    from lumi_state.core import read_tabular
    data = _xlsx({"A": pd.DataFrame({"a": [1]}), "B": pd.DataFrame({"b": [2, 3]})})
    df, notices = read_tabular(data, "w.xlsx", kind="xlsx", sheet=0)
    assert list(df.columns) == ["a"]
    assert any("'A'" in n and "'B'" in n for n in notices)
    df2, _ = read_tabular(data, "w.xlsx", kind="xlsx", sheet="B")
    assert list(df2.columns) == ["b"]


def test_landing_sheet_picker_loads_chosen_sheet():
    data = _xlsx({"A": pd.DataFrame({"a": [1]}), "B": pd.DataFrame({"b": [2, 3]})})
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("w.xlsx", data, XLSX_MIME).run()
    assert not at.exception
    assert at.session_state.raw_data is None  # waiting for the sheet choice
    sb = [s for s in at.selectbox if s.key.startswith("xlsx_sheet_pick_")][0]
    sb.select("B").run()
    [b for b in at.button if b.key.startswith("xlsx_sheet_go_")][0].click().run()
    assert not at.exception
    assert list(at.session_state.raw_data.columns) == ["b"]
    assert at.session_state.xlsx_sheet == "B"


def test_guest_has_visible_reset_button(monkeypatch):
    import ui_utils
    monkeypatch.setattr(ui_utils, "is_authenticated_user", lambda: False)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("d.csv", b"a,b\n1,2\n3,4\n", "text/csv").run()
    assert not at.exception
    assert any(b.key == "header_guest_reset_btn" for b in at.button)


def test_css_contrast_and_reduced_motion():
    css = (Path(__file__).parent.parent / "styles" / "global.css").read_text()
    assert "prefers-reduced-motion: reduce" in css
    assert "#737373 !important;\n    margin-top: -2.25rem" not in css
