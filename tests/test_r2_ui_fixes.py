"""Round-2 UI regressions: guarded step removal, unsafe regex, violation browser, rerender."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _loaded(csv):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    assert not at.exception
    return at


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def test_remove_blocked_when_later_step_uses_created_name():
    at = _loaded(b"Total Amount,B\n1,2\n2,3\n3,4\n")
    at.session_state["_is_testing"] = True
    # Drive through the real UI: normalize names, then rename the normalized column.
    at.selectbox(key="trans_type_select").select("Normalize Column Names").run()
    _btn(at, "btn_norm_cols").click().run()
    assert at.session_state["current_df"].columns.tolist()[0] == "total_amount"
    at.selectbox(key="trans_type_select").select("Rename Column").run()
    at.selectbox(key="rename_target_col").select("total_amount").run()
    at.text_input(key="rename_new_name_input").set_value("amt").run()
    _btn(at, "btn_rename").click().run()
    assert "amt" in at.session_state["current_df"].columns
    # Rulebook re-rendered in the same interaction (no stale name in the new-rule column list)
    assert not any("total_amount" in (s.options or []) for s in at.selectbox if s.key == "rule_col_select")
    recipe_len = len(at.session_state["cleaning_recipe"])
    _btn(at, "rm_step_1").click().run()
    assert len(at.session_state["cleaning_recipe"]) == recipe_len
    assert any("Cannot remove step 2" in e.value for e in at.error)


def test_remove_allowed_when_nothing_depends_on_it():
    at = _loaded(b"Total Amount,B\n1,2\n2,3\n")
    at.selectbox(key="trans_type_select").select("Normalize Column Names").run()
    _btn(at, "btn_norm_cols").click().run()
    n = len(at.session_state["cleaning_recipe"])
    _btn(at, "rm_step_1").click().run()
    assert len(at.session_state["cleaning_recipe"]) == n - 1
    assert not at.error


def test_unsafe_regex_adds_no_step():
    at = _loaded(b"A\naaa\nbbb\n")
    at.selectbox(key="trans_type_select").select("Find and Replace").run()
    at.text_input(key="find_input").set_value("(a+)+$").run()
    at.toggle(key="replace_use_regex").set_value(True).run()
    _btn(at, "btn_fr").click().run()
    assert at.session_state["cleaning_recipe"] == []
    assert any("not applied" in e.value for e in at.error)


def test_violation_browser_handles_duplicate_index():
    at = _loaded(b"A\n1\n\n3\n")
    import pandas as pd
    df = at.session_state["current_df"]
    dup = pd.concat([df, df], ignore_index=False)
    at.session_state["current_df"] = dup
    at.session_state["rules"] = [{"type": "Null Check", "col": "A", "desc": "A is NOT NULL",
                                  "enabled": True, "color": "red"}]
    at.run()
    assert not at.exception
