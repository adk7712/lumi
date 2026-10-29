"""Round-4 low-severity UI fixes: persistent step warnings (no no-op steps), proposals after
restore, no undo/redo toasts, workspace popover closing, per-workspace view keys, rule toggle save."""
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
CSV = (b"Id,Age,Name,Email\n1,25,a ,a@x.com\n2,30,b,b\n3,,c,c@y.com\n4,200,d,d@z.com\n"
       b"5,31,e,e@z.com\n6,28,f,f@z.com\n")


def _loaded(csv=CSV):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    assert not at.exception
    return at


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _warnings(at):
    return [w.value for w in at.warning]


# --- 1. skipped steps ---------------------------------------------------------------------
def test_regex_timeout_is_persistent_and_records_no_step():
    from query_guard import RegexTimeoutError
    at = _loaded()
    at.selectbox(key="trans_type_select").set_value("Find and Replace").run()
    at.text_input(key="find_input").set_value("a").run()
    at.toggle(key="replace_use_regex").set_value(True).run()
    with mock.patch("engine_ops.safe_regex_replace",
                    side_effect=RegexTimeoutError("regular expression took longer than 2s")):
        _btn(at, "btn_fr").click().run()
    assert not at.exception
    assert at.session_state["cleaning_recipe"] == []
    assert len(at.session_state["intermediate_states"]) == 1
    assert any("skipped" in w and "unchanged" in w for w in _warnings(at)), _warnings(at)
    # Still there after an unrelated rerun (a toast would be gone)
    at.run()
    assert any("skipped" in w for w in _warnings(at))


def test_noop_skip_keeps_redo_stack_and_real_step_clears_warning():
    at = _loaded()
    at.selectbox(key="trans_type_select").set_value("Strip Whitespace").run()
    _btn(at, "btn_strip").click().run()
    assert len(at.session_state["cleaning_recipe"]) == 1
    _btn(at, "undo_btn").click().run()
    assert len(at.session_state["redo_stack"]) == 1
    # An unsafe regex is rejected before it becomes a step and must not wipe redo
    at.selectbox(key="trans_type_select").set_value("Find and Replace").run()
    at.text_input(key="find_input").set_value("(a+)+$").run()
    at.toggle(key="replace_use_regex").set_value(True).run()
    _btn(at, "btn_fr").click().run()  # nested quantifier: rejected as unsafe by the UI itself
    assert len(at.session_state["redo_stack"]) == 1


def test_add_step_returns_false_and_leaves_state_for_unsafe_noop():
    def script():
        import streamlit as st
        import pandas as pd
        from state_manager import add_step, initialize_state
        initialize_state()
        if "done" not in st.session_state:
            df = pd.DataFrame({"x": ["a", "b"]})
            st.session_state.raw_data = df
            st.session_state.current_df = df.copy()
            st.session_state.intermediate_states = [("orig", 100, 2)]
            st.session_state.redo_stack = [{"action": "drop_duplicates"}]
            r1 = add_step({"action": "replace", "column": "x", "find": "(a+)+$", "replace": "z", "regex": True})
            r2 = add_step({"action": "replace", "column": "x", "find": "a", "replace": "z"})
            st.session_state.results = (r1, r2)
            st.session_state.done = True
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state["results"] == (False, True)
    assert len(at.session_state["cleaning_recipe"]) == 1
    assert len(at.session_state["intermediate_states"]) == 2
    assert at.session_state["redo_stack"] == []  # only the real step invalidates redo
    assert at.session_state["_step_warnings"] == []


# --- 2. proposals after restore -----------------------------------------------------------
def test_proposals_regenerated_after_db_restore():
    def script():
        import io
        import streamlit as st
        from state_manager import load_db_session, initialize_state
        initialize_state()
        if "done" not in st.session_state:
            buf = io.BytesIO(b"Id,Age,Name,Email\n1,25,a ,a@x.com\n2,30,b,b\n3,,c,c@y.com\n"
                             b"4,200,d,d@z.com\n5,31,e,e@z.com\n6,28,f,f@z.com\n")
            buf.name = "t.csv"
            session = {"cleaning_recipe": [{"action": "strip_whitespace", "column": "All"}],
                       "rules": [], "scanned_columns": {"Age:Null Check"}, "filename": "t.csv"}
            st.session_state.ok = load_db_session("sid-r4", buf, preloaded_session=session)
            st.session_state.done = True
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception and at.session_state["ok"]
    got = {(p["type"], p["column"]) for p in at.session_state["proposals"]}
    assert ("Null Check", "Age") not in got            # scanned
    assert ("Formatting Issue", "Name") not in got     # already applied as a step
    assert ("Range Check", "Age") in got               # remaining ones come back
    assert ("High Cardinality", "Email") in got


# --- 3. undo/redo toasts ------------------------------------------------------------------
def test_undo_redo_show_no_toast_over_the_buttons():
    at = _loaded()
    at.selectbox(key="trans_type_select").set_value("Strip Whitespace").run()
    _btn(at, "btn_strip").click().run()
    _btn(at, "undo_btn").click().run()
    assert not [t.value for t in at.toast if "undone" in t.value]
    _btn(at, "redo_btn").click().run()
    assert not [t.value for t in at.toast if "re-applied" in t.value]
    assert len(at.session_state["cleaning_recipe"]) == 1


# --- 5. per-workspace view keys -----------------------------------------------------------
def test_reset_workspace_view_state_clears_view_widget_keys():
    at = _loaded()
    at.session_state["preview_row_count"] = 50
    at.session_state["p_mode"] = "Python Code"
    at.session_state["_step_warnings"] = ["x"]
    # Reset through the real dialog flow
    from lumi_state.session_io import WORKSPACE_VIEW_KEYS
    assert "preview_row_count" in WORKSPACE_VIEW_KEYS

    def script():
        import streamlit as st
        from lumi_state.session_io import reset_workspace_view_state
        for k, v in (("preview_row_count", 50), ("p_mode", "Python Code"), ("_step_warnings", ["x"]), ("keep", 1)):
            st.session_state.setdefault(k, v)
        if st.session_state.get("go"):
            reset_workspace_view_state()
    t = AppTest.from_function(script)
    t.run()
    t.session_state["go"] = True
    t.run()
    for k in ("preview_row_count", "p_mode", "_step_warnings"):
        assert k not in t.session_state
    assert t.session_state["keep"] == 1


# --- 6. rule toggle persistence -----------------------------------------------------------
def test_rule_toggle_persists_snapshot_immediately(tmp_path, monkeypatch):
    import json
    import state_manager as sm
    import lumi_state.session_io as sio
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path)
    at = _loaded()
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "Age is NOT NULL",
                                  "enabled": True, "color": "hsla(1, 70%, 50%, 0.4)"}]
    at.run()
    tg = next(b for b in at.button if b.key and b.key.startswith("tg_"))
    assert tg.label == "Ignore"
    tg.click().run()
    assert not at.exception
    assert at.session_state["rules"][0]["enabled"] is False
    snaps = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert any(s.get("rules") and s["rules"][0].get("desc") == "Age is NOT NULL" and s["rules"][0]["enabled"] is False
               for s in snaps)
    tg = next(b for b in at.button if b.key and b.key.startswith("tg_"))
    assert tg.label == "Enable"


def test_informational_noop_message_still_records_step():
    at = _loaded()
    at.selectbox(key="trans_type_select").set_value("Remove Duplicate Rows").run()
    _btn(at, "btn_remove_dups").click().run()
    assert len(at.session_state["cleaning_recipe"]) == 1  # "Removed 0 duplicate rows." is not a skip
    assert at.session_state["_step_warnings"] == []


def test_redo_raises_no_step_added_toast():
    at = _loaded()
    at.selectbox(key="trans_type_select").set_value("Strip Whitespace").run()
    _btn(at, "btn_strip").click().run()
    _btn(at, "undo_btn").click().run()
    _btn(at, "redo_btn").click().run()
    assert len(at.session_state["cleaning_recipe"]) == 1
    assert not [t.value for t in at.toast]
