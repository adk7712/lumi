"""Column renames: rules/proposals follow the rename before the save, and undo reverts them."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import state_manager as sm

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _base(df):
    st.session_state["raw_data"] = df
    st.session_state["current_df"] = df.copy()
    st.session_state["cleaning_recipe"] = []
    st.session_state["rules"] = []
    st.session_state["proposals"] = []
    st.session_state["scanned_columns"] = set()
    st.session_state["active_features"] = []
    st.session_state["redo_stack"] = []
    st.session_state["session_id"] = None
    st.session_state["intermediate_states"] = [sm.build_original_entry(df)]


def test_rules_are_renamed_before_the_save(monkeypatch):
    _base(pd.DataFrame({"Total Amount": [1, 2, None]}))
    st.session_state["rules"] = [{"type": "Null Check", "col": "Total Amount", "desc": "Total Amount is NOT NULL"}]
    saved_cols = []
    monkeypatch.setattr(sm, "save_session_state", lambda: None)
    monkeypatch.setattr(sm, "save_db_session", lambda: saved_cols.append(st.session_state["rules"][0]["col"]))
    sm.add_step({"action": "normalize_column_names", "value": "snake_case"}, quiet=True)
    assert saved_cols == ["total_amount"]
    assert st.session_state["rules"][0]["desc"] == "total_amount is NOT NULL"


def test_accept_all_applies_later_proposals_to_new_names(monkeypatch):
    from views import rulebook
    _base(pd.DataFrame({"First Name": ["a", "b"], "City X": ["c", "c"], "Total Amount": [1.0, 2.0]}))
    monkeypatch.setattr(sm, "save_session_state", lambda: None)
    monkeypatch.setattr(sm, "save_db_session", lambda: None)
    st.session_state["proposals"] = [
        {"type": "column_names", "column": "__dataset__",
         "rule_data": {"action": "normalize_column_names", "value": "snake_case"}},
        {"type": "Constant Value", "column": "City X", "rule_data": {"action": "drop_column", "column": "City X"}},
        {"type": "Range Check", "column": "Total Amount",
         "rule_data": {"type": "Range Check", "col": "Total Amount", "min": 0.0, "max": 5.0,
                       "desc": "Total Amount within statistical bounds"}},
    ]
    rulebook.handle_accept_all()
    df = st.session_state["current_df"]
    assert list(df.columns) == ["first_name", "total_amount"]
    assert st.session_state["cleaning_recipe"][1] == {
        "action": "drop_column", "column": "city_x",
        "_source_proposal_type": "Constant Value", "_source_proposal_col": "city_x"}
    rule = st.session_state["rules"][0]
    assert rule["col"] == "total_amount" and rule["desc"] == "total_amount within statistical bounds"


def test_replay_does_not_chain_renames(monkeypatch):
    # a -> b, then c -> a: replaying must not move the rule on the new 'a' (originally 'c') to 'b'.
    _base(pd.DataFrame({"a": [1], "c": [2]}))
    monkeypatch.setattr(sm, "save_session_state", lambda: None)
    monkeypatch.setattr(sm, "save_db_session", lambda: None)
    st.session_state["rules"] = [{"type": "Null Check", "col": "a", "desc": "a is NOT NULL"}]
    sm.replay_steps([{"action": "rename_column", "column": "a", "value": "b"},
                     {"action": "rename_column", "column": "c", "value": "a"}])
    assert st.session_state["rules"][0]["col"] == "a"


def _btn(at, key):
    return next(b for b in at.button if b.key == key)


def test_undo_rename_restores_rule_column():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    csv = b"Id,Age\n1,25\n2,\n3,35\n"
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "Age is NOT NULL", "enabled": True,
                                  "color": "hsla(0, 70%, 50%, 0.4)"}]
    at.selectbox(key="trans_type_select").select("Rename Column").run()
    at.selectbox(key="rename_target_col").select("Age").run()
    at.text_input(key="rename_new_name_input").input("years").run()
    _btn(at, "btn_rename").click().run()
    assert at.session_state.rules[0]["col"] == "years"
    _btn(at, "undo_btn").click().run()
    assert "Age" in at.session_state.current_df.columns
    assert at.session_state.rules[0]["col"] == "Age"
    assert at.session_state.rules[0]["desc"] == "Age is NOT NULL"
    _btn(at, "redo_btn").click().run()
    assert at.session_state.rules[0]["col"] == "years"
