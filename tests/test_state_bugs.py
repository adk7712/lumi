import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
os.environ["LUMI_TESTING"] = "1"
import logging
import pandas as pd
import streamlit as st
import state_manager as sm


def _base(monkeypatch=None):
    df = pd.DataFrame({"age": [1, None, 3], "page": [1, 2, 3]})
    st.session_state["raw_data"] = df
    st.session_state["current_df"] = df.copy()
    st.session_state["cleaning_recipe"] = []
    st.session_state["rules"] = []
    st.session_state["scanned_columns"] = set()
    st.session_state["active_features"] = []
    st.session_state["redo_stack"] = []
    st.session_state["session_id"] = None
    st.session_state["intermediate_states"] = [sm.build_original_entry(df)]
    return df


def test_reset_intermediate_states_reseeds_original():
    df = _base()
    st.session_state["intermediate_states"] = []
    sm.reset_intermediate_states()
    assert st.session_state["intermediate_states"] == [("Original Data", sm.calculate_health(df), 3)]


def test_handle_clear_all_reseeds_original(monkeypatch):
    from views import rulebook
    _base()
    st.session_state["intermediate_states"].append(("x", 50, 3))
    st.session_state["proposals"] = []
    monkeypatch.setattr(rulebook, "save_session_state", lambda: None)
    monkeypatch.setattr(rulebook, "save_db_session", lambda: None)
    rulebook.handle_clear_all()
    assert len(st.session_state["intermediate_states"]) == 1
    assert st.session_state["intermediate_states"][0][0] == "Original Data"


def test_rename_substring_column_not_corrupted():
    _base()
    st.session_state["rules"] = [
        {"type": "Null Check", "col": "age", "desc": "age is NOT NULL"},
        {"type": "Null Check", "col": "page", "desc": "page is NOT NULL"},
        {"type": "Relational Check", "col_a": "age", "col_b": "page", "op": ">", "target_type": "Feature", "desc": "age > page"},
    ]
    st.session_state["scanned_columns"] = {"age:Null Check", "page:Null Check"}
    st.session_state["active_features"] = ["age", "page"]
    sm.sync_column_rename("age", "years")
    r = st.session_state["rules"]
    assert r[0]["desc"] == "years is NOT NULL"
    assert r[1]["desc"] == "page is NOT NULL" and r[1]["col"] == "page"
    assert r[2]["desc"] == "years > page" and r[2]["col_a"] == "years" and r[2]["col_b"] == "page"
    assert st.session_state["scanned_columns"] == {"years:Null Check", "page:Null Check"}
    assert st.session_state["active_features"] == ["years", "page"]


def test_rename_custom_expression_bare_backtick_and_literals():
    _base()
    q = "age > 100 or `age` < 0 or page == 'age' or page > 5"
    st.session_state["rules"] = [{"type": "Custom Expression", "query": q, "desc": f"Violates: {q}"}]
    sm.sync_column_rename("age", "years")
    rule = st.session_state["rules"][0]
    assert rule["query"] == "years > 100 or `years` < 0 or page == 'age' or page > 5"
    assert rule["desc"] == f"Violates: {rule['query']}"


def test_rename_to_non_identifier_uses_backticks():
    _base()
    st.session_state["rules"] = [{"type": "Custom Expression", "query": "age > 1", "desc": "Violates: age > 1"}]
    sm.sync_column_rename("age", "user age")
    assert st.session_state["rules"][0]["query"] == "`user age` > 1"


def test_rename_updates_source_proposal_tags():
    _base()
    st.session_state["cleaning_recipe"] = [{"action": "drop_nulls", "column": "age", "_source_proposal_col": "age", "_source_proposal_type": "Null"}]
    sm.sync_column_rename("age", "years")
    assert st.session_state["cleaning_recipe"][0]["_source_proposal_col"] == "years"
    assert st.session_state["cleaning_recipe"][0]["column"] == "age"  # history untouched


def test_add_step_quiet_no_toast_no_persist(monkeypatch):
    _base()
    toasts, saves = [], []
    monkeypatch.setattr(st, "toast", lambda m, *a, **k: toasts.append(m))
    monkeypatch.setattr(sm, "save_session_state", lambda: saves.append("s"))
    monkeypatch.setattr(sm, "save_db_session", lambda: saves.append("d"))
    sm.add_step({"action": "drop_nulls", "column": "age"}, quiet=True, persist=False)
    assert toasts == [] and saves == []
    assert len(st.session_state["cleaning_recipe"]) == 1


def test_replay_steps_persists_once(monkeypatch):
    _base()
    toasts, saves = [], []
    monkeypatch.setattr(st, "toast", lambda m, *a, **k: toasts.append(m))
    monkeypatch.setattr(sm, "save_session_state", lambda: saves.append("s"))
    monkeypatch.setattr(sm, "save_db_session", lambda: saves.append("d"))
    sm.replay_steps([{"action": "drop_nulls", "column": "age"}, {"action": "drop_duplicates"}])
    assert toasts == []
    assert saves == ["s", "d"]
    assert len(st.session_state["cleaning_recipe"]) == 2
    assert len(st.session_state["intermediate_states"]) == 3


def test_save_session_state_failure_logs_and_warns_once(monkeypatch, caplog):
    _base()
    st.session_state["last_file_hash"] = "abc"
    st.session_state.pop("_save_warning_shown", None)
    toasts = []
    monkeypatch.setattr(st, "toast", lambda m, *a, **k: toasts.append(m))
    monkeypatch.setattr(sm, "CACHE_DIR", Path("/nonexistent_root_dir/x/y"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    with caplog.at_level(logging.ERROR, logger="lumi.state"):
        sm.save_session_state()
        sm.save_session_state()
    assert any("local session cache" in r.message for r in caplog.records)
    assert len(toasts) == 1


def test_save_db_session_rotates_id_when_refused(monkeypatch, caplog):
    _base()
    st.session_state["session_id"] = "old-sid"
    calls = []

    def fake_save(session_id, **kw):
        calls.append(session_id)
        return session_id != "old-sid"

    import persistence
    monkeypatch.setattr(persistence, "save_session", fake_save)
    with caplog.at_level(logging.WARNING, logger="lumi.state"):
        sm.save_db_session()
    assert len(calls) == 2 and calls[0] == "old-sid" and calls[1] != "old-sid"
    assert st.session_state["session_id"] == calls[1]
    assert any("refused" in r.message for r in caplog.records)
