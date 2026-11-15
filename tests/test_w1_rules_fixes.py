"""Wave 1 rules/rulebook fixes: A13, A20, A17, A18, Q11, A28, Q12, Q6."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import state_manager as sm
from query_guard import safe_query_index, validate_query_expression, UnsafeQueryError
from rule_utils import _relational_violations, evaluate_rule

APP_PATH = str(Path(__file__).parent.parent / "app.py")
CSV = (b"Id,Age,Score,Name,Alley\n1,25,100,Ann,Pave\n2,30,200,Bob,\n3,35,300,,Grvl\n")


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _loaded(csv=CSV):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    assert at.session_state.raw_data is not None
    return at


def _msgs(elements):
    return [e.value for e in elements]


# ---- A13 ----
def test_a13_range_inputs_follow_target_column():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Range Check").run()
    at.selectbox(key="rule_target_col").select("Age").run()
    assert at.number_input(key="range_min_input_Age").value == 25
    at.selectbox(key="rule_target_col").select("Score").run()
    assert at.number_input(key="range_min_input_Score").value == 100
    assert at.number_input(key="range_max_input_Score").value == 300


# ---- A20 ----
def test_a20_range_min_gt_max_rejected():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Range Check").run()
    at.selectbox(key="rule_target_col").select("Age").run()
    at.number_input(key="range_min_input_Age").set_value(50.0)
    at.number_input(key="range_max_input_Age").set_value(10.0).run()
    _btn(at, "btn_add_range").click().run()
    assert at.session_state.rules == []
    assert any("cannot be greater" in m for m in _msgs(at.error))


@pytest.mark.parametrize("val", ["", "   ", "nan", "NaN"])
def test_a20_relational_constant_rejected(val):
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Relational Check").run()
    at.radio(key="rel_target_type_radio").set_value("Constant Value").run()
    at.text_input(key="rel_val_input").set_value(val).run()
    _btn(at, "btn_add_rel_val").click().run()
    assert at.session_state.rules == []
    assert at.error


def test_a20_informational_empty_rejected():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Informational").run()
    at.text_area(key="info_note_input").set_value("   ").run()
    _btn(at, "btn_add_info").click().run()
    assert at.session_state.rules == []
    assert any("empty" in m for m in _msgs(at.error))


def test_a20_range_bounds_helper():
    from views.rulebook import _range_bounds_error
    assert _range_bounds_error(float("inf"), 1.0)
    assert _range_bounds_error(float("nan"), 1.0)
    assert _range_bounds_error(5, 1)
    assert _range_bounds_error(1, 5) is None


def test_a20_duplicate_rule_not_added_and_warns():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Null Check").run()
    _btn(at, "btn_add_null").click().run()
    assert len(at.session_state.rules) == 1
    _btn(at, "btn_add_null").click().run()
    assert len(at.session_state.rules) == 1
    assert any("identical rule" in m for m in _msgs(at.warning))


def test_a20_add_rule_returns_bool_and_ignores_flags_and_desc():
    st.session_state.clear()
    st.session_state.rules = []
    r = {"type": "Range Check", "col": "a", "min": 0.0, "max": 5.0, "desc": "one"}
    assert sm.add_rule(r) is True
    assert sm.add_rule({**r, "desc": "two", "enabled": False, "resolved": True}) is False
    assert sm.add_rule({**r, "max": 6.0}) is True
    assert len(st.session_state.rules) == 2
    # Informational notes differ by desc
    assert sm.add_rule({"type": "Informational", "desc": "a"}) is True
    assert sm.add_rule({"type": "Informational", "desc": "b"}) is True
    assert sm.add_rule({"type": "Informational", "desc": "a"}) is False


# ---- A17 ----
def test_a17_remove_step_discards_scanned_and_regenerates(monkeypatch):
    from views import audit_log
    st.session_state.clear()
    df = pd.DataFrame({"a": [1, 2, 3]})
    ss = st.session_state
    ss.raw_data = df
    ss.current_df = df
    ss.rules = []
    ss.redo_stack = []
    ss.cleaning_recipe = [{"action": "drop_duplicates", "_source_proposal_type": "Duplicates", "_source_proposal_col": "__dataset__"}]
    ss.intermediate_states = [("Original", 100, 3), ("drop_duplicates", 100, 3)]
    ss.scanned_columns = {"__dataset__:Duplicates", "x:Other"}
    called = []
    monkeypatch.setattr(audit_log, "regenerate_proposals", lambda: called.append(1))
    for n in ("save_session_state", "save_db_session", "_update_project_cache"):
        monkeypatch.setattr(audit_log, n, lambda *a, **k: None)
    audit_log._remove_step(1)
    assert ss.scanned_columns == {"x:Other"}
    assert called == [1]


# ---- A18 ----
def test_a18_replay_skips_error_only_steps():
    st.session_state.clear()
    df = pd.DataFrame({"name": ["a", "b"], "age": [1, 2]})
    ss = st.session_state
    ss.raw_data = df
    ss.current_df = df
    ss.cleaning_recipe = []
    ss.intermediate_states = [("Original", 100, 2)]
    ss.rules = []
    ss.redo_stack = []
    bad = {"action": "log_transform", "column": "name"}  # log of text: error only
    good = {"action": "drop_duplicates"}
    sm.replay_steps([bad, good], persist=False)
    assert ss.cleaning_recipe == [good]
    assert len(ss.intermediate_states) == 2
    assert any("skipped 1 step" in w and "log_transform on name" in w for w in ss["_step_warnings"])


# ---- Q11 ----
def test_q11_text_null_check_has_no_mean_median():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Null Check").run()
    at.selectbox(key="rule_target_col").select("Name").run()
    _btn(at, "btn_add_null").click().run()
    sel = next(s for s in at.selectbox if s.key and s.key.startswith("res_"))
    assert "Fill with Mean" not in sel.options and "Drop Rows" in sel.options


def test_q11_numeric_null_check_keeps_mean_median():
    csv = b"Id,Age\n1,25\n2,\n3,35\n"
    at = _loaded(csv)
    at.selectbox(key="rule_type_select").select("Null Check").run()
    at.selectbox(key="rule_target_col").select("Age").run()
    _btn(at, "btn_add_null").click().run()
    sel = next(s for s in at.selectbox if s.key and s.key.startswith("res_"))
    assert "Fill with Mean" in sel.options and "Fill with Median" in sel.options


# ---- A28 ----
def test_a28_no_match_warning_survives_rerun():
    at = _loaded()
    at.selectbox(key="rule_type_select").select("Custom Expression").run()
    at.text_input(key="custom_query_input").set_value("Age > 1000")
    at.run()
    next(b for b in at.button if b.label == "Add Rule").click().run()
    assert len(at.session_state.rules) == 1
    assert any("matches no rows" in m for m in _msgs(at.warning))
    at.run()
    assert not any("matches no rows" in m for m in _msgs(at.warning))


# ---- Q12 ----
def test_q12_str_methods_ignore_nan():
    df = pd.DataFrame({"first name": ["Jo", None, "Al"], "x": ["a", None, "b"]})
    assert list(safe_query_index(df, "x.str.contains('a')")) == [0]
    assert list(safe_query_index(df, "x.str.startswith('a') or x.str.endswith('b')")) == [0, 2]
    assert list(safe_query_index(df, "`first name`.str.startswith('J')")) == [0]
    assert list(safe_query_index(df, "x.str.contains('a', na=True)")) == [0, 1]


def test_q12_syntax_error_hint():
    with pytest.raises(UnsafeQueryError, match="backticks"):
        validate_query_expression("first name == 'a'")


def test_q12_export_compiles_and_runs():
    from codegen import generate_pipeline_code
    rules = [{"type": "Custom Expression", "query": "x.str.contains('a')", "desc": "d", "enabled": True}]
    script = generate_pipeline_code([], rules)
    compile(script, "<export>", "exec")
    assert "_default_na_false" in script
    ns = {"__name__": "exported"}
    exec(script, ns)
    out = ns["validate_data"](pd.DataFrame({"x": ["a", None, "b"]}))
    assert out is not None


# ---- Q6 ----
def test_q6_thousands_strings_compare_numerically():
    a = pd.Series(["25,649", "9,000", None, "100"])
    # valid state: a > 10000 -> 9,000 and 100 violate; null never violates
    assert _relational_violations(a, 10000.0, ">").tolist() == [False, True, False, True]
    b = pd.Series(["1,000", "10,000", "1", "2"])
    assert _relational_violations(a, b, ">").tolist() == [False, True, False, False]


def test_q6_non_numeric_strings_stay_lexical():
    a = pd.Series(["abc", "1,2", "x"])
    out = _relational_violations(a, "b", "<")
    assert out.dtype == bool and out.tolist() == [False, False, True]


def test_q6_requires_at_least_one_comma():
    a = pd.Series(["9", "10"])  # plain digits: untouched (lexical compare)
    assert _relational_violations(a, "5", ">").tolist() == [False, True]
