"""Undo persists to DB, redo preserves the stack, undo un-resolves rules."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
from streamlit.testing.v1 import AppTest

import state_manager as sm
from engine import apply_recipe

APP_PATH = str(Path(__file__).parent.parent / "app.py")
CSV = b"Id,Age,Sex,LotArea,Alley\n1,25,male,500,Pave\n2,30,female,600,Grvl\n3,35,male,700,\n"


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _loaded():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", CSV, "text/csv").run()
    return at


def _add_strip(at):
    at.selectbox(key="trans_type_select").select("Strip Whitespace").run()
    _btn(at, "btn_strip").click().run()


def _invariant(at):
    exp = apply_recipe(at.session_state.raw_data.copy(), list(at.session_state.cleaning_recipe))[0]
    pd.testing.assert_frame_equal(at.session_state.current_df.reset_index(drop=True), exp.reset_index(drop=True))
    assert len(at.session_state.intermediate_states) == len(at.session_state.cleaning_recipe) + 1


def test_undo_persists_to_db(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "save_db_session", lambda *a, **k: calls.append(1))
    at = _loaded()
    _add_strip(at)
    n = len(calls)
    _btn(at, "undo_btn").click().run()
    assert len(calls) > n
    assert len(at.session_state.cleaning_recipe) == 0
    _invariant(at)


def test_double_undo_double_redo_and_new_action_clears():
    at = _loaded()
    _add_strip(at)
    _add_strip(at)
    steps = [dict(s) for s in at.session_state.cleaning_recipe]
    assert len(steps) == 2
    _btn(at, "undo_btn").click().run()
    _btn(at, "undo_btn").click().run()
    assert at.session_state.cleaning_recipe == [] and len(at.session_state.redo_stack) == 2
    _invariant(at)
    _btn(at, "redo_btn").click().run()
    assert len(at.session_state.cleaning_recipe) == 1 and len(at.session_state.redo_stack) == 1
    assert not _btn(at, "redo_btn").disabled
    _invariant(at)
    _btn(at, "redo_btn").click().run()
    assert at.session_state.cleaning_recipe == steps
    assert at.session_state.redo_stack == [] and _btn(at, "redo_btn").disabled
    _invariant(at)
    # new action after undo clears redo
    _btn(at, "undo_btn").click().run()
    assert len(at.session_state.redo_stack) == 1
    _add_strip(at)
    assert at.session_state.redo_stack == []
    _invariant(at)


def test_undo_unresolves_rule():
    at = _loaded()
    rule = {"type": "Null Check", "col": "Alley", "desc": "Alley is NOT NULL", "enabled": True,
            "color": "hsla(10, 70%, 50%, 0.4)"}
    at.session_state.rules.append(rule)
    at.run()
    step = {"action": "drop_nulls", "column": "Alley"}
    at.session_state.cleaning_recipe.append(step)
    at.session_state.current_df = apply_recipe(at.session_state.raw_data.copy(), [step])[0]
    at.session_state.intermediate_states.append(("drop_nulls on Alley", 100, 2))
    at.session_state.rules[0]["resolved"] = True
    at.run()
    _btn(at, "undo_btn").click().run()
    assert not at.session_state.rules[0].get("resolved")
    assert len(at.session_state.cleaning_recipe) == 0


def _resolved_null_rule_with_step(at):
    rule = {"type": "Null Check", "col": "Alley", "desc": "Alley is NOT NULL", "enabled": True,
            "color": "hsla(10, 70%, 50%, 0.4)"}
    at.session_state.rules.append(rule)
    at.run()
    step = {"action": "drop_nulls", "column": "Alley"}
    at.session_state.cleaning_recipe.append(step)
    at.session_state.current_df = apply_recipe(at.session_state.raw_data.copy(), [step])[0]
    at.session_state.intermediate_states.append(("drop_nulls on Alley", 100, 2))
    at.session_state.rules[0]["resolved"] = True
    at.run()


def test_redo_restores_resolved_flag():
    at = _loaded()
    _resolved_null_rule_with_step(at)
    _btn(at, "undo_btn").click().run()
    assert not at.session_state.rules[0].get("resolved")
    _btn(at, "redo_btn").click().run()
    assert at.session_state.rules[0].get("resolved")
    assert "_unresolved_rules" not in at.session_state.cleaning_recipe[-1]
    _invariant(at)


def test_audit_remove_unresolves_rule():
    at = _loaded()
    _resolved_null_rule_with_step(at)
    _btn(at, "rm_step_1").click().run()
    assert len(at.session_state.cleaning_recipe) == 0
    assert not at.session_state.rules[0].get("resolved")
