"""Destructive actions (logo reset, Clear Rules) require confirmation."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
CSV = b"Id,Age,Sex,LotArea,Alley\n1,25,male,500,Pave\n2,30,female,600,Grvl\n3,35,male,700,\n"


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _loaded():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", CSV, "text/csv").run()
    assert at.session_state.raw_data is not None
    return at


def _with_rule_and_step(at):
    at.selectbox(key="rule_type_select").select("Informational").run()
    at.text_area(key="info_note_input").input("note").run()
    _btn(at, "btn_add_info").click().run()
    at.selectbox(key="trans_type_select").select("Strip Whitespace").run()
    _btn(at, "btn_strip").click().run()
    assert len(at.session_state.rules) == 1 and len(at.session_state.cleaning_recipe) == 1


def test_no_data_has_no_logo_and_is_harmless():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert _btn(at, "lumi_logo") is None


def test_logo_shows_confirmation_and_does_not_reset():
    at = _loaded()
    _with_rule_and_step(at)
    _btn(at, "lumi_logo").click().run()
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 1 and len(at.session_state.rules) == 1
    assert _btn(at, "confirm_reset_btn") is not None
    # AppTest cannot re-run a dialog fragment, so the confirm click itself is not exercised here;
    # confirming the dialog only calls handle_reset (initialize_state(from_reset=True)).


def test_logo_cancel_keeps_workspace():
    at = _loaded()
    _with_rule_and_step(at)
    _btn(at, "lumi_logo").click().run()
    _btn(at, "cancel_reset_btn").click().run()
    assert len(at.session_state.cleaning_recipe) == 1 and len(at.session_state.rules) == 1
    assert at.session_state.raw_data is not None


def test_clear_rules_confirm_clears_rules_keeps_steps():
    at = _loaded()
    _with_rule_and_step(at)
    _btn(at, "clear_all_rules_btn").click().run()
    assert len(at.session_state.rules) == 1
    _btn(at, "confirm_clear_rules_btn").click().run()
    assert not at.exception
    assert len(at.session_state.rules) == 0
    assert len(at.session_state.cleaning_recipe) == 1


def test_clear_rules_cancel_keeps_everything():
    at = _loaded()
    _with_rule_and_step(at)
    _btn(at, "clear_all_rules_btn").click().run()
    _btn(at, "cancel_clear_rules_btn").click().run()
    assert len(at.session_state.rules) == 1
    assert len(at.session_state.cleaning_recipe) == 1
