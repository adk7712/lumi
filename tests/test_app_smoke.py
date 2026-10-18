"""End-to-end smoke tests for app.py using streamlit.testing.v1.AppTest."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
TIMEOUT = 30

CSV = (
    b"Id,Age,Sex,LotArea,Alley\n"
    b"1,25,male,500,Pave\n2,30,female,600,Grvl\n3,35,male,700,\n"
    b"4,40,female,650,Pave\n5,29,male,5000,Grvl\n5,29,male,5000,Grvl\n"
)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    return at


def _loaded_app() -> AppTest:
    at = _fresh_app()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("smoke.csv", CSV, "text/csv").run()
    assert not at.exception
    assert at.session_state.raw_data is not None
    return at


def _add_dedupe_step(at):
    at.selectbox(key="trans_type_select").select("Remove Duplicate Rows").run()
    _btn(at, "btn_remove_dups").click().run()


def test_every_transformation_type_renders():
    at = _loaded_app()
    for opt in list(at.selectbox(key="trans_type_select").options):
        at.selectbox(key="trans_type_select").select(opt).run()
        assert not at.exception, opt


def _btn(at, key):
    return next(b for b in at.button if b.key == key)


def test_landing_page_loads_clean():
    at = _fresh_app()
    assert not at.exception
    assert at.session_state.raw_data is None


def test_dataset_loaded_and_all_tabs_render():
    at = _loaded_app()
    labels = [t.label for t in at.tabs]
    for name in ["Overview", "Diagnostics", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"]:
        assert name in labels
    assert not at.exception
    assert len(at.session_state.intermediate_states) == 1


def test_diagnostics_with_outlier_panel_selected_columns():
    at = _loaded_app()
    at.multiselect(key="active_features").select("Age").select("LotArea").run()
    assert not at.exception


def test_pipeline_preview_modes_render():
    at = _loaded_app()
    for mode in ["Raw Data (Before)", "Cleaned Data (After)", "Python Code"]:
        at.radio(key="p_mode").set_value(mode).run()
        assert not at.exception, mode


def test_add_step_undo_redo_and_audit_log():
    at = _loaded_app()
    assert len(at.session_state.cleaning_recipe) == 0

    _add_dedupe_step(at)
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 1
    assert len(at.session_state.current_df) == 5  # duplicate row removed
    assert len(at.session_state.intermediate_states) == 2

    _btn(at, "undo_btn").click().run()
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 0
    assert len(at.session_state.current_df) == 6
    assert len(at.session_state.redo_stack) == 1

    _btn(at, "redo_btn").click().run()
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 1
    assert len(at.session_state.current_df) == 5

    # Audit log renders the step and its Remove button
    assert any(b.key == "rm_step_1" for b in at.button)
    _btn(at, "rm_step_1").click().run()
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 0


def test_pipeline_export_contains_functions():
    at = _loaded_app()
    _add_dedupe_step(at)
    at.radio(key="p_mode").set_value("Python Code").run()
    assert not at.exception
    code = "\n".join(str(c.value) for c in at.code)
    assert "def clean_data" in code
    assert "def validate_data" in code


def test_dev_email_form_hidden_without_dev_auth(monkeypatch):
    """ui_utils.is_dev_auth_enabled gates the email form (LUMI_TESTING=1 enables it by design)."""
    import ui_utils

    monkeypatch.setenv("LUMI_TESTING", "1")
    assert ui_utils.is_dev_auth_enabled() is True

    monkeypatch.delenv("LUMI_TESTING", raising=False)
    monkeypatch.delenv("LUMI_DEV_AUTH", raising=False)
    monkeypatch.setattr(ui_utils.st, "secrets", {}, raising=False)
    assert ui_utils.is_dev_auth_enabled() is False

    def dialog_script():
        import streamlit as st
        import ui_utils as u
        u.show_auth_dialog()
        st.write("done")

    at = AppTest.from_function(dialog_script, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert not any(t.key == "auth_dev_email_input" for t in at.text_input)
    assert not any(b.key == "auth_dev_email_login_btn" for b in at.button)
