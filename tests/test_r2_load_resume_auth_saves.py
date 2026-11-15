"""R2: resume header check via recipe replay, dev-auth sign-in button, background save retry."""
import os
import sys
import threading
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import pytest

import ui_utils
from lumi_state import session_io


# --- 4. resume: original headers, not scanned (post-cleaning) names --------------------

def test_recipe_fits_identical_original_headers():
    from views.landing import recipe_fits_headers
    header = pd.DataFrame(columns=["Customer Name", "Age"])
    recipe = [
        {"action": "rename_column", "column": "Customer Name", "value": "name"},
        {"action": "fill_null", "column": "name", "value": "x", "method": "value"},
        {"action": "normalize_column_names", "value": "snake_case"},
        {"action": "drop_column", "column": "age"},
    ]
    assert recipe_fits_headers(header, recipe) == []


def test_recipe_does_not_fit_other_headers():
    from views.landing import recipe_fits_headers
    header = pd.DataFrame(columns=["foo", "bar"])
    recipe = [{"action": "rename_column", "column": "Customer Name", "value": "name"}]
    problems = recipe_fits_headers(header, recipe)
    assert problems and "Customer Name" in problems[0]


def test_empty_recipe_always_fits():
    from views.landing import recipe_fits_headers
    assert recipe_fits_headers(pd.DataFrame(columns=["a"]), []) == []


# --- 5. dev auth reachable when OIDC is configured -------------------------------------

@pytest.mark.parametrize("dev", [True, False])
def test_sign_in_button_routes_to_dialog_when_dev_auth(monkeypatch, dev):
    monkeypatch.setattr(ui_utils, "is_auth_configured", lambda: True)
    monkeypatch.setattr(ui_utils, "is_dev_auth_enabled", lambda: dev)
    dialog = mock.MagicMock()
    monkeypatch.setattr(ui_utils, "show_auth_dialog", dialog)
    with mock.patch.object(ui_utils.st, "button", return_value=True) as btn:
        ui_utils.render_sign_in_button("k")
    kwargs = btn.call_args.kwargs
    if dev:
        assert "on_click" not in kwargs
        dialog.assert_called_once()
    else:  # production: unchanged, straight to Google
        assert kwargs["on_click"] is ui_utils.st.login
        assert kwargs["kwargs"] == {"provider": "google"}
        dialog.assert_not_called()


def test_sign_in_button_without_oidc_opens_dialog(monkeypatch):
    monkeypatch.setattr(ui_utils, "is_auth_configured", lambda: False)
    monkeypatch.setattr(ui_utils, "is_dev_auth_enabled", lambda: False)
    dialog = mock.MagicMock()
    monkeypatch.setattr(ui_utils, "show_auth_dialog", dialog)
    with mock.patch.object(ui_utils.st, "button", return_value=True):
        ui_utils.render_sign_in_button("k")
    dialog.assert_called_once()


# --- 7. background save retry ----------------------------------------------------------

@pytest.fixture
def clean_saves(monkeypatch):
    monkeypatch.setattr(session_io, "_SAVE_RETRY_DELAYS", (0, 0))
    for m in (session_io._pending_saves, session_io._failed_saves, session_io._rotated_sessions):
        m.clear()
    session_io._running_saves.clear()
    yield
    for m in (session_io._pending_saves, session_io._failed_saves, session_io._rotated_sessions):
        m.clear()
    session_io._running_saves.clear()


def _drain(sid, job):
    session_io._pending_saves[sid] = job
    session_io._running_saves.add(sid)
    session_io._drain_saves(sid)


def test_errored_save_is_retried_then_succeeds(clean_saves):
    calls = []

    def job(target):
        calls.append(target)
        return None if len(calls) < 3 else target  # two DB errors, then success

    _drain("s1", job)
    assert len(calls) == 3
    assert "s1" not in session_io._failed_saves


def test_errored_save_gives_up_and_records_failure(clean_saves):
    calls = []

    def job(target):
        calls.append(target)
        raise RuntimeError("db down")

    _drain("s2", job)
    assert len(calls) == 1 + len(session_io._SAVE_RETRY_DELAYS)
    assert "s2" in session_io._failed_saves


def test_refusal_is_never_retried(clean_saves):
    calls = []

    def job(target):
        calls.append(target)
        return False  # refused (and the rotated save failed): not retryable

    _drain("s3", job)
    assert len(calls) == 1
    assert "s3" in session_io._failed_saves


def test_bg_save_returns_none_for_db_error_and_false_for_refusal(monkeypatch):
    import persistence
    import streamlit as st
    captured = {}
    monkeypatch.setattr(session_io, "_is_sync_save", lambda: False)
    monkeypatch.setattr(session_io, "_submit_save", lambda sid, job: captured.setdefault("job", job))
    monkeypatch.setattr(session_io, "_cache_saved_project", lambda *a, **k: None)
    st.session_state.session_id = "sx"
    try:
        session_io.save_db_session()
        job = captured["job"]
        monkeypatch.setattr(persistence, "save_session", lambda **kw: None)
        assert job("sx") is None
        monkeypatch.setattr(persistence, "save_session", lambda **kw: False)
        assert job("sx") is False
        monkeypatch.setattr(persistence, "save_session", lambda **kw: True)
        assert job("sx") == "sx"
    finally:
        st.session_state.pop("session_id", None)
