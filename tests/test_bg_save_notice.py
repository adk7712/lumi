"""Background DB save outcomes are surfaced on the next main-thread run (A6)."""
import os
from concurrent.futures import Future

import pytest
import streamlit as st

import state_manager as sm
from lumi_state import session_io


class _ImmediateExecutor:
    def submit(self, fn, *a, **k):
        f = Future()
        try:
            f.set_result(fn(*a, **k))
        except Exception as e:  # pragma: no cover
            f.set_exception(e)
        return f


@pytest.fixture
def prod_mode(monkeypatch):
    monkeypatch.delenv("LUMI_TESTING", raising=False)
    monkeypatch.setattr(session_io, "_db_executor", _ImmediateExecutor())
    session_io._rotated_sessions.clear()
    session_io._failed_saves.clear()
    st.session_state.clear()
    st.session_state["session_id"] = "sid-1"
    st.session_state["_is_testing"] = False
    toasts = []
    monkeypatch.setattr(st, "toast", lambda m, *a, **k: toasts.append(m))
    monkeypatch.setattr(session_io, "_set_session_cookie", lambda s: None)
    yield toasts
    session_io._rotated_sessions.clear()
    session_io._failed_saves.clear()
    st.session_state.clear()


def test_failure_toast_once_on_next_run(prod_mode, monkeypatch):
    import persistence
    monkeypatch.setattr(persistence, "save_session", lambda **kw: False)
    sm.save_db_session()
    assert prod_mode == []  # background: nothing shown yet
    assert "sid-1" in session_io._failed_saves or "sid-1" not in st.session_state.get("session_id", "")
    # the failed save rotated nothing (retry failed too) -> key is the old sid
    sm._apply_rotated_session()
    assert len(prod_mode) == 1
    assert "sid-1" not in session_io._failed_saves
    # repeated runs / failures do not spam
    sm._apply_rotated_session()
    session_io._on_bg_done(_done(False), "sid-1")
    sm._apply_rotated_session()
    assert len(prod_mode) == 1


def _done(v):
    f = Future()
    f.set_result(v)
    return f


def test_success_no_toast(prod_mode, monkeypatch):
    import persistence
    monkeypatch.setattr(persistence, "save_session", lambda **kw: True)
    sm.save_db_session()
    sm._apply_rotated_session()
    assert prod_mode == []
    assert not session_io._failed_saves and not session_io._rotated_sessions


def test_rotation_still_applied_and_consumed(prod_mode, monkeypatch):
    import persistence
    monkeypatch.setattr(persistence, "save_session", lambda session_id, **kw: session_id != "sid-1")
    sm.save_db_session()
    assert session_io._rotated_sessions.get("sid-1")
    new = session_io._rotated_sessions["sid-1"]
    sm._apply_rotated_session()
    assert st.session_state["session_id"] == new
    assert not session_io._rotated_sessions
    assert prod_mode == []


def test_crashed_future_records_failure(prod_mode):
    f = Future()
    f.set_exception(RuntimeError("boom"))
    session_io._on_bg_done(f, "sid-1")
    sm._apply_rotated_session()
    assert len(prod_mode) == 1


def test_later_success_clears_stale_failure(prod_mode):
    session_io._on_bg_done(_done(False), "sid-1")
    session_io._on_bg_done(_done("sid-1"), "sid-1")
    sm._apply_rotated_session()
    assert prod_mode == []


def test_maps_are_bounded(prod_mode):
    cap = session_io._BG_MAP_CAP
    for i in range(cap + 50):
        session_io._on_bg_done(_done(False), f"f{i}")
        session_io._on_bg_done(_done(f"new{i}"), f"r{i}")
    assert len(session_io._failed_saves) == cap
    assert len(session_io._rotated_sessions) == cap
    assert "f0" not in session_io._failed_saves and f"f{cap + 49}" in session_io._failed_saves
