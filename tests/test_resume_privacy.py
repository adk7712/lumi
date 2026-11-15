"""V1 (owner-scoped local resume cache) and V2 (single CookieController per run, signed cookie).

The local resume cache must only be offered to the same owner: the signed session id for
guests, the account id for signed-in users. Legacy unscoped ``{file_hash}.json`` files must
never be offered to anyone.
"""
import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))
os.environ["LUMI_TESTING"] = "1"

import pandas as pd
import pytest
import streamlit as st

import state_manager as sm
import ui_utils
from lumi_state import session_io
from ui_utils import SESSION_COOKIE, sign_cookie_value, verify_cookie_value

CSV = b"col1,col2\n1, abc \n2, def \n"
_STATE_KEYS = (
    "session_id", "last_file_hash", "dev_user_email", "_is_testing",
    ui_utils._COOKIE_CTRL_STATE_KEY, ui_utils._COOKIE_OPS_STATE_KEY,
)


class _Upload(io.BytesIO):
    def __init__(self, name="data.csv", content=CSV):
        super().__init__(content)
        self.name = name
        self.size = len(content)
        self.type = "text/csv"


def _controller(values=None):
    ctrl = mock.MagicMock()
    store = dict(values or {})
    ctrl.get.side_effect = lambda k: store.get(k)
    ctrl.getAll.side_effect = lambda: store
    return ctrl


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    for k in _STATE_KEYS:
        st.session_state.pop(k, None)
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("LUMI_COOKIE_SECRET", "test-secret-for-resume-privacy")
    monkeypatch.setattr(ui_utils.st, "secrets", {}, raising=False)
    yield
    for k in _STATE_KEYS:
        st.session_state.pop(k, None)


def _as_guest(monkeypatch, cookie_sid=None, raw_cookie=None):
    """Anonymous visitor whose browser carries ``cookie_sid`` in a signed lumi_session cookie."""
    monkeypatch.setattr(ui_utils, "is_authenticated_user", lambda: False)
    raw = raw_cookie if raw_cookie is not None else (
        sign_cookie_value(cookie_sid, purpose="session") if cookie_sid else None)
    monkeypatch.setattr(ui_utils, "get_cookie_controller", lambda: _controller({SESSION_COOKIE: raw}))
    st.session_state["session_id"] = None


def _as_user(monkeypatch, email):
    monkeypatch.setattr(ui_utils, "is_authenticated_user", lambda: True)
    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: email)


def _save_progress(file_hash, recipe=None):
    st.session_state["last_file_hash"] = file_hash
    st.session_state["cleaning_recipe"] = recipe or [{"action": "strip_whitespace", "column": "col2"}]
    st.session_state["rules"] = [{"type": "Null Check", "col": "col1"}]
    st.session_state["scanned_columns"] = set()
    sm.save_session_state()


# --- V1: owner scoping -------------------------------------------------------

def test_different_guest_same_file_gets_no_restore_offer(monkeypatch):
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress("hash-1")

    _as_guest(monkeypatch, cookie_sid="sid-bob")
    assert sm.has_resume_cache("hash-1") is False


def test_same_guest_after_reload_gets_restore_offer(monkeypatch):
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress("hash-1")

    # Tab closed and reopened: no session state, session id recovered from the signed cookie.
    _as_guest(monkeypatch, cookie_sid="sid-alice")
    assert sm.has_resume_cache("hash-1") is True
    assert sm.has_resume_cache("other-file-hash") is False


def test_forged_or_unsigned_cookie_grants_no_restore(monkeypatch):
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress("hash-1")

    _as_guest(monkeypatch, raw_cookie="sid-alice")  # legacy unsigned value
    assert sm.has_resume_cache("hash-1") is False
    sig = sign_cookie_value("sid-x", purpose="session").rsplit(".", 1)[1]
    _as_guest(monkeypatch, raw_cookie=f"sid-alice.{sig}")  # signature for another id
    assert sm.has_resume_cache("hash-1") is False


def test_signed_in_users_are_scoped_by_account(monkeypatch):
    _as_user(monkeypatch, "a@example.com")
    _save_progress("hash-1")
    assert sm.has_resume_cache("hash-1") is True

    _as_user(monkeypatch, "b@example.com")
    assert sm.has_resume_cache("hash-1") is False


def test_no_owner_means_nothing_written(monkeypatch):
    _as_guest(monkeypatch)  # no session id, no cookie
    _save_progress("hash-1")
    cache_dir = sm.CACHE_DIR
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_cache_file_name_does_not_reveal_file_hash(monkeypatch):
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress("hash-1")
    names = [p.name for p in sm.CACHE_DIR.iterdir()]
    assert names == [sm.resume_cache_path("hash-1", "guest:sid-alice").name]
    assert "hash-1" not in names[0]


def test_legacy_unscoped_cache_is_ignored_and_deleted(monkeypatch):
    sm.CACHE_DIR.mkdir(parents=True)
    legacy = sm.CACHE_DIR / "hash-1.json"
    legacy.write_text(json.dumps({"cleaning_recipe": [{"action": "drop_nulls"}], "rules": [], "scanned_columns": []}))

    _as_guest(monkeypatch, cookie_sid="sid-anyone")
    assert sm.has_resume_cache("hash-1") is False
    assert not legacy.exists()


def test_restore_loads_only_own_progress_and_keeps_guest_session_id(monkeypatch):
    monkeypatch.setattr(session_io, "_set_session_cookie", lambda sid: None)
    monkeypatch.setattr(session_io, "save_db_session", lambda: None)
    monkeypatch.setattr(session_io, "cache_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(session_io, "load_data", lambda buf, nrows=None: pd.read_csv(buf, nrows=nrows))
    sm.initialize_state(from_reset=True)

    file_hash = sm.calculate_file_hash(_Upload())
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress(file_hash)

    # Bob (other browser) restoring the same file gets a clean workspace.
    sm.initialize_state(from_reset=True)
    _as_guest(monkeypatch, cookie_sid="sid-bob")
    sm.load_session_state(file_hash, _Upload())
    assert st.session_state["cleaning_recipe"] == []
    assert st.session_state["rules"] == []

    # Alice after a reload gets her steps back, under the same session id.
    sm.initialize_state(from_reset=True)
    _as_guest(monkeypatch, cookie_sid="sid-alice")
    sm.load_session_state(file_hash, _Upload())
    assert st.session_state["cleaning_recipe"] == [{"action": "strip_whitespace", "column": "col2"}]
    assert st.session_state["session_id"] == "sid-alice"


def test_discard_removes_only_own_cache(monkeypatch):
    _as_guest(monkeypatch)
    st.session_state["session_id"] = "sid-alice"
    _save_progress("hash-1")
    st.session_state["session_id"] = "sid-bob"
    _save_progress("hash-1")

    sm.discard_resume_cache("hash-1", "guest:sid-bob")
    assert sm.has_resume_cache("hash-1", "guest:sid-bob") is False
    assert sm.has_resume_cache("hash-1", "guest:sid-alice") is True


# --- V2: one CookieController per run, signed cookie written -----------------

def test_single_cookie_controller_per_run(monkeypatch):
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")
    st.session_state["_is_testing"] = False
    run_marker = {"m": set()}
    monkeypatch.setattr(ui_utils, "_current_run_marker", lambda: run_marker["m"])

    with mock.patch("streamlit_cookies_controller.CookieController") as ctor:
        ctor.return_value = _controller()
        # Everything that touches cookies in one run: app top, auth, landing read, upload write.
        ui_utils.sync_cookies()
        ui_utils.get_logged_in_user()
        ui_utils.read_session_cookie()
        session_io._set_session_cookie("sid-1")
        ui_utils.remove_cookie(ui_utils.USER_EMAIL_COOKIE)
        assert ctor.call_count == 1

        run_marker["m"] = set()  # next script run
        ui_utils.read_session_cookie()
        ui_utils.get_logged_in_user()
        assert ctor.call_count == 2


def test_signed_session_cookie_written_on_upload(monkeypatch):
    monkeypatch.setattr(session_io, "save_db_session", lambda: None)
    monkeypatch.setattr(session_io, "cache_uploaded_file", lambda *a, **k: None)
    monkeypatch.setattr(session_io, "load_data", lambda buf, nrows=None: pd.read_csv(buf, nrows=nrows))
    run_marker = {"m": set()}
    monkeypatch.setattr(ui_utils, "_current_run_marker", lambda: run_marker["m"])
    ctrl = _controller()
    sm.initialize_state(from_reset=True)

    with mock.patch("streamlit_cookies_controller.CookieController", return_value=ctrl) as ctor:
        sm.process_uploaded_file(_Upload(), "hash-1")
        sid = st.session_state["session_id"]
        signed = sign_cookie_value(sid, purpose="session")
        ctrl.set.assert_called_once_with(SESSION_COOKIE, signed, max_age=ui_utils.COOKIE_MAX_AGE_SECONDS)
        assert verify_cookie_value(signed, purpose="session") == sid
        assert signed != sid  # never written unsigned

        # The upload is followed by st.rerun(); the write is re-asserted on the next runs.
        for _ in range(ui_utils._COOKIE_REASSERT_RUNS):
            run_marker["m"] = set()
            ui_utils.sync_cookies()
        assert ctrl.set.call_count == 1 + ui_utils._COOKIE_REASSERT_RUNS
        assert all(c.args == (SESSION_COOKIE, signed) for c in ctrl.set.call_args_list)
        run_marker["m"] = set()
        ui_utils.sync_cookies()
        assert ctrl.set.call_count == 1 + ui_utils._COOKIE_REASSERT_RUNS  # queue drained
        assert ctor.call_count == 1 + ui_utils._COOKIE_REASSERT_RUNS + 1  # one per run


def test_remove_cookie_tolerates_missing_local_copy(monkeypatch):
    ctrl = _controller()
    ctrl.remove.side_effect = KeyError(SESSION_COOKIE)
    monkeypatch.setattr(ui_utils, "get_cookie_controller", lambda: ctrl)
    assert ui_utils.remove_cookie(SESSION_COOKIE) is True
    assert st.session_state[ui_utils._COOKIE_OPS_STATE_KEY][SESSION_COOKIE][0] is None
