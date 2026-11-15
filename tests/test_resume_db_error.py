"""A DB error while resuming must keep the resume cookie (it is not an expired session)."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

import persistence
import ui_utils
from views import landing

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _raise(*a, **k):
    raise persistence.SessionLoadError("pool timeout")


def test_resume_db_error_keeps_cookie_and_does_not_cache(monkeypatch):
    removed = []
    monkeypatch.setattr(persistence, "load_session", _raise)
    monkeypatch.setattr(landing, "remove_cookie", lambda name: removed.append(name))
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["_is_testing"] = True
    at.session_state["resume_session_id"] = "sid-down"
    at.run()
    assert not at.exception
    assert removed == []
    assert at.session_state["resume_session_id"] == "sid-down"
    assert "_cached_resume_sid-down" not in at.session_state
    assert any("Could not reach the database" in e.value for e in at.error)


def test_resume_missing_session_still_clears(monkeypatch):
    removed = []
    monkeypatch.setattr(persistence, "load_session", lambda *a, **k: None)
    monkeypatch.setattr(landing, "remove_cookie", lambda name: removed.append(name))
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["_is_testing"] = True
    at.session_state["resume_session_id"] = "sid-gone"
    at.run()
    assert removed == [ui_utils.SESSION_COOKIE]
    assert "resume_session_id" not in at.session_state
