"""After st.login (fresh session state), the guest workspace is claimed via the signed cookie."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

import persistence
import ui_utils

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _app(monkeypatch, owner, reconcile_ok=True):
    claimed = []
    monkeypatch.setattr(ui_utils, "get_logged_in_user", lambda: "a@example.com")
    monkeypatch.setattr(ui_utils, "read_session_cookie", lambda *a, **k: "guest-sid")
    monkeypatch.setattr(persistence, "load_session", lambda sid, user=None: {"session_id": sid, "user_id": owner})
    monkeypatch.setattr(persistence, "reconcile_session", lambda sid, user: claimed.append((sid, user)) or reconcile_ok)
    monkeypatch.setattr(persistence, "get_user_projects", lambda *a, **k: [])
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at, claimed


def test_guest_workspace_claimed_after_login(monkeypatch):
    at, claimed = _app(monkeypatch, owner="guest_123")
    assert not at.exception
    assert claimed == [("guest-sid", "a@example.com")]
    assert at.session_state["_reconciled_session"] == "guest-sid"
    at.run()
    assert claimed == [("guest-sid", "a@example.com")]  # only once


def test_own_workspace_not_reclaimed(monkeypatch):
    at, claimed = _app(monkeypatch, owner="a@example.com")
    assert not at.exception
    assert claimed == []
