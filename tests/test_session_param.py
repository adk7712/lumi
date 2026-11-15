"""The ?session= URL parameter is only honoured when it carries a valid signature."""
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


def _run_with_param(monkeypatch, value):
    looked_up = []
    monkeypatch.setattr(persistence, "load_session", lambda sid, *a, **k: looked_up.append(sid))
    # The fallback is for anonymous visitors only (dev auth would otherwise count as signed in).
    monkeypatch.setattr(ui_utils, "is_authenticated_user", lambda: False)
    monkeypatch.setattr(landing, "is_authenticated_user", lambda: False)  # bound at import
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["session"] = value
    at.run()
    return at, looked_up


def test_unsigned_session_param_is_ignored(monkeypatch):
    at, looked_up = _run_with_param(monkeypatch, "victim-session-id")
    assert not at.exception
    assert looked_up == []


def test_signed_session_param_is_used(monkeypatch):
    at, looked_up = _run_with_param(monkeypatch, ui_utils.sign_cookie_value("sid-ok", purpose="session"))
    assert not at.exception
    assert looked_up == ["sid-ok"]
