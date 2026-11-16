"""W4: dev-auth fail-closed (A24) and timestamped signed cookies (I15)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

import ui_utils
from ui_utils import is_dev_auth_enabled, sign_cookie_value, verify_cookie_value

DAY = 24 * 60 * 60


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("LUMI_DEV_AUTH", "LUMI_TESTING", "LUMI_ENV"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ui_utils.st, "secrets", {}, raising=False)
    monkeypatch.setenv("LUMI_COOKIE_SECRET", "w4-secret")
    monkeypatch.setattr(ui_utils, "_DEV_AUTH_PROD_WARNED", False)


def test_dev_auth_off_when_env_unset(monkeypatch):
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")
    assert is_dev_auth_enabled() is False


def test_dev_auth_on_with_explicit_development(monkeypatch):
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")
    monkeypatch.setenv("LUMI_ENV", "development")
    assert is_dev_auth_enabled() is True


def test_dev_auth_secret_needs_explicit_development(monkeypatch):
    monkeypatch.setattr(ui_utils.st, "secrets", {"dev_auth": True}, raising=False)
    assert is_dev_auth_enabled() is False
    monkeypatch.setattr(ui_utils.st, "secrets", {"dev_auth": True, "env": "development"}, raising=False)
    assert is_dev_auth_enabled() is True


def test_dev_auth_on_with_testing(monkeypatch):
    monkeypatch.setenv("LUMI_TESTING", "1")
    assert is_dev_auth_enabled() is True


def test_dev_auth_off_in_production(monkeypatch):
    monkeypatch.setenv("LUMI_ENV", "production")
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")
    monkeypatch.setenv("LUMI_TESTING", "1")
    assert is_dev_auth_enabled() is False


def test_cookie_roundtrip_with_dots():
    signed = sign_cookie_value("a.b@c.com", "user_email")
    assert verify_cookie_value(signed, "user_email") == "a.b@c.com"


def test_cookie_expired_rejected(monkeypatch):
    signed = sign_cookie_value("sid", "session")
    now = ui_utils.time.time()
    monkeypatch.setattr(ui_utils.time, "time", lambda: now + 30 * DAY + 10)
    assert verify_cookie_value(signed, "session") is None
    monkeypatch.setattr(ui_utils.time, "time", lambda: now + 29 * DAY)
    assert verify_cookie_value(signed, "session") == "sid"


def test_cookie_future_timestamp_rejected(monkeypatch):
    now = ui_utils.time.time()
    monkeypatch.setattr(ui_utils.time, "time", lambda: now + 3600)
    signed = sign_cookie_value("sid", "session")
    monkeypatch.setattr(ui_utils.time, "time", lambda: now)
    assert verify_cookie_value(signed, "session") is None
    # within the 5 minute skew is accepted
    monkeypatch.setattr(ui_utils.time, "time", lambda: now + 3600 - 60)
    assert verify_cookie_value(signed, "session") == "sid"


def test_cookie_old_format_rejected():
    import hashlib
    import hmac
    sig = hmac.new(b"w4-secret", b"lumi-cookie-v1|session|sid", hashlib.sha256).hexdigest()
    assert verify_cookie_value(f"sid.{sig}", "session") is None


def test_cookie_tampered_rejected():
    signed = sign_cookie_value("sid", "session")
    value, ts, sig = signed.rsplit(".", 2)
    assert verify_cookie_value(f"other.{ts}.{sig}", "session") is None
    assert verify_cookie_value(f"{value}.{int(ts) + 1}.{sig}", "session") is None
    assert verify_cookie_value(f"{value}.{ts}.{sig[:-1]}0" if sig[-1] != "0" else f"{value}.{ts}.{sig[:-1]}1", "session") is None
    assert verify_cookie_value(signed, "user_email") is None


def test_cookie_secure_except_localhost(monkeypatch):
    import types
    import ui_utils
    for host, want in [("lumiapp.streamlit.app", True), ("localhost:8501", False),
                       ("127.0.0.1:8501", False), ("[::1]:8501", False), ("", True)]:
        monkeypatch.setattr(ui_utils.st, "context", types.SimpleNamespace(headers={"Host": host}))
        assert ui_utils._cookie_secure() is want, host
