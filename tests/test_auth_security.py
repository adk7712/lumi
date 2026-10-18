"""Security tests for S1 (dev-auth gate) and S2 (signed cookies)."""
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import streamlit as st

import ui_utils
from ui_utils import (
    get_logged_in_user,
    is_dev_auth_enabled,
    read_session_cookie,
    sign_cookie_value,
    verify_cookie_value,
)


class _EmptyUser(dict):
    pass


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Reset session state/user and give each test a deterministic cookie secret."""
    for key in ("dev_user_email", "_is_testing"):
        st.session_state.pop(key, None)
    monkeypatch.setattr(st, "user", _EmptyUser(), raising=False)
    monkeypatch.setenv("LUMI_COOKIE_SECRET", "unit-test-secret")
    yield
    for key in ("dev_user_email", "_is_testing"):
        st.session_state.pop(key, None)


@pytest.fixture
def dev_auth_off(monkeypatch):
    monkeypatch.delenv("LUMI_DEV_AUTH", raising=False)
    monkeypatch.delenv("LUMI_TESTING", raising=False)
    monkeypatch.setattr(ui_utils.st, "secrets", {}, raising=False)


@pytest.fixture
def dev_auth_on(monkeypatch):
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")


def _cookie_controller(values):
    controller = mock.MagicMock()
    controller.get.side_effect = lambda key: values.get(key)
    return controller


# --- S1: dev login gate -----------------------------------------------------

def test_dev_auth_disabled_without_flag(dev_auth_off):
    assert is_dev_auth_enabled() is False


def test_dev_auth_enabled_by_env_or_secret(monkeypatch, dev_auth_off):
    monkeypatch.setenv("LUMI_DEV_AUTH", "1")
    assert is_dev_auth_enabled() is True
    monkeypatch.delenv("LUMI_DEV_AUTH")
    monkeypatch.setattr(ui_utils.st, "secrets", {"dev_auth": True}, raising=False)
    assert is_dev_auth_enabled() is True


def test_dev_email_in_session_ignored_without_flag(dev_auth_off):
    st.session_state.dev_user_email = "victim@company.com"
    assert get_logged_in_user() is None
    assert "dev_user_email" not in st.session_state


def test_guest_identity_still_works_without_flag(dev_auth_off):
    st.session_state.dev_user_email = "guest_abc@lumi.ai"
    assert get_logged_in_user() == "guest_abc@lumi.ai"


def test_dev_login_form_not_rendered_without_flag(dev_auth_off):
    with mock.patch.object(ui_utils.st, "text_input") as text_input, \
         mock.patch.object(ui_utils.st, "button", return_value=False) as button, \
         mock.patch.object(ui_utils.st, "markdown"), \
         mock.patch.object(ui_utils.st, "info"):
        ui_utils._render_auth_options()
        text_input.assert_not_called()
        keys = [c.kwargs.get("key") for c in button.call_args_list]
        assert "auth_dev_email_login_btn" not in keys
        assert "auth_dev_login_btn" in keys  # guest path remains available


def test_st_user_wins_over_dev_email(dev_auth_on, monkeypatch):
    monkeypatch.setattr(st, "user", _EmptyUser({"email": "real@oidc.com"}), raising=False)
    st.session_state.dev_user_email = "dev@example.com"
    assert get_logged_in_user() == "real@oidc.com"


# --- S2: signed cookies -----------------------------------------------------

def test_valid_signature_roundtrip():
    signed = sign_cookie_value("a@b.com", purpose="user_email")
    assert verify_cookie_value(signed, purpose="user_email") == "a@b.com"


def test_tampered_signature_rejected():
    signed = sign_cookie_value("a@b.com", purpose="user_email")
    value, sig = signed.rsplit(".", 1)
    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert verify_cookie_value(f"{value}.{flipped}", purpose="user_email") is None
    # Swapping the email while keeping the old signature must fail too
    assert verify_cookie_value(f"evil@b.com.{sig}", purpose="user_email") is None
    # Signature for one purpose cannot be reused for another
    assert verify_cookie_value(signed, purpose="session") is None


def test_unsigned_values_rejected():
    for raw in (None, "", "a@b.com", "4f1c2a6e-0000-4000-8000-000000000000", 123):
        assert verify_cookie_value(raw, purpose="user_email") is None


def test_forged_email_cookie_ignored_without_flag(dev_auth_off):
    controller = _cookie_controller({"lumi_user_email": sign_cookie_value("a@b.com", "user_email")})
    with mock.patch("streamlit_cookies_controller.CookieController", return_value=controller):
        assert get_logged_in_user() is None
    controller.get.assert_not_called()


def test_unsigned_email_cookie_ignored_with_flag(dev_auth_on):
    controller = _cookie_controller({"lumi_user_email": "victim@company.com"})
    with mock.patch("streamlit_cookies_controller.CookieController", return_value=controller):
        assert get_logged_in_user() is None
    assert "dev_user_email" not in st.session_state


def test_signed_email_cookie_accepted_with_flag(dev_auth_on):
    controller = _cookie_controller({"lumi_user_email": sign_cookie_value("dev@example.com", "user_email")})
    with mock.patch("streamlit_cookies_controller.CookieController", return_value=controller):
        assert get_logged_in_user() == "dev@example.com"


def test_session_cookie_verification():
    good = _cookie_controller({"lumi_session": sign_cookie_value("sid-123", "session")})
    assert read_session_cookie(good) == "sid-123"
    legacy = _cookie_controller({"lumi_session": "sid-123"})
    assert read_session_cookie(legacy) is None
    forged = _cookie_controller({"lumi_session": "other-sid." + sign_cookie_value("sid-123", "session").rsplit(".", 1)[1]})
    assert read_session_cookie(forged) is None
    broken = mock.MagicMock()
    broken.get.side_effect = RuntimeError("no runtime")
    assert read_session_cookie(broken) is None


def test_session_ids_are_uuid4(monkeypatch):
    import uuid
    import inspect
    import state_manager
    src = inspect.getsource(state_manager.process_uploaded_file)
    assert "uuid.uuid4()" in src
    assert uuid.UUID(str(uuid.uuid4())).version == 4
