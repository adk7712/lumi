"""W1 DB hardening: guest-id prefix, JSON encoding, atomic cache, workspace cap, perf log, cache ids."""
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LUMI_TESTING", "1")

import numpy as np
import pandas as pd
import pytest
import streamlit as st

import persistence
from lumi_db import pool as pool_mod, schema as schema_mod, sessions as sessions_mod
from lumi_state import session_io

DB = Path("test_lumi_w1.db")
GUEST = "guest_" + "a" * 32 + "@lumi.ai"


def _rm():
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            Path(str(DB) + suffix).unlink()
        except FileNotFoundError:
            pass


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    monkeypatch.setattr(persistence, "DB_PATH", DB)
    _rm()
    persistence._DB_INITIALIZED = False
    schema_mod._reset_migration_backoff()
    persistence.init_db()
    yield
    _rm()


def _save(sid, user, **kw):
    return persistence.save_session(sid, "f.csv", kw.get("recipe", []), [], set(), user_id=user)


def test_real_account_with_guest_prefix_is_not_a_guest():
    user = "guest_jane@gmail.com"
    assert _save("s1", user) is True
    assert [p["session_id"] for p in persistence.get_user_projects(user)] == ["s1"]
    assert persistence.rename_session("s1", "New", user) is True
    assert persistence.toggle_pin_session("s1", user) is True
    assert persistence.count_user_sessions(user) == 1
    assert persistence.delete_session("s1", user) is True


def test_real_guest_id_still_locked_out():
    assert persistence.get_user_projects(GUEST) == []
    assert persistence.rename_session("s1", "x", GUEST) is False
    assert persistence.count_user_sessions(GUEST) == 0


def test_save_session_accepts_numpy_datetime_set_values():
    recipe = [{"n": np.int64(3), "b": np.bool_(True), "f": np.float64("nan"), "ts": pd.Timestamp("2024-01-02"),
               "dt": datetime(2024, 1, 2, 3), "s": {"b", "a"}, "o": object}]
    assert _save("s1", "u@x.com", recipe=recipe) is True
    got = persistence.load_session("s1", "u@x.com")["cleaning_recipe"][0]
    assert got["n"] == 3 and got["b"] is True
    assert got["ts"].startswith("2024-01-02") and got["s"] == ["a", "b"]
    assert isinstance(got["o"], str)


def test_json_default_numpy_scalars():
    assert json.dumps({"a": np.int64(1), "b": np.float32(1.5)}, default=sessions_mod.json_default) == '{"a": 1, "b": 1.5}'


def test_workspace_limit_enforced_in_db_layer():
    user = "limit@x.com"
    for i in range(5):
        assert _save(f"s{i}", user) is True
    assert _save("s5", user) is False
    assert sessions_mod.last_save_refusal() == "limit"
    assert persistence.count_user_sessions(user) == 5
    assert _save("s0", user, recipe=[{"a": 1}]) is True  # updating an existing row is allowed


def test_limit_refusal_does_not_rotate_session_id(monkeypatch):
    user = "limit2@x.com"
    for i in range(5):
        _save(f"s{i}", user)
    st.session_state.clear()
    st.session_state.session_id = "new-session"
    st.session_state.filename = "f.csv"
    monkeypatch.setattr("ui_utils.get_logged_in_user", lambda: user)
    calls = []
    real = persistence.save_session

    def spy(*a, **k):
        calls.append(k.get("session_id"))
        return real(*a, **k)

    monkeypatch.setattr(persistence, "save_session", spy)
    session_io.save_db_session()
    assert calls == ["new-session"]  # no fresh-uuid retry
    assert st.session_state.session_id == "new-session"


def test_rename_rejects_long_names():
    user = "ren@x.com"
    _save("s1", user)
    assert persistence.rename_session("s1", "x" * 101, user) is False
    assert persistence.rename_session("s1", "  " + "x" * 100 + "  ", user) is True


def test_resume_cache_write_is_atomic_and_encodes(monkeypatch, tmp_path):
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path)
    st.session_state.clear()
    st.session_state.last_file_hash = "h"
    monkeypatch.setattr(session_io, "resume_owner", lambda: "owner")
    st.session_state.cleaning_recipe = [{"n": np.int64(2), "ts": pd.Timestamp("2024-01-01")}]
    session_io.save_session_state()
    path = session_io.resume_cache_path("h", "owner")
    assert json.loads(path.read_text())["cleaning_recipe"][0]["n"] == 2
    good = path.read_text()

    class Bad:  # json default() returns something unserializable -> dumps raises
        pass

    monkeypatch.setattr("lumi_db.sessions.json_default", lambda o: (_ for _ in ()).throw(ValueError("boom")))
    st.session_state.cleaning_recipe = [{"x": Bad()}]
    session_io.save_session_state()
    assert path.read_text() == good
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp leftovers


def test_perf_log_hashes_ids_and_rotates(monkeypatch, tmp_path):
    log = tmp_path / "db_perf_log.jsonl"
    monkeypatch.setattr(pool_mod, "DB_PERF_LOG", log)
    pool_mod._log_db_perf("op", 1.0, {"user_id": "me@x.com", "session_id": "sid-123", "n": 1})
    rec = json.loads(log.read_text())
    assert "me@x.com" not in log.read_text() and "sid-123" not in log.read_text()
    assert len(rec["details"]["user_id"]) == 12 and rec["details"]["n"] == 1
    log.write_text("x" * (pool_mod._PERF_LOG_MAX_BYTES + 1))
    (tmp_path / "db_perf_log.jsonl.1").write_text("old")
    pool_mod._log_db_perf("op", 1.0, {})
    assert (tmp_path / "db_perf_log.jsonl.1").read_text().startswith("xxx")
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "a" * 129, "x.dat\0", None])
def test_cache_helpers_refuse_unsafe_ids(monkeypatch, tmp_path, bad):
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "evil.dat").write_bytes(b"x")
    assert session_io.load_cached_file(bad, "f.csv") is None
    session_io.delete_cached_file(bad)
    assert (tmp_path / "evil.dat").exists()
