"""Background DB saves for one session run serially and keep only the newest snapshot."""
import threading
import time

import pytest
import streamlit as st

import state_manager as sm
from lumi_state import session_io


@pytest.fixture(autouse=True)
def _clean():
    session_io._pending_saves.clear()
    session_io._running_saves.clear()
    session_io._rotated_sessions.clear()
    session_io._failed_saves.clear()
    yield
    session_io._pending_saves.clear()
    session_io._running_saves.clear()
    session_io._rotated_sessions.clear()
    session_io._failed_saves.clear()


def _wait_idle(sid, timeout=5):
    with session_io._save_cond:
        assert session_io._save_cond.wait_for(lambda: sid not in session_io._running_saves, timeout=timeout)


def test_saves_for_one_session_never_overlap_and_newest_wins():
    active, peak, written, lock = [0], [0], [], threading.Lock()

    def job_for(n):
        def job(target):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                written.append(n)
                active[0] -= 1
            return target
        return job

    for n in range(10):
        session_io._submit_save("sid-x", job_for(n))
    _wait_idle("sid-x")
    assert peak[0] == 1
    assert written[-1] == 9
    assert written == sorted(written)
    assert len(written) < 10  # unstarted snapshots were replaced by newer ones


def test_cancel_pending_saves_drops_queued_and_waits_for_running():
    started, release, written = threading.Event(), threading.Event(), []

    def slow(target):
        started.set()
        release.wait(2)
        written.append("slow")
        return target

    session_io._submit_save("sid-d", slow)
    started.wait(2)
    session_io._submit_save("sid-d", lambda t: written.append("queued") or t)
    threading.Timer(0.05, release.set).start()
    assert session_io.cancel_pending_saves("sid-d", timeout=2)
    assert written == ["slow"]


def test_save_db_session_end_to_end_keeps_latest_recipe(monkeypatch):
    import persistence
    db = {}

    def fake_save(session_id, recipe, **kw):
        time.sleep(0.01 * (5 - len(recipe)) if len(recipe) < 5 else 0)  # older snapshots are slower
        db[session_id] = len(recipe)
        return True

    monkeypatch.delenv("LUMI_TESTING", raising=False)
    monkeypatch.setattr(persistence, "save_session", fake_save)
    st.session_state.clear()
    st.session_state["session_id"] = "sid-e"
    st.session_state["_is_testing"] = False
    st.session_state["cleaning_recipe"] = []
    for i in range(6):
        st.session_state["cleaning_recipe"].append({"action": "drop_duplicates"})
        sm.save_db_session()
    _wait_idle("sid-e")
    assert db["sid-e"] == 6
    st.session_state.clear()
