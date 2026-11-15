"""Sampling metadata + Overview notice for large uploads."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _csv(n):
    return ("Id,Age\n" + "\n".join(f"{i},{20 + i % 50}" for i in range(n)) + "\n").encode()


def _upload(n):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("big.csv", _csv(n), "text/csv").run()
    assert not at.exception
    return at


def _infos(at):
    return [i.value for i in at.info]


def test_random_sample_notice():
    at = _upload(15000)
    assert at.session_state.sample_info == {"shown": 10000, "total": 15000, "method": "random"}
    assert any("Showing 10,000 of 15,000 rows (random sample)" in v for v in _infos(at))


def test_no_notice_when_small():
    at = _upload(500)
    assert at.session_state.sample_info is None
    assert not any("Showing" in v and "rows" in v for v in _infos(at))


def test_head_load_notice(monkeypatch):
    import lumi_state.session_io as sio
    monkeypatch.setattr(sio, "LARGE_FILE_THRESHOLD_BYTES", 10)
    at = _upload(15000)
    info = at.session_state.sample_info
    assert info["method"] == "head" and info["total"] is None and info["shown"] == 10000
    assert any("first 10,000 rows" in v for v in _infos(at))
