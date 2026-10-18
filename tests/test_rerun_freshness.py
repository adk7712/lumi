"""Mutations made by buttons must be reflected in the SAME run (no extra at.run())."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
CSV = (
    b"Id,Age,Sex,LotArea,Alley\n"
    b"1,25,male,500,Pave\n2,30,female,600,Grvl\n3,35,male,700,\n"
    b"4,40,female,650,Pave\n5,29,male,5000,Grvl\n5,29,male,5000,Grvl\n"
)


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_quick_action_refreshes_overview_in_same_run():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("s.csv", CSV, "text/csv").run()
    assert not at.exception
    assert _metric(at, "Rows") == "6" and _metric(at, "Duplicates") == "1"
    assert any("Recipe Steps:** 0" in m.value for m in at.markdown)

    next(b for b in at.button if b.key == "qa_drop_duplicates").click().run()

    assert not at.exception
    assert _metric(at, "Rows") == "5"
    assert _metric(at, "Duplicates") == "0"
    assert any("Recipe Steps:** 1" in m.value for m in at.markdown)
    assert not any(b.key == "qa_drop_duplicates" for b in at.button)
    assert not at.session_state["_needs_rerun"] if "_needs_rerun" in at.session_state else True
