"""Regression tests: upload limits, load_data cache bounds, workspace name, sign-out, PostHog escaping."""
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

from lumi_state import core
from lumi_state.core import UnsafeFileError, check_xlsx_safe, read_tabular

ROOT = Path(__file__).resolve().parent.parent


def _zip(entries, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_upload_limit_configured():
    text = (ROOT / ".streamlit" / "config.toml").read_text()
    assert "maxUploadSize" in text
    line = next(l for l in text.splitlines() if l.strip().startswith("maxUploadSize"))
    assert int(line.split("=")[1].split("#")[0]) <= 100


def test_xlsx_over_uncompressed_cap_rejected(monkeypatch):
    monkeypatch.setattr(core, "MAX_XLSX_UNCOMPRESSED_BYTES", 1000)
    with pytest.raises(UnsafeFileError):
        check_xlsx_safe(_zip({"xl/sheet1.xml": b"a" * 5000}))


def test_xlsx_extreme_ratio_rejected(monkeypatch):
    monkeypatch.setattr(core, "_XLSX_RATIO_MIN_BYTES", 1000)
    with pytest.raises(UnsafeFileError):
        check_xlsx_safe(_zip({"xl/sheet1.xml": b"0" * 200_000}))


def test_read_tabular_rejects_bomb_before_parsing(monkeypatch):
    monkeypatch.setattr(core, "MAX_XLSX_UNCOMPRESSED_BYTES", 1000)
    with patch.object(pd, "read_excel") as rx:
        with pytest.raises(UnsafeFileError):
            read_tabular(_zip({"x": b"a" * 5000}), "bomb.xlsx")
        rx.assert_not_called()


def test_load_data_surfaces_bomb_as_error(monkeypatch):
    monkeypatch.setattr(core, "MAX_XLSX_UNCOMPRESSED_BYTES", 1000)
    raw = _zip({"x": b"a" * 5000})
    out = core.load_data.__wrapped__(core.CachedFileWrapper(raw, "bomb.xlsx"))
    assert out.empty
    # Reported via attrs; process_uploaded_file keeps it on the landing page (a cached st.error only flashed).
    assert "Could not open this file" in out.attrs["lumi_load_error"]


def test_normal_xlsx_still_loads(tmp_path):
    p = tmp_path / "ok.xlsx"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(p, index=False)
    df, _ = read_tabular(str(p), "ok.xlsx")
    assert list(df.columns) == ["a", "b"] and len(df) == 2
    df2, _ = read_tabular(p.read_bytes(), "ok.xlsx")
    assert len(df2) == 2


def test_load_data_cache_is_bounded():
    src = (ROOT / "lumi_state" / "core.py").read_text()
    decorator = src[src.index("@st.cache_data(hash_funcs={"):src.index("def load_data")]
    assert "max_entries=" in decorator and "ttl=" in decorator


def test_missingness_map_downsamples_before_masking(monkeypatch):
    import ui_utils
    monkeypatch.setattr(ui_utils, "PLOT_SAMPLE_SIZE", 50, raising=False)
    n = 400
    df = pd.DataFrame({"a": [None if i % 7 == 0 else i for i in range(n)], "b": range(n)})
    seen = {}
    real = ui_utils.downsample_for_plot

    def spy(obj, *a, **k):
        seen["rows"] = len(obj)
        seen["dtype_bool"] = all(str(t) == "bool" for t in getattr(obj, "dtypes", []))
        return real(obj, 50)
    monkeypatch.setattr(ui_utils, "downsample_for_plot", spy)
    fig, sampled = ui_utils.plot_missingness_map.__wrapped__(df)
    assert sampled is True and fig is not None
    assert seen["rows"] == n and not seen["dtype_bool"]  # sampler got the raw frame, not a full mask
    assert len(fig.data[0].z) == 50 and len(fig.data[0].z[0]) == 2
    # no-null frame -> no figure
    assert ui_utils.plot_missingness_map.__wrapped__(pd.DataFrame({"a": [1, 2]})) == (None, False)


def test_new_workspace_name_survives_reset():
    from state_manager import initialize_state
    st.session_state.clear()
    initialize_state()
    st.session_state._pending_workspace_name = "My Workspace"
    initialize_state(from_reset=True)
    assert st.session_state._pending_workspace_name == "My Workspace"
    st.session_state.clear()


def test_signout_clears_cookie_and_user_state():
    import ui_utils
    st.session_state.clear()
    st.session_state["_cache_user_projects_a@x.com"] = [1]
    st.session_state["_cached_resume_sid"] = {"x": 1}
    st.session_state["_cached_cookie_sid"] = 1
    st.session_state["_reconciled_session"] = "sid"
    st.session_state["resume_session_id"] = "sid"
    st.session_state["session_id"] = "sid"
    with patch.object(ui_utils, "remove_cookie") as rc, patch.object(ui_utils.st, "logout"), \
            patch.object(ui_utils.st, "rerun"):
        ui_utils.handle_signout()
    removed = [c.args[0] for c in rc.call_args_list]
    assert ui_utils.SESSION_COOKIE in removed
    for k in ("_cache_user_projects_a@x.com", "_cached_resume_sid", "_cached_cookie_sid",
              "_reconciled_session", "resume_session_id"):
        assert k not in st.session_state
    assert st.session_state.get("session_id") is None
    st.session_state.clear()


def test_posthog_init_runs_js_and_escapes(monkeypatch):
    import ui_utils
    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    obj = MagicMock()
    obj.secrets = {"POSTHOG_API_KEY": "phc_k</script><b>"}
    ui_utils.inject_posthog(obj)
    assert obj.html.call_args.kwargs.get("unsafe_allow_javascript") is True
    assert "</script><b>" not in obj.html.call_args[0][0]


def test_capture_script_escapes_name_and_props():
    import ui_utils
    js = ui_utils.build_capture_script([("evil');alert(1);//</script>", {"k": "</script><img>"})])
    body = js[len("<script>"):-len("</script>")]
    assert "</" not in body
    assert '"evil\');alert(1);//<\\/script>"' in body  # stays inside a JSON string literal


def test_flush_pending_events_uses_js_html():
    import ui_utils
    st.session_state.clear()
    ui_utils.queue_event("upload", {"rows": 3})
    with patch.object(ui_utils.st, "html") as h:
        ui_utils.flush_pending_events()
    assert h.call_args.kwargs.get("unsafe_allow_javascript") is True
    assert st.session_state.pending_posthog_events == []
    st.session_state.clear()


def test_ci_workflow_and_runtime_pin():
    wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert "permissions:" in wf and "contents: read" in wf
    assert "ruff check" in wf and "pip-audit" in wf
    assert (ROOT / ".python-version").read_text().strip() == "3.13"
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert "ruff==" in dev and "pip-audit==" in dev
