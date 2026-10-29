"""Round-3 UI regressions: stable stateful tabs, Rulebook pd.NA bounds, current proposals,
Audit Log numbering."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")
TAB_NAMES = ["Overview", "Diagnostics", "Rulebook", "Transformations", "Audit Log", "Pipeline Preview"]


def _loaded(csv):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    assert not at.exception
    return at


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _tab_container(at):
    """(index among main's top-level children, block proto) of the main st.tabs container."""
    for i, child in enumerate(at.main.children.values()):
        proto = getattr(child, "proto", None)
        if type(proto).__name__ == "Block" and proto.WhichOneof("type") == "tab_container":
            return i, proto
    raise AssertionError("main tab container not found")


def test_tabs_position_stable_when_top_elements_appear():
    """Conditional elements above the tabs (queued PostHog script, toasts) must not shift the
    tabs in the element tree: a shift made the browser remount st.tabs on the first tab."""
    at = _loaded(b"A,B\n1,2\n3,4\n")
    pos, _ = _tab_container(at)
    at.session_state["pending_posthog_events"] = [("probe_event", {"x": 1})]
    at.run()
    assert _tab_container(at)[0] == pos
    at.run()  # queue flushed: the script is gone again
    assert _tab_container(at)[0] == pos


def test_selected_tab_is_server_state_and_survives_widget_reruns():
    at = _loaded(b"A,B\n1,2\n3,4\n")
    _, proto = _tab_container(at)
    assert proto.tab_container.id, "tabs must be stateful (keyed, on_change) widgets"
    at.session_state["_main_tab_selected"] = "Transformations"
    at.run()
    assert _tab_container(at)[1].tab_container.default_tab_index == TAB_NAMES.index("Transformations")
    # Interact with widgets in other views: the selected tab is re-sent unchanged every run
    at.selectbox(key="trans_type_select").select("Normalize Column Names").run()
    _btn(at, "btn_norm_cols").click().run()
    assert not at.exception
    assert _tab_container(at)[1].tab_container.default_tab_index == TAB_NAMES.index("Transformations")


def test_tab_query_param_opens_that_tab():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.query_params["tab"] = "Audit Log"
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", b"A\n1\n2\n", "text/csv").run()
    assert not at.exception
    assert _tab_container(at)[1].tab_container.default_tab_index == TAB_NAMES.index("Audit Log")


def test_numeric_bounds_handles_all_na_and_empty():
    from views.rulebook import _numeric_bounds
    assert _numeric_bounds(pd.Series([pd.NA, pd.NA], dtype="Int64")) == (0.0, 0.0)
    assert _numeric_bounds(pd.Series([], dtype="Float64")) == (0.0, 0.0)
    assert _numeric_bounds(pd.Series([float("nan")])) == (0.0, 0.0)
    assert _numeric_bounds(pd.Series([3, 1, pd.NA], dtype="Int64")) == (1.0, 3.0)


def test_rulebook_range_on_all_na_nullable_column_does_not_crash():
    at = _loaded(b"Id,Flag\n1,True\n2,False\n3,\n")
    at.selectbox(key="trans_type_select").select("Cast Data Type").run()
    at.selectbox(key="cast_target_col").select("Flag").run()
    at.selectbox(key="cast_dtype_select").select("int64").run()
    _btn(at, "btn_cast").click().run()
    assert not at.exception
    at.selectbox(key="rule_type_select").select("Range Check").run()
    at.selectbox(key="rule_target_col").select("Flag").run()
    at.number_input(key="range_min_input").set_value(18.0).run()
    at.number_input(key="range_max_input").set_value(60.0).run()
    _btn(at, "btn_add_range").click().run()
    assert not at.exception
    res = next(s for s in at.selectbox if (s.key or "").startswith("range_res_"))
    res.select("Drop Rows").run()
    _btn(at, "btn_" + res.key).click().run()
    assert not at.exception, at.exception
    at.run()  # Rulebook re-renders the Range Check form over the now-empty column
    assert not at.exception, at.exception


def test_proposals_after_undo_target_current_columns():
    at = _loaded(b"Total Amount,B\n1,\n2,\n3,\n4,5\n")
    at.selectbox(key="trans_type_select").select("Normalize Column Names").run()
    _btn(at, "btn_norm_cols").click().run()
    at.selectbox(key="trans_type_select").select("Rename Column").run()
    at.selectbox(key="rename_target_col").select("total_amount").run()
    at.text_input(key="rename_new_name_input").set_value("amt").run()
    _btn(at, "btn_rename").click().run()
    _btn(at, "undo_btn").click().run()  # back to total_amount, B
    assert not at.exception
    cols = set(at.session_state["current_df"].columns)
    for p in at.session_state["proposals"]:
        data = p.get("rule_data") or {}
        for ref in [p.get("column")] + [data.get(f) for f in ("col", "col_a", "col_b", "column")]:
            assert ref in (None, "__dataset__") or ref in cols, (p, cols)
        # The already-applied name normalization is not offered again
        assert data.get("action") != "normalize_column_names"
    _btn(at, "redo_btn").click().run()
    assert not at.exception
    cols = set(at.session_state["current_df"].columns)
    assert "amt" in cols
    for p in at.session_state["proposals"]:
        data = p.get("rule_data") or {}
        assert p.get("column") in cols | {"__dataset__"}, p
        assert data.get("col") in (None,) or data.get("col") in cols, p


def test_proposal_covered_by_active_rule_is_not_reoffered():
    at = _loaded(b"A,B\n1,\n2,\n3,\n4,5\n")
    null_props = [p for p in at.session_state["proposals"]
                  if (p.get("rule_data") or {}).get("type") == "Null Check"]
    if not null_props:
        return  # scout heuristics changed; nothing to check
    col = null_props[0]["rule_data"]["col"]
    at.selectbox(key="rule_type_select").select("Null Check").run()
    at.selectbox(key="rule_target_col").select(col).run()
    _btn(at, "btn_add_null").click().run()
    at.run()
    assert not any((p.get("rule_data") or {}).get("type") == "Null Check"
                   and (p.get("rule_data") or {}).get("col") == col
                   for p in at.session_state["proposals"])


def test_audit_log_blocker_numbers_match_visible_list():
    at = _loaded(b"Total Amount,B\n1,2\n2,3\n3,4\n")
    at.selectbox(key="trans_type_select").select("Normalize Column Names").run()
    _btn(at, "btn_norm_cols").click().run()
    at.selectbox(key="trans_type_select").select("Rename Column").run()
    at.selectbox(key="rename_target_col").select("total_amount").run()
    at.text_input(key="rename_new_name_input").set_value("amt").run()
    _btn(at, "btn_rename").click().run()
    # Visible list: 1. Original Data, 2. normalize, 3. rename
    _btn(at, "rm_step_1").click().run()
    msgs = [e.value for e in at.error]
    assert any("Cannot remove step 2" in m and "step 3 (rename_column)" in m for m in msgs), msgs
