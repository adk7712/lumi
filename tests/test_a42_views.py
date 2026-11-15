"""View-layer regressions: HTML escaping, content-keyed caches, stable widget keys,
atomic step removal, save ordering, violation-browser slicing, shared rule masks."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

import state_manager as sm
import lumi_state.steps as steps
import rule_utils

APP_PATH = str(Path(__file__).parent.parent / "app.py")
EVIL = "<img src=x onerror=alert(1)>"
CSV = ("Id,Age," + EVIL + "\n1,25,a\n2,30,b\n3,,<b>c</b>\n").encode()


def _loaded(csv=CSV):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", csv, "text/csv").run()
    assert not at.exception
    return at


def _btn(at, key):
    return next((b for b in at.button if b.key == key), None)


def _markdown(at):
    return [m.value for m in at.markdown]


def _no_raw_tag(at):
    for v in _markdown(at):
        assert "<img" not in v, v
        assert "<b>c</b>" not in v, v


def test_html_is_escaped_in_diagnostics_rulebook_and_audit_log():
    at = _loaded()
    at.session_state["active_features"] = [EVIL]
    at.session_state["rules"] = [{"type": "Null Check", "col": EVIL, "desc": EVIL + " is NOT NULL",
                                  "enabled": True, "color": "hsla(1, 70%, 50%, 0.4)", "error": EVIL}]
    at.session_state["proposals"] = [{"type": EVIL, "column": EVIL, "title": EVIL, "reason": EVIL,
                                      "rule_data": {"type": "Null Check", "col": EVIL, "desc": "x"}}]
    at.session_state["intermediate_states"] = [("orig", 100, 3), (EVIL + " on " + EVIL, 90, 3)]
    at.session_state["cleaning_recipe"] = [{"action": "drop_duplicates"}]
    at.run()
    assert not at.exception
    _no_raw_tag(at)
    joined = "".join(_markdown(at))
    assert "&lt;img" in joined


def test_overview_metrics_cache_tracks_content_and_never_mutates_cached_list():
    at = _loaded(b"Id,Age,Name\n1,25, a \n2,30,b\n3,,c\n")
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "Age is NOT NULL",
                                  "enabled": True, "color": "red"}]
    at.run()
    sig1, metrics1 = at.session_state["_overview_metrics"]
    assert metrics1["total_violations"] == 1
    ws_before = list(metrics1["whitespace_cols"])
    at.run()
    at.run()
    assert at.session_state["_overview_metrics"][1]["whitespace_cols"] == ws_before
    # Toggling a rule keeps lengths equal but must refresh the metrics (one cache entry only)
    rules = [dict(r) for r in at.session_state["rules"]]
    rules[0]["enabled"] = False
    at.session_state["rules"] = rules
    at.run()
    sig2, metrics2 = at.session_state["_overview_metrics"]
    assert sig2 != sig1 and metrics2["total_violations"] == 0
    assert [k for k in at.session_state.filtered_state if k.startswith("_overview_metrics")] == ["_overview_metrics"]


def test_evidence_report_cache_refreshes_on_rule_edit_and_workspace():
    at = _loaded(b"Id,Age\n1,25\n2,30\n3,200\n")
    rule = {"type": "Range Check", "col": "Age", "min": 0.0, "max": 100.0, "desc": "Age in [0, 100]",
            "enabled": True, "color": "red"}
    at.session_state["rules"] = [rule]
    at.run()
    k1 = at.session_state["report_cache_key"]
    at.session_state["rules"] = [dict(rule, max=500.0, desc="Age in [0, 500]")]
    at.run()
    assert at.session_state["report_cache_key"] != k1
    k2 = at.session_state["report_cache_key"]
    at.session_state["session_id"] = "other-workspace"
    at.run()
    assert at.session_state["report_cache_key"] != k2


def test_rule_widget_keys_are_stable_across_removal():
    at = _loaded(b"Id,Age,Sex\n1,,m\n2,3,\n")
    mk = lambda col: {"type": "Null Check", "col": col, "desc": f"{col} is NOT NULL",
                      "enabled": True, "color": "red"}
    at.session_state["rules"] = [mk("Age"), mk("Sex")]
    at.run()
    uid_sex = steps.rule_uids(at.session_state["rules"])[1]
    key = f"res_{uid_sex}"
    at.selectbox(key=key).select("Drop Rows")
    at.run()
    # remove the first rule: the Sex rule must keep its own selection
    _btn(at, f"del_{steps.rule_uids(at.session_state['rules'])[0]}").click().run()
    assert [r["col"] for r in at.session_state["rules"]] == ["Sex"]
    assert at.selectbox(key=key).value == "Drop Rows"


def test_diagnostics_ignores_stale_active_features():
    at = _loaded(b"Id,Age\n1,25\n2,30\n")
    at.session_state["active_features"] = ["Age", "GoneColumn"]
    at.run()
    assert not at.exception
    assert at.session_state["active_features"] == ["Age"]


def test_audit_log_remove_restores_state_on_failure(monkeypatch):
    from views import audit_log
    at = _loaded(b"Id,Age\n1,25\n1,25\n2,30\n")
    _btn(at, "qa_drop_duplicates").click().run()
    assert len(at.session_state["cleaning_recipe"]) == 1
    before = (list(at.session_state["cleaning_recipe"]), list(at.session_state["intermediate_states"]))

    def boom(*a, **k):
        raise RuntimeError("replay failed")
    monkeypatch.setattr(audit_log, "replay_steps", boom)
    monkeypatch.setattr(audit_log, "get_state_at_step", boom)
    _btn(at, "rm_step_1").click().run()
    assert any("replay failed" in e.value for e in at.error)
    assert (list(at.session_state["cleaning_recipe"]), list(at.session_state["intermediate_states"])) == before
    assert len(at.session_state["current_df"]) == 2


def test_redo_saves_once_after_rules_are_reresolved(monkeypatch):
    at = _loaded(b"Id,Age\n1,\n2,30\n")
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "Age is NOT NULL",
                                  "enabled": True, "color": "red", "resolved": False}]
    at.run()
    at.session_state["redo_stack"] = [{"action": "drop_missing_rows", "column": "Age",
                                       "_unresolved_rules": [(0, "Age is NOT NULL")]}]
    snapshots = []
    orig = sm.reresolve_rules
    monkeypatch.setattr(sm, "reresolve_rules", lambda ch: (orig(ch), snapshots.append("reresolved")))
    monkeypatch.setattr(sm, "save_db_session", lambda *a, **k: snapshots.append("saved"))
    monkeypatch.setattr(steps, "save_db_session", lambda *a, **k: snapshots.append("saved"))
    _btn(at, "redo_btn").click().run()
    assert snapshots and snapshots[0] == "reresolved" and snapshots.count("saved") == 1


def test_replay_steps_persist_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(steps, "save_session_state", lambda: calls.append("s"))
    monkeypatch.setattr(steps, "save_db_session", lambda: calls.append("d"))
    monkeypatch.setattr(steps, "_update_project_cache", lambda: calls.append("p"))
    steps.replay_steps([], persist=False)
    assert calls == []
    steps.replay_steps([])
    assert calls == ["s", "d", "p"]


def test_violation_browser_styles_only_the_preview_slice(monkeypatch):
    import ui_utils
    from views import violation_browser
    n = ui_utils.VIOLATION_PREVIEW_LIMIT + 50
    csv = "Id,Age\n" + "\n".join(f"{i}," for i in range(n))
    sizes = []
    real = violation_browser.get_heatmap_styles
    monkeypatch.setattr(violation_browser, "get_heatmap_styles",
                        lambda d, r: (sizes.append(len(d)), real(d, r))[1])
    at = _loaded(csv.encode())
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "Age is NOT NULL",
                                  "enabled": True, "color": "red"}]
    at.run()
    assert not at.exception
    assert sizes and max(sizes) == ui_utils.VIOLATION_PREVIEW_LIMIT


def test_rule_masks_are_computed_once_per_run(monkeypatch):
    calls = []
    real = rule_utils.evaluate_rule
    def counting(df, rule):
        calls.append(rule.get("desc"))
        return real(df, rule)
    import ui_utils
    monkeypatch.setattr(rule_utils, "evaluate_rule", counting)
    monkeypatch.setattr(ui_utils, "evaluate_rule", lambda d, r: real(d, r), raising=False)
    from views import rulebook
    monkeypatch.setattr(rulebook, "generate_evidence_report", lambda *a, **k: "report")
    at = _loaded(b"Id,Age\n1,\n2,30\n")
    at.session_state["rules"] = [{"type": "Null Check", "col": "Age", "desc": "AGE_RULE",
                                  "enabled": True, "color": "red"}]
    calls.clear()
    at.run()
    assert calls.count("AGE_RULE") == 1
    calls.clear()
    at.run()
    assert calls.count("AGE_RULE") == 0
