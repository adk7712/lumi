import numpy as np
import pandas as pd

from engine_ops import TRANSFORM_REGISTRY
from report import generate_evidence_report

STEP_FIXTURES = {
    "drop_column": {"column": "a"},
    "drop_nulls": {"column": "a"},
    "fill_null": {"column": "a", "value": "mean"},
    "cap_range": {"column": "a", "min": 0, "max": 5},
    "cast_type": {"column": "a", "dtype": "float64"},
    "drop_violated": {"rule": {"type": "Null Check", "col": "a", "desc": "a not null"}},
    "replace": {"column": "a", "find": "x", "replace": "y", "regex": False},
    "strip_whitespace": {"column": "a"},
    "normalize_text": {"column": "a", "value": "lowercase"},
    "log_transform": {"column": "a"},
    "rename_column": {"column": "a", "value": "b"},
    "reorder_columns": {"value": ["b", "a"]},
    "extract_datetime": {"column": "a", "component": "year", "new_column": "yr"},
    "drop_duplicates": {},
    "drop_empty_columns": {},
    "drop_empty_rows": {},
    "normalize_column_names": {"value": "snake_case"},
}


def _lineage(action):
    step = dict(STEP_FIXTURES[action], action=action)
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [], [step])
    return [l for l in out.splitlines() if l.startswith("1. ")][0]


def test_fixtures_cover_registry():
    assert set(STEP_FIXTURES) == set(TRANSFORM_REGISTRY)


def test_no_registry_action_uses_generic_fallback():
    for action in TRANSFORM_REGISTRY:
        line = _lineage(action)
        assert "custom action" not in line, action


def test_unknown_action_falls_back():
    assert "custom action" in _lineage_unknown()


def _lineage_unknown():
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [], [{"action": "bogus", "column": "a"}])
    return out


def test_new_descriptions_content():
    assert "log(1 + x)" in _lineage("log_transform")
    assert "`snake_case`" in _lineage("normalize_column_names")
    assert "`a`" in _lineage("drop_nulls")


def test_before_after_metrics():
    orig = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, None, None]})
    clean = orig.dropna()
    out = generate_evidence_report(clean, [], None, original_df=orig)
    assert "Data Cleaning Impact Metrics" in out
    assert "4 rows × 2 cols" in out and "1 rows × 2 cols" in out
    assert "| 3 | 0 | -3 |" in out
    assert "37.50%" in out and "0.00%" in out and "-37.50%" in out


def test_no_metrics_without_original():
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [])
    assert "Data Cleaning Impact Metrics" not in out


def test_rule_summary_and_range_check_ignores_nan():
    df = pd.DataFrame({"a": [1.0, 50.0, np.nan, 99.0]})
    rules = [
        {"type": "Range Check", "col": "a", "min": 0, "max": 10, "desc": "a in 0..10"},
        {"type": "Null Check", "col": "a", "desc": "a not null"},
        {"type": "Informational", "desc": "note"},
    ]
    out = generate_evidence_report(df, rules)
    assert "| Range Check | `a in 0..10` | FAILED | 2 |" in out  # NaN not counted
    assert "| Null Check | `a not null` | FAILED | 1 |" in out
    assert "| Informational | `note` | INFO | N/A |" in out
    assert "Total Rule Violations: 3" in out
    assert "`[1, 3]`" in out


def test_passed_rule_and_disabled_rule():
    df = pd.DataFrame({"a": [1, 2]})
    rules = [
        {"type": "Null Check", "col": "a", "desc": "ok"},
        {"type": "Null Check", "col": "a", "desc": "off", "enabled": False},
    ]
    out = generate_evidence_report(df, rules)
    assert "Total Active Rules: 1" in out
    assert "| Null Check | `ok` | PASSED | 0 |" in out
    assert "`off`" not in out
    assert "Violation Details" not in out


def test_violation_index_cap_100():
    df = pd.DataFrame({"a": [None] * 150})
    out = generate_evidence_report(df, [{"type": "Null Check", "col": "a", "desc": "nn"}])
    assert "**Violation Count:** 150" in out
    assert "(truncated)" in out
    line = [l for l in out.splitlines() if "Violating Row Indices" in l][0]
    assert "99" in line and "100," not in line and "149" not in line


def test_exactly_100_not_truncated():
    df = pd.DataFrame({"a": [None] * 100})
    out = generate_evidence_report(df, [{"type": "Null Check", "col": "a", "desc": "nn"}])
    assert "(truncated)" not in out


def test_rule_error_reported():
    df = pd.DataFrame({"a": [1]})
    out = generate_evidence_report(df, [{"type": "Range Check", "desc": "bad", "min": 0, "max": 1}])
    assert "| Range Check | `bad` | ERROR | 0 |" in out
    assert "Evaluation Error" in out


def test_empty_recipe_and_no_rules():
    for recipe in (None, []):
        out = generate_evidence_report(pd.DataFrame({"a": [1, 2]}), [], recipe)
        assert "Data Lineage" not in out
        assert "Total Rows Evaluated: 2" in out
        assert "Total Active Rules: 0" in out
        assert "Total Rule Violations: 0" in out
        assert "## Rule Evaluation Summary" in out
