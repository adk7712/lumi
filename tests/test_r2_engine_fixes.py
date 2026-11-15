"""Round-2 engine fixes: resolution-step rule snapshots, ReDoS guard, scout skew, dup-column export."""
import sys
import time
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from engine import apply_recipe, generate_pipeline_code
from rule_utils import create_resolution_step, evaluate_rule
from query_guard import unsafe_regex_reason
import query_guard


# ---- 1. resolution step must not alias the live rule ----------------------------------

RULES = [
    {"type": "Range Check", "col": "Age", "min": 30.0, "max": 60.0, "desc": "Age 30-60", "color": "x", "enabled": True},
    {"type": "Relational Check", "col_a": "Age", "op": ">", "target_type": "Value", "value": 30, "desc": "Age > 30"},
    {"type": "Custom Expression", "query": "Age > 60", "desc": "Violates: Age > 60"},
]


@pytest.mark.parametrize("rule", RULES)
def test_resolution_step_is_snapshot(rule):
    live = dict(rule)
    step = create_resolution_step(live, "Drop Rows" if live["type"] == "Range Check" else "Drop Violated Rows")
    assert step["action"] == "drop_violated"
    assert step["rule"] is not live
    live["col"] = "Years"
    live["col_a"] = "Years"
    live["query"] = "Years > 60"
    assert step["rule"].get("col") in (None, "Age")
    assert step["rule"].get("col_a") in (None, "Age")
    assert step["rule"].get("query") in (None, "Age > 60")


def test_range_drop_survives_rename_replay():
    raw = pd.DataFrame({"Id": [1, 2, 3, 4], "Age": [25, 80, 40, 90], "City": list("abcd")})
    rule = dict(RULES[0])
    step = create_resolution_step(rule, "Drop Rows")
    recipe = [step, {"action": "rename_column", "column": "Age", "value": "Years"}]
    rule["col"] = "Years"  # what sync_column_rename does to the live rule
    out, _ = apply_recipe(raw, recipe)
    assert out["Id"].tolist() == [3]


def test_other_resolutions_snapshot_scalars():
    rule = dict(RULES[0])
    for m in ("Cap at Bounds", "Log Transform"):
        step = create_resolution_step(rule, m)
        assert step.get("rule") is None or step["rule"] is not rule


# ---- 2. ReDoS guard --------------------------------------------------------------------

BAD = [r"^(a|a)+$", r"^(a|aa)+$", r"(a|a)*$", r"^(\w|a)+$", r"^(a?){30}a{30}$",
       r"(a+)+$", r"^(a*)*$", r"(a|b){1000}", r"^(ab?)+$"]
GOOD = [r"(abc)+", r"[a-z]+", r"\d{3}-\d{4}", r"^(abc|def)$", r"^[A-Z]{2}\d{4}$", r"(ab)?c", r"\s+", r"(\d{3})-(\d{4})"]


@pytest.mark.parametrize("pat", BAD)
def test_redos_patterns_rejected(pat):
    t = time.perf_counter()
    assert unsafe_regex_reason(pat)
    assert time.perf_counter() - t < 0.5


@pytest.mark.parametrize("pat", GOOD)
def test_ordinary_patterns_allowed(pat):
    assert unsafe_regex_reason(pat) is None


def test_engine_skips_alternation_redos_fast():
    df = pd.DataFrame({"t": ["a" * 40 + "b"] * 3})
    t = time.perf_counter()
    out, msgs = apply_recipe(df, [{"action": "replace", "column": "t", "find": r"^(a|a)+$", "replace": "x", "regex": True}])
    assert time.perf_counter() - t < 1.0
    assert any("unsafe" in m for m in msgs)
    assert out["t"].tolist() == df["t"].tolist()


# ---- 3. scout skew with NA -------------------------------------------------------------

def test_scout_nullable_short_series():
    from scout import generate_proposals
    df = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64"), "b": pd.array([1.0, None], dtype="Float64")})
    generate_proposals(df, set())


# ---- 4. duplicate column names: export == engine --------------------------------------

@pytest.mark.parametrize("step", [
    {"action": "drop_empty_columns"},
    {"action": "log_transform", "column": "a"},
    {"action": "strip_whitespace", "column": "a"},
    {"action": "strip_whitespace", "column": "All"},
])
def test_duplicate_columns_export_does_not_raise(step):
    df = pd.DataFrame([[1, 2, None], [3, 4, None]], columns=["a", "a", "b"])
    df.iloc[:, 0] = df.iloc[:, 0].astype(object)
    eng, _ = apply_recipe(df.copy(), [step])  # engine must not raise
    ns = {}
    exec(generate_pipeline_code([step], []), ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns["clean_data"](df.copy())


# ---- 5. misc parity --------------------------------------------------------------------

def test_unknown_normalize_method_empty_name_agrees():
    df = pd.DataFrame([[1, 2]], columns=["", "b"])
    step = {"action": "normalize_column_names", "value": "bogus"}
    eng, _ = apply_recipe(df.copy(), [step])
    ns = {}
    exec(generate_pipeline_code([step], []), ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = ns["clean_data"](df.copy())
    assert list(eng.columns) == list(exp.columns)


def test_relational_unsupported_op_missing_column_consistent():
    rule = {"type": "Relational Check", "col_a": "nope", "op": "~~", "target_type": "Value", "value": 1, "desc": "r"}
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        evaluate_rule(df, rule)
    ns = {}
    exec(generate_pipeline_code([], [rule]), ns)
    res = ns["validate_data"](df)
    assert isinstance(res["r"], str) and res["r"].startswith("Error")
