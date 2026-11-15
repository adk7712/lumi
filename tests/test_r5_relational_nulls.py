"""Owner decision (R5): a null on either side of a Relational Check is NOT a violation
(same as Range Check), so Drop Violated Rows keeps rows with missing values.
Engine, exported script, validation counts and Drop Violated Rows must agree."""
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from engine import apply_recipe, generate_pipeline_code
from rule_utils import evaluate_rule, create_resolution_step

OPS = [">", "<", "==", "!=", ">=", "<="]
NA = {"float": np.nan, "Int64": pd.NA, "object": None}


def _col(kind, vals):
    vals = [NA[kind] if v is None else v for v in vals]
    if kind == "Int64":
        return pd.array(vals, dtype="Int64")
    if kind == "float":
        return pd.Series(vals, dtype="float64")
    return pd.Series(vals, dtype=object)


def _export(rule):
    ns = {}
    exec(generate_pipeline_code([{"action": "drop_violated", "rule": rule}], [rule]), ns)
    return ns


@pytest.mark.parametrize("kind", ["float", "Int64", "object"])
@pytest.mark.parametrize("op", OPS)
def test_feature_nulls_never_violate_and_export_matches(kind, op):
    df = pd.DataFrame({
        "a": _col(kind, [1, 2, 3, None, None, 5]),
        "b": _col(kind, [2, 2, 1, 2, None, None]),
    })
    rule = {"type": "Relational Check", "col_a": "a", "op": op, "col_b": "b",
            "target_type": "Feature", "desc": "d"}
    mask = evaluate_rule(df, rule)
    assert mask.dtype == bool
    assert not mask.iloc[3:].any()  # any null side -> not a violation
    ns = _export(rule)
    assert ns["_relational_violations"](df["a"], df["b"], op).tolist() == mask.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ns["clean_data"](df.copy())
    assert len(out) == int((~mask).sum())
    assert ns["validate_data"](df.copy()).get("d", 0) == int(mask.sum())


@pytest.mark.parametrize("kind", ["float", "Int64", "object"])
@pytest.mark.parametrize("op", OPS)
def test_value_nulls_never_violate_and_export_matches(kind, op):
    df = pd.DataFrame({"a": _col(kind, [1, 2, 3, None])})
    rule = {"type": "Relational Check", "col_a": "a", "op": op, "value": 2,
            "target_type": "Value", "desc": "d"}
    mask = evaluate_rule(df, rule)
    assert not mask.iloc[3]
    ns = _export(rule)
    assert ns["_relational_violations"](df["a"], 2, op).tolist() == mask.tolist()


def test_null_constant_and_mixed_type_cells():
    df = pd.DataFrame({"a": pd.Series([1, "x", None], dtype=object)})
    rule = {"type": "Relational Check", "col_a": "a", "op": ">=", "value": 0, "target_type": "Value"}
    assert evaluate_rule(df, rule).tolist() == [False, True, False]  # only the incomparable cell
    rule = dict(rule, value=np.nan)
    assert not evaluate_rule(df, rule).any()
    rule = dict(rule, op="??", value=1)  # unsupported operator still flags every row
    assert evaluate_rule(df, rule).all()


def test_drop_violated_rows_keeps_null_rows():
    df = pd.DataFrame({"a": [1.0, np.nan, 5.0, 3.0], "b": [2.0, 2.0, 1.0, np.nan]})
    rule = {"type": "Relational Check", "col_a": "a", "op": "<", "col_b": "b",
            "target_type": "Feature", "desc": "a < b"}
    step = create_resolution_step(rule, "Drop Violated Rows")
    assert step["action"] == "drop_violated"
    out, _ = apply_recipe(df.copy(), [step])
    assert out.index.tolist() == [0, 1, 3]  # only the genuine violation (5 < 1) is dropped
    ns = _export(rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ns["clean_data"](df.copy()).index.tolist() == [0, 1, 3]
