"""Wave-1 engine/export fixes: engine output and exported script must agree."""
import warnings

import numpy as np
import pandas as pd
import pytest

from codegen import generate_pipeline_code
from engine import apply_recipe


def _both(step, df):
    out, msgs = apply_recipe(df.copy(), [step])
    ns = {}
    exec(generate_pipeline_code([step]), ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp = ns["clean_data"](df.copy())
    return out, msgs, exp, [str(x.message) for x in w]


def _same(out, exp, col):
    pd.testing.assert_series_equal(out[col].reset_index(drop=True), exp[col].reset_index(drop=True), check_dtype=False)


# A11: total fuzzy budget across "All" columns
def test_fuzzy_all_total_budget():
    df = pd.DataFrame({f"c{i}": [f"val{j}x" for j in range(900)] for i in range(4)})
    step = {"action": "normalize_text", "column": "All", "value": "fuzzy_dedupe"}
    out, msgs, exp, warns = _both(step, df)
    assert sum("budget" in m for m in msgs) == 1  # 3 columns fit in 3000, the 4th is skipped
    assert sum("budget" in w for w in warns) == 1
    for c in df.columns:
        _same(out, exp, c)


# A14
def test_fill_mean_median_ignores_inf():
    df = pd.DataFrame({"x": [1.0, 3.0, np.inf, np.nan]})
    for v, want in (("mean", 2.0), ("median", 2.0)):
        out, msgs, exp, _ = _both({"action": "fill_null", "column": "x", "value": v}, df)
        assert out["x"].iloc[3] == want and not msgs
        _same(out, exp, "x")


@pytest.mark.parametrize("method", ["knn", "iterative"])
def test_imputers_skip_empty_and_inf_features(method):
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0], "empty": [np.nan] * 5,
                       "inf": [1.0, np.inf, 2.0, 3.0, 4.0], "ok": [2.0, 4.0, 6.0, 8.0, 10.0]})
    out, msgs, exp, warns = _both({"action": "fill_null", "column": "x", "value": method}, df)
    assert not msgs and not warns
    assert out["x"].notna().all()
    _same(out, exp, "x")


def test_imputer_all_null_target_warns():
    df = pd.DataFrame({"x": [np.nan] * 3, "y": [1.0, 2.0, 3.0]})
    out, msgs, exp, warns = _both({"action": "fill_null", "column": "x", "value": "knn"}, df)
    assert msgs and warns and out["x"].isna().all()


# Q5
def test_two_digit_years_pivot():
    df = pd.DataFrame({"d": ["Aug-62", "Apr-05", "Jan-49", "Jan-50"]})
    step = {"action": "cast_type", "column": "d", "dtype": "datetime64[ns]"}
    out, msgs, exp, warns = _both(step, df)
    assert out["d"].tolist() == [pd.Timestamp(y, m, 1) for y, m in ((1962, 8), (2005, 4), (2049, 1), (1950, 1))]
    assert any("4 value(s)" in m and "two-digit" in m for m in msgs)
    assert any("4 value(s)" in w and "two-digit" in w for w in warns)
    _same(out, exp, "d")


def test_two_digit_year_dmy():
    df = pd.DataFrame({"d": ["15/01/05", "15/01/99"]})
    step = {"action": "cast_type", "column": "d", "dtype": "datetime64[ns]"}
    out, msgs, exp, _ = _both(step, df)
    assert out["d"].tolist() == [pd.Timestamp(2005, 1, 15), pd.Timestamp(1999, 1, 15)]
    _same(out, exp, "d")


def test_four_digit_years_unaffected():
    df = pd.DataFrame({"d": ["2024-01-15", "2023-12-31"]})
    out, msgs, _, _ = _both({"action": "cast_type", "column": "d", "dtype": "datetime64[ns]"}, df)
    assert not msgs


# Q6
def test_cast_strips_thousands_commas():
    df = pd.DataFrame({"n": ["25,649", "1,000.5", "7", None]})
    out, msgs, exp, warns = _both({"action": "cast_type", "column": "n", "dtype": "float64"}, df)
    assert out["n"].dropna().tolist() == [25649.0, 1000.5, 7.0]
    assert not msgs and not warns
    _same(out, exp, "n")


def test_cast_keeps_commas_when_not_all_clean():
    df = pd.DataFrame({"n": ["1,5", "2", "3"]})  # decimal comma-like: not stripped
    out, msgs, exp, warns = _both({"action": "cast_type", "column": "n", "dtype": "float64"}, df)
    assert out["n"].isna().tolist() == [True, False, False]
    assert any("1 value(s) in 'n' could not be converted" in m for m in msgs)
    assert any("1 value(s) in 'n' could not be converted" in w for w in warns)


# Q7
@pytest.mark.parametrize("vals", [
    ["No Hair", "No Hair", "Silver Hair", "Silver Hair", "Auburn Hair"],
])
def test_fuzzy_does_not_merge_distinct_words(vals):
    out, _, exp, _ = _both({"action": "normalize_text", "column": "c", "value": "fuzzy_dedupe"}, pd.DataFrame({"c": vals}))
    assert out["c"].tolist() == vals
    _same(out, exp, "c")


def test_fuzzy_still_merges_typos_and_is_deterministic():
    vals = ["Jon Smith", "John Smith", "ACME Inc", "Acme Inc."]
    out, _, exp, _ = _both({"action": "normalize_text", "column": "c", "value": "fuzzy_dedupe"}, pd.DataFrame({"c": vals}))
    assert out["c"].tolist() == ["John Smith", "John Smith", "ACME Inc", "ACME Inc"]  # tie -> alphabetical
    _same(out, exp, "c")


# Q9
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_cap_integer_column_rounds_bounds_inward(dtype):
    df = pd.DataFrame({"x": pd.array([0, 3, 10, 20], dtype=dtype)})
    out, _, exp, _ = _both({"action": "cap_range", "column": "x", "min": 2.5, "max": 12.7}, df)
    assert out["x"].tolist() == [3, 3, 10, 12]
    assert pd.api.types.is_integer_dtype(out["x"])
    assert pd.api.types.is_integer_dtype(exp["x"])
    _same(out, exp, "x")


# I12
def test_log_transform_warns_on_negatives():
    df = pd.DataFrame({"x": [-1.0, 0.0, 3.0, -5.0]})
    out, msgs, exp, warns = _both({"action": "log_transform", "column": "x"}, df)
    assert any("2 negative value(s)" in m for m in msgs)
    assert any("2 negative value(s)" in w for w in warns)
    _same(out, exp, "x")


# Q11
def test_report_wording_no_unconditional_success():
    import inspect, report
    assert "successfully executed" not in inspect.getsource(report)
