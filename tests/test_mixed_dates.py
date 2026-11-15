"""Mixed date formats are parsed, and unreadable values are reported (A39)."""
import warnings

import pandas as pd

from codegen import generate_pipeline_code
from engine import apply_recipe

MIXED = ["2024-01-15", "15/01/2024", "Jan 5, 2024", "2024-02-01", None]


def _export(step, df):
    ns = {}
    exec(generate_pipeline_code([step]), ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ns["clean_data"](df.copy())
    return out, [str(x.message) for x in w]


def test_cast_parses_mixed_formats_in_engine_and_export():
    step = {"action": "cast_type", "column": "d", "dtype": "datetime64[ns]"}
    df = pd.DataFrame({"d": MIXED})
    out, msgs = apply_recipe(df.copy(), [step])
    assert out["d"].notna().sum() == 4
    assert not [m for m in msgs if "could not be read" in m]
    exp, _ = _export(step, df)
    pd.testing.assert_series_equal(out["d"], exp["d"])


def test_unreadable_dates_are_reported():
    step = {"action": "cast_type", "column": "d", "dtype": "datetime64[ns]"}
    df = pd.DataFrame({"d": ["2024-01-15", "not a date", "", None]})
    out, msgs = apply_recipe(df.copy(), [step])
    assert any("1 value(s) in 'd' could not be read as dates" in m for m in msgs)
    _, warns = _export(step, df)
    assert any("1 value(s) in 'd' could not be read as dates" in m for m in warns)


def test_extract_datetime_uses_mixed_parser():
    step = {"action": "extract_datetime", "column": "d", "component": "year", "new_column": "y"}
    df = pd.DataFrame({"d": MIXED})
    out, _ = apply_recipe(df.copy(), [step])
    assert out["y"].tolist()[:4] == [2024, 2024, 2024, 2024]
    exp, _ = _export(step, df)
    pd.testing.assert_series_equal(out["y"], exp["y"])
