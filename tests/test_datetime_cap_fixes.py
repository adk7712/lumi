"""V4 (int -> datetime) and V5 (cap_range dtype) fixes: engine/codegen parity."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from engine import apply_recipe
from state_manager import downcast_dtypes
from test_codegen_parity import run_script


def both(df, recipe):
    eng, msgs = apply_recipe(downcast_dtypes(df.copy()), recipe)
    script, _, warns = run_script(recipe, df)
    return eng, msgs, script, warns


CAST = [{'action': 'cast_type', 'column': 'c', 'dtype': 'datetime64[ns]'}]


def extract(comp):
    return [{'action': 'extract_datetime', 'column': 'c', 'new_column': 'o', 'component': comp}]


@pytest.mark.parametrize("vals,expect", [
    ([2008, 2010, 1999], ['2008-01-01', '2010-01-01', '1999-01-01']),
    ([20080315, 19991231, 20240229], ['2008-03-15', '1999-12-31', '2024-02-29']),
])
def test_cast_int_years_and_yyyymmdd(vals, expect):
    df = pd.DataFrame({'c': vals})
    eng, msgs, script, _ = both(df, CAST)
    assert msgs == []
    assert list(eng['c']) == list(pd.to_datetime(expect))
    assert_frame_equal(eng, script)


def test_cast_integral_float_with_nan_is_years():
    df = pd.DataFrame({'c': [2008.0, np.nan, 2010.0]})
    eng, msgs, script, _ = both(df, CAST)
    assert eng['c'].iloc[0].year == 2008 and pd.isna(eng['c'].iloc[1])
    assert_frame_equal(eng, script)


@pytest.mark.parametrize("vals", [[5, 100, 2008], [1.5, 2008.0], [99999999, 20080101], [2008, 20080101]])
def test_cast_ambiguous_numeric_left_unchanged_with_warning(vals):
    df = pd.DataFrame({'c': vals})
    eng, msgs, script, warns = both(df, CAST)
    assert len(msgs) == 1 and 'Warning' in msgs[0]
    assert_frame_equal(eng, downcast_dtypes(df))
    assert_frame_equal(script, downcast_dtypes(df))
    assert len(warns) >= 1


def test_cast_strings_unchanged_behaviour():
    df = pd.DataFrame({'c': ['2024-01-05', 'bad', None]})
    eng, msgs, script, _ = both(df, CAST)
    assert msgs == [] and eng['c'].iloc[0] == pd.Timestamp('2024-01-05')
    assert_frame_equal(eng, script)


def test_extract_year_from_int_year_column():
    df = pd.DataFrame({'c': [2008, 2009, 2010]})
    eng, msgs, script, _ = both(df, extract('year'))
    assert list(eng['o']) == [2008, 2009, 2010]
    assert_frame_equal(eng, script)


def test_extract_month_from_yyyymmdd():
    df = pd.DataFrame({'c': [20080315, 20091101]})
    eng, _, script, _ = both(df, extract('month'))
    assert list(eng['o']) == [3, 11]
    assert_frame_equal(eng, script)


def test_extract_from_unplausible_ints_creates_no_column():
    df = pd.DataFrame({'c': [1, 2, 3]})
    eng, msgs, script, warns = both(df, extract('year'))
    assert 'o' not in eng.columns and 'o' not in script.columns
    assert len(msgs) == 1 and warns


# ---------------------------------------------------------------- cap_range

def cap(lo, hi):
    return [{'action': 'cap_range', 'column': 'c', 'min': lo, 'max': hi}]


@pytest.mark.parametrize("dtype", ['int64', 'int32', 'int8', 'Int64'])
def test_cap_int_bounds_integral_preserves_dtype(dtype):
    df = pd.DataFrame({'c': pd.array([-5, 3, 50, 7], dtype=dtype)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        eng, msgs, script, warns = both(df, cap(0, 10))
    assert msgs == [] and eng['c'].dtype == downcast_dtypes(df)['c'].dtype
    assert list(eng['c']) == [0, 3, 10, 7]
    assert not [w for w in warns if 'Future' in w or 'incompatible' in w]
    assert_frame_equal(eng, script)


def test_cap_int_with_float_valued_integral_bounds_keeps_int():
    df = pd.DataFrame({'c': [-5, 3, 50]})
    eng, _, script, _ = both(df, cap(0.0, 10.0))
    assert pd.api.types.is_integer_dtype(eng['c'])
    assert_frame_equal(eng, script)


def test_cap_int_with_fractional_bound_becomes_float():
    df = pd.DataFrame({'c': [-5, 3, 50]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        eng, _, script, _ = both(df, cap(0.5, 10.5))
    assert eng['c'].dtype == np.float64 and list(eng['c']) == [0.5, 3.0, 10.5]
    assert_frame_equal(eng, script)


def test_cap_int8_bounds_beyond_dtype_range():
    df = pd.DataFrame({'c': pd.array([-100, 5, 100], dtype='int8')})
    eng, _, script, _ = both(df, cap(-1000, 1000))
    assert eng['c'].dtype == np.int8 and list(eng['c']) == [-100, 5, 100]
    assert_frame_equal(eng, script)


def test_cap_float_with_nan():
    df = pd.DataFrame({'c': [-1.0, np.nan, 20.0]})
    eng, _, script, _ = both(df, cap(0, 10))
    assert pd.api.types.is_float_dtype(eng['c']) and eng['c'].isna().sum() == 1
    assert_frame_equal(eng, script)
