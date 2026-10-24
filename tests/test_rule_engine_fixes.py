"""Regression tests A2-A5: nullable masks, relational robustness, text ops, vacuous truth.
Every case checks the engine AND the exported script (parity)."""
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from engine import apply_recipe, generate_pipeline_code
from rule_utils import evaluate_rule
from state_manager import downcast_dtypes


def script(recipe, df, rules=None):
    ns = {}
    exec(generate_pipeline_code(recipe, rules), ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ns['clean_data'](df.copy()), ns


def both(recipe, df, check_dtype=True):
    eng, msgs = apply_recipe(downcast_dtypes(df.copy()), recipe)
    exp, _ = script(recipe, df)
    assert_frame_equal(eng, exp, check_dtype=check_dtype)
    return eng, msgs


def violations(rule, df):
    eng = evaluate_rule(df, rule)
    assert eng.dtype == bool
    _, ns = script([], df, [rule])
    rep = ns['validate_data'](df.copy())
    n = int(eng.sum())
    assert rep.get(rule['desc'], 0) == n
    return eng


# ------------------------------------------------------------------ A2
RANGE = {'type': 'Range Check', 'col': 'a', 'min': 0, 'max': 10, 'desc': 'r'}


def test_a2_issue_repro_keeps_nan_row():
    df = pd.DataFrame({'a': [1.0, np.nan, 500, 4]})
    recipe = [{'action': 'cast_type', 'column': 'a', 'dtype': 'int64'},
              {'action': 'drop_violated', 'rule': RANGE}]
    out, _ = both(recipe, df)
    assert out['a'].isna().sum() == 1 and len(out) == 3


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_a2_range_nullable_mask_is_plain_bool(dtype):
    df = pd.DataFrame({'a': pd.array([1, None, 500, 4], dtype=dtype)})
    m = violations(RANGE, df)
    assert m.tolist() == [False, False, True, False]
    assert_frame_equal(apply_recipe(df, [{'action': 'drop_violated', 'rule': RANGE}])[0], df[[True, True, False, True]])


def test_a2_all_rule_types_return_bool_series():
    df = pd.DataFrame({'a': pd.array([1, None, 5], dtype='Int64'),
                       'b': pd.array([2, 2, None], dtype='Int64'),
                       's': pd.array(['x', None, 'y'], dtype='string')})
    rules = [
        {'type': 'Null Check', 'col': 'a'},
        RANGE,
        {'type': 'Relational Check', 'col_a': 'a', 'op': '<', 'target_type': 'Feature', 'col_b': 'b'},
        {'type': 'Relational Check', 'col_a': 'a', 'op': '!=', 'target_type': 'Value', 'value': 3},
        {'type': 'Relational Check', 'col_a': 's', 'op': '==', 'target_type': 'Value', 'value': 'x'},
        {'type': 'Custom Expression', 'query': 'a > 2'},
        {'type': 'Informational'},
    ]
    for i, r in enumerate(rules):
        m = evaluate_rule(df, dict(r, desc=f"d{i}"))
        assert isinstance(m, pd.Series) and m.dtype == bool and list(m.index) == list(df.index)


def test_a2_relational_null_semantics_float_and_nullable_agree():
    for op in [">", "<", ">=", "<=", "==", "!="]:
        r = {'type': 'Relational Check', 'col_a': 'a', 'op': op, 'target_type': 'Value', 'value': 2, 'desc': 'd'}
        f = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        n = pd.DataFrame({'a': pd.array([1, None, 3], dtype='Int64')})
        mf, mn = violations(r, f), violations(r, n)
        assert mf.tolist() == mn.tolist()
        assert mf.tolist()[1] is True or bool(mf.iloc[1])  # null is always a violation


# ------------------------------------------------------------------ A3
def test_a3_one_bad_cell_flags_only_that_row():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': pd.Series([10, 'x', 30], dtype=object)})
    r = {'type': 'Relational Check', 'col_a': 'a', 'op': '<', 'target_type': 'Feature', 'col_b': 'b', 'desc': 'd'}
    assert violations(r, df).tolist() == [False, True, False]
    out, _ = both([{'action': 'drop_violated', 'rule': r}], df)
    assert out['a'].tolist() == [1, 3]


def test_a3_scalar_value_and_mixed_object_column():
    df = pd.DataFrame({'a': pd.Series([1, 'x', None, 20], dtype=object)})
    r = {'type': 'Relational Check', 'col_a': 'a', 'op': '>=', 'target_type': 'Value', 'value': 5, 'desc': 'd'}
    assert violations(r, df).tolist() == [True, True, True, False]


def test_a3_incomparable_everywhere_and_unsupported_op():
    df = pd.DataFrame({'a': [1, 2]})
    r = {'type': 'Relational Check', 'col_a': 'a', 'op': '<', 'target_type': 'Value', 'value': 'zzz', 'desc': 'd'}
    assert violations(r, df).tolist() == [True, True]
    r2 = dict(r, op='~~')
    assert evaluate_rule(df, r2).tolist() == [True, True]


# ------------------------------------------------------------------ A4
def test_a4_bool_column_untouched():
    df = pd.DataFrame({'flag': [True, False, True], 't': [' a ', 'B', None]})
    for act in [{'action': 'strip_whitespace', 'column': 'flag'},
                {'action': 'normalize_text', 'column': 'flag', 'value': 'uppercase'},
                {'action': 'strip_whitespace', 'column': 'All'},
                {'action': 'normalize_text', 'column': 'All', 'value': 'lowercase'}]:
        out, _ = both([act], df)
        assert out['flag'].tolist() == [True, False, True]
        assert out['flag'].dtype == bool


@pytest.mark.parametrize("act", [
    {'action': 'strip_whitespace', 'column': 'o'},
    {'action': 'strip_whitespace', 'column': 'All'},
    {'action': 'normalize_text', 'column': 'o', 'value': 'uppercase'},
    {'action': 'normalize_text', 'column': 'All', 'value': 'titlecase'},
    {'action': 'normalize_text', 'column': 'o', 'value': 'remove_punctuation'},
])
def test_a4_mixed_object_values_keep_type(act):
    df = pd.DataFrame({'o': pd.Series([1, True, ' a b. ', None, 2.5], dtype=object)})
    out, _ = both([act], df)
    assert out['o'].iloc[0] == 1 and type(out['o'].iloc[0]) is int
    assert out['o'].iloc[1] is True
    assert out['o'].iloc[4] == 2.5
    assert out['o'].iloc[3] is None
    assert isinstance(out['o'].iloc[2], str)


def test_a4_text_still_transformed_and_numeric_skipped():
    df = pd.DataFrame({'t': [' Hi, ', None, 'yo '], 'n': [1.5, 2.5, 3.5]})
    out, msgs = both([{'action': 'strip_whitespace', 'column': 't'}], df)
    assert out['t'].tolist()[0] == 'Hi,' and out['t'].isna().iloc[1]
    out, msgs = both([{'action': 'normalize_text', 'column': 'n', 'value': 'remove_punctuation'}], df)
    assert out['n'].tolist() == [1.5, 2.5, 3.5]
    out, _ = both([{'action': 'normalize_text', 'column': 't', 'value': 'remove_punctuation'}], df)
    assert out['t'].iloc[0] == ' Hi '


def test_a4_string_dtype_preserved():
    df = pd.DataFrame({'t': pd.array([' a ', None, 'B'], dtype='string')})
    out, _ = both([{'action': 'strip_whitespace', 'column': 't'},
                   {'action': 'normalize_text', 'column': 't', 'value': 'uppercase'}], df)
    assert str(out['t'].dtype) == 'string' and out['t'].tolist()[0] == 'A' and pd.isna(out['t'].iloc[1])


# ------------------------------------------------------------------ A5
def test_a5_drop_empty_columns_zero_rows_is_noop():
    df = pd.DataFrame({'a': pd.Series([], dtype=float), 'b': pd.Series([], dtype=object)})
    out, msgs = both([{'action': 'drop_empty_columns'}], df)
    assert list(out.columns) == ['a', 'b']
    assert any('no rows' in m for m in msgs)


def test_a5_after_filtering_all_rows():
    df = pd.DataFrame({'a': [1, 2], 'b': [None, 3.0]})
    recipe = [{'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': 'a', 'min': 100, 'max': 200, 'desc': 'x'}},
              {'action': 'drop_empty_columns'}]
    out, _ = both(recipe, df)
    assert list(out.columns) == ['a', 'b'] and len(out) == 0


def test_a5_drop_empty_columns_still_works_and_zero_column_rows():
    df = pd.DataFrame({'a': [1, 2], 'e': [np.nan, np.nan]})
    out, _ = both([{'action': 'drop_empty_columns'}], df)
    assert list(out.columns) == ['a']
    nocols = pd.DataFrame(index=range(3))
    out, _ = both([{'action': 'drop_empty_rows'}], nocols)
    assert len(out) == 3
