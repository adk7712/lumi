"""Engine/export parity for failure paths, name de-duplication, imputation cap and regex safety."""
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from engine import apply_recipe, generate_pipeline_code
from engine_ops import predict_column_renames
import engine_ops
import codegen
import query_guard
from state_manager import downcast_dtypes


def run_script(recipe, df):
    ns = {}
    exec(generate_pipeline_code(recipe, []), ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ns['clean_data'](df.copy())
    # Only the script's own warnings: stray ResourceWarnings from other tests' GC (unclosed sqlite) are noise
    return out, [str(x.message) for x in w if not issubclass(x.category, ResourceWarning)]


def both(recipe, df):
    eng, msgs = apply_recipe(downcast_dtypes(df.copy()), recipe)
    exp, warns = run_script(recipe, df)
    assert_frame_equal(eng, exp, check_dtype=True)
    return eng, msgs, warns


# ---- 1. engine skips == export skips -------------------------------------------------

def test_cast_int_with_fraction_skips_in_both_paths():
    df = pd.DataFrame({'a': [1.5, 2.0, np.nan]})
    eng, msgs, warns = both([{'action': 'cast_type', 'column': 'a', 'dtype': 'int64'}], df)
    assert msgs and 'a' in msgs[0]
    assert warns
    assert eng['a'].tolist()[0] == 1.5


@pytest.mark.parametrize('how', ['mean', 'median'])
def test_fill_int_column_fractional_stat_is_safe(how):
    df = pd.DataFrame({'a': pd.array([1, 2, None, 2, 4], dtype='Int64')})
    eng, msgs, warns = both([{'action': 'fill_null', 'column': 'a', 'value': how}], df)
    assert not msgs and not warns
    assert eng['a'].isna().sum() == 0
    assert str(eng['a'].dtype).lower().startswith('int')


def test_export_wraps_data_dependent_steps_but_not_trivial_ones():
    code = generate_pipeline_code([
        {'action': 'replace', 'column': 'a', 'find': 'x', 'replace': 'y'},
        {'action': 'drop_duplicates'},
    ], [])
    assert code.count('except Exception') == 1


def test_export_replace_failure_warns_not_raises():
    # Replacement that cannot be stored back in an int dtype raises in astype
    df = pd.DataFrame({'a': pd.array([1, 2, 3], dtype='Int64')})
    recipe = [{'action': 'fill_null', 'column': 'a', 'value': 'oops'}]
    eng, msgs = apply_recipe(downcast_dtypes(df.copy()), recipe)
    exp, warns = run_script(recipe, df)
    assert_frame_equal(eng, exp)


# ---- 3. column-name de-duplication ---------------------------------------------------

def test_predict_renames_dedupes():
    assert predict_column_renames(['A b', 'a_b'], 'snake_case') == {'A b': 'a_b', 'a_b': 'a_b_2'}
    out = predict_column_renames(['a', 'A', 'a_2'], 'lowercase')
    assert sorted(out.values()) == ['a', 'a_2', 'a_2_2'] or len(set(out.values())) == 3


def test_dedupe_never_collides_with_existing():
    out = predict_column_renames(['A', 'a', 'a_2'], 'lowercase')
    assert len(set(out.values())) == 3
    assert out['a_2'] == 'a_2'


def test_normalize_names_parity_and_unique():
    df = pd.DataFrame([[1, 2, 3]], columns=['A b', 'a_b', 'a-b'])
    eng, _, _ = both([{'action': 'normalize_column_names', 'value': 'snake_case'}], df)
    assert list(eng.columns) == ['a_b', 'a_b_2', 'a_b_3']


def test_normalize_names_on_duplicate_input_columns():
    df = pd.DataFrame([[1, 2]], columns=['x', 'x'])
    eng, _, _ = both([{'action': 'normalize_column_names', 'value': 'snake_case'}], df)
    assert list(eng.columns) == ['x', 'x_2']


def test_scout_survives_duplicate_column_names():
    from scout import generate_proposals
    df = pd.DataFrame([[1, 2, 'a'], [3, 4, 'b']], columns=['x', 'x', 'y'])
    props = generate_proposals(df, set())
    assert any(p['column'] == '__dataset__' for p in props)
    df2 = pd.DataFrame({'A b': [1, 2], 'a_b': [3, 4]})
    assert isinstance(generate_proposals(df2, set()), list)


# ---- 5. imputation row cap -----------------------------------------------------------

def test_knn_row_cap_matches_export(monkeypatch):
    monkeypatch.setattr(engine_ops, 'KNN_MAX_ROWS', 50)
    monkeypatch.setattr(codegen, 'KNN_MAX_ROWS', 50)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'t': rng.normal(size=200), 'u': rng.normal(size=200)})
    df.loc[::7, 't'] = np.nan
    for how in ('knn', 'iterative'):
        eng, msgs, warns = both([{'action': 'fill_null', 'column': 't', 'value': how}], df)
        assert not msgs and eng['t'].isna().sum() == 0


# ---- 6. ReDoS ------------------------------------------------------------------------

@pytest.mark.parametrize('pat', ['(a+)+', '(a*)*', '(.*a){10}', '(x+x+)+y', '((a)+)*', '(?:a|b+)+', 'a' * 201])
def test_unsafe_regex_rejected(pat):
    assert query_guard.unsafe_regex_reason(pat)


@pytest.mark.parametrize('pat', ['[aA]pple', r'\d+', r'(foo|bar)', r'(ab)+', r'\(a+\)+', r'[(a+)+]', r'(a+)b', r'(a){1}'])
def test_safe_regex_allowed(pat):
    assert query_guard.unsafe_regex_reason(pat) is None


def test_replace_redos_skipped_in_both_paths():
    df = pd.DataFrame({'t': ['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!', 'b']})
    recipe = [{'action': 'replace', 'column': 't', 'find': '(a+)+$', 'replace': 'X', 'regex': True}]
    eng, msgs, warns = both(recipe, df)
    assert msgs and warns
    assert eng['t'].tolist() == df['t'].tolist()
    recipe[0]['column'] = 'All'
    eng, msgs, warns = both(recipe, df)
    assert msgs and warns


def test_query_guard_rejects_nested_quantifier_in_contains():
    with pytest.raises(query_guard.UnsafeQueryError):
        query_guard.validate_query_expression("t.str.contains('(a+)+$')")
    with pytest.raises(query_guard.UnsafeQueryError):
        query_guard.validate_query_expression("t.str.contains('(a+)+$', regex=True)")
    query_guard.validate_query_expression("t.str.contains('(a+)+$', regex=False)")
    query_guard.validate_query_expression("t.str.contains('^ab')")
