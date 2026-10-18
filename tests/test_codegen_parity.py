"""Parity tests: the exported standalone script (codegen.clean_data / validate_data)
must produce exactly what the in-app engine produces.

The app downcasts numeric dtypes on upload (state_manager.downcast_dtypes) before the
recipe ever runs, so the engine side is fed the downcast frame while the exported
script is fed the raw frame (it downcasts itself as its first step).
"""
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from engine import apply_recipe, generate_pipeline_code, generate_notebook_code
from engine_ops import TRANSFORM_REGISTRY
from rule_utils import evaluate_rule
from state_manager import downcast_dtypes


# --------------------------------------------------------------------------- helpers

def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        'int_col': [1, 2, 3, 2, 50, 2],
        'float_col': [1.5, np.nan, 3.25, 1.5, 100.0, np.nan],
        'neg_float': [-2.0, 0.5, np.nan, 4.0, 10.0, 0.5],
        'text': ['  Apple ', 'banana', np.nan, 'APPLE!', 'Cherry, pie', 'banana'],
        'Mixed Case-Col': ['Foo', 'foo', 'FOO.', None, 'bar', 'foo'],
        'num_str': ['1', '2', 'x', None, '5', '2'],
        'date_str': ['2024-01-05 10:30', '2023-12-31 23:00', None, 'not a date',
                     '2025-06-15 08:00', '2023-12-31 23:00'],
        'dt': pd.to_datetime(['2024-01-05 10:30', '2023-12-31 23:00', None,
                              '2022-02-02 02:02', '2025-06-15 08:00', '2023-12-31 23:00']),
        'empty': [np.nan] * 6,
    })


def make_dup_df() -> pd.DataFrame:
    df = make_df()
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)  # exact duplicate row (incl. NaNs)
    df.loc[len(df)] = [np.nan] * df.shape[1]               # fully empty row
    return df


def make_wide_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 30
    base = rng.normal(size=n)
    data = {'target': base * 3 + rng.normal(scale=0.1, size=n)}
    for i in range(12):  # > 10 other numeric columns -> exercises correlation-based selection
        data[f'f{i}'] = base * (i % 4 - 1.5) + rng.normal(scale=0.5 + i, size=n)
    data['label'] = ['a', 'b', None] * 10
    df = pd.DataFrame(data)
    df.loc[[2, 7, 11, 19], 'target'] = np.nan
    df.loc[[3, 8], 'f0'] = np.nan
    return df


def run_script(recipe, df, rules=None):
    code = generate_pipeline_code(recipe, rules)
    ns = {}
    exec(code, ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ns['clean_data'](df.copy())
    return out, ns, [str(x.message) for x in w]


def assert_parity(recipe, df):
    raw = df.copy()
    engine_out, msgs = apply_recipe(downcast_dtypes(df.copy()), recipe)
    script_out, _, _ = run_script(recipe, raw)
    assert_frame_equal(engine_out, script_out, check_dtype=True)
    assert_frame_equal(df, raw)  # exported script must not mutate the caller's frame
    return engine_out, msgs


# --------------------------------------------------------------------------- transform parity

STEPS = {
    # drop_column / drop_nulls
    'drop_column': {'action': 'drop_column', 'column': 'text'},
    'drop_nulls': {'action': 'drop_nulls', 'column': 'float_col'},
    # fill_null sub-modes
    'fill_mean': {'action': 'fill_null', 'column': 'float_col', 'value': 'mean'},
    'fill_median': {'action': 'fill_null', 'column': 'float_col', 'value': 'median'},
    'fill_mean_numeric_strings': {'action': 'fill_null', 'column': 'num_str', 'value': 'mean'},
    'fill_median_numeric_strings': {'action': 'fill_null', 'column': 'num_str', 'value': 'median'},
    'fill_mean_non_numeric': {'action': 'fill_null', 'column': 'text', 'value': 'mean'},
    'fill_mode': {'action': 'fill_null', 'column': 'float_col', 'value': 'mode'},
    'fill_mode_text': {'action': 'fill_null', 'column': 'text', 'value': 'mode'},
    'fill_mode_all_null': {'action': 'fill_null', 'column': 'empty', 'value': 'mode'},
    'fill_const_num': {'action': 'fill_null', 'column': 'float_col', 'value': 0},
    'fill_const_str': {'action': 'fill_null', 'column': 'text', 'value': 'missing'},
    'fill_knn_non_numeric': {'action': 'fill_null', 'column': 'text', 'value': 'knn'},
    'fill_iterative_non_numeric': {'action': 'fill_null', 'column': 'text', 'value': 'iterative'},
    'fill_knn_small': {'action': 'fill_null', 'column': 'float_col', 'value': 'knn'},
    # cap_range
    'cap_float': {'action': 'cap_range', 'column': 'float_col', 'min': 0.0, 'max': 10.0},
    'cap_int': {'action': 'cap_range', 'column': 'int_col', 'min': 0, 'max': 10},
    # cast_type
    'cast_float': {'action': 'cast_type', 'column': 'num_str', 'dtype': 'float64'},
    'cast_int': {'action': 'cast_type', 'column': 'num_str', 'dtype': 'int'},
    'cast_string': {'action': 'cast_type', 'column': 'int_col', 'dtype': 'string'},
    'cast_object': {'action': 'cast_type', 'column': 'float_col', 'dtype': 'object'},
    'cast_datetime': {'action': 'cast_type', 'column': 'date_str', 'dtype': 'datetime64[ns]'},
    # drop_violated, one per rule type
    'dv_null': {'action': 'drop_violated', 'rule': {'type': 'Null Check', 'col': 'text'}},
    'dv_range_float': {'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': 'float_col', 'min': 1.0, 'max': 50.0}},
    'dv_range_strings': {'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': 'num_str', 'min': 1, 'max': 4}},
    'dv_range_bad_bounds': {'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': 'float_col', 'min': None, 'max': 5}},
    'dv_rel_feature': {'action': 'drop_violated', 'rule': {'type': 'Relational Check', 'col_a': 'neg_float', 'op': '<', 'col_b': 'float_col', 'target_type': 'Feature'}},
    'dv_rel_value': {'action': 'drop_violated', 'rule': {'type': 'Relational Check', 'col_a': 'int_col', 'op': '>=', 'value': 2, 'target_type': 'Value'}},
    'dv_rel_type_error': {'action': 'drop_violated', 'rule': {'type': 'Relational Check', 'col_a': 'text', 'op': '>', 'value': 5, 'target_type': 'Value'}},
    'dv_custom': {'action': 'drop_violated', 'rule': {'type': 'Custom Expression', 'query': 'int_col > 2'}},
    'dv_info': {'action': 'drop_violated', 'rule': {'type': 'Informational', 'desc': 'x'}},
    # replace
    'replace_col': {'action': 'replace', 'column': 'text', 'find': 'banana', 'replace': 'BANANA'},
    'replace_regex': {'action': 'replace', 'column': 'text', 'find': r'[aA]pple', 'replace': 'pear', 'regex': True},
    'replace_all': {'action': 'replace', 'column': 'All', 'find': 'foo', 'replace': 'baz'},
    # strip_whitespace
    'strip_col': {'action': 'strip_whitespace', 'column': 'text'},
    'strip_all': {'action': 'strip_whitespace', 'column': 'All'},
    # normalize_text modes
    'norm_lower': {'action': 'normalize_text', 'column': 'text', 'value': 'lowercase'},
    'norm_upper': {'action': 'normalize_text', 'column': 'text', 'value': 'uppercase'},
    'norm_title': {'action': 'normalize_text', 'column': 'text', 'value': 'titlecase'},
    'norm_punct': {'action': 'normalize_text', 'column': 'text', 'value': 'remove_punctuation'},
    'norm_fuzzy': {'action': 'normalize_text', 'column': 'Mixed Case-Col', 'value': 'fuzzy_dedupe'},
    'norm_lower_all': {'action': 'normalize_text', 'column': 'All', 'value': 'lowercase'},
    'norm_punct_all': {'action': 'normalize_text', 'column': 'All', 'value': 'remove_punctuation'},
    'norm_fuzzy_all': {'action': 'normalize_text', 'column': 'All', 'value': 'fuzzy_dedupe'},
    'norm_numeric_col_skipped': {'action': 'normalize_text', 'column': 'float_col', 'value': 'remove_punctuation'},
    'norm_unknown_method': {'action': 'normalize_text', 'column': 'text', 'value': 'reverse'},
    # log_transform
    'log_numeric': {'action': 'log_transform', 'column': 'neg_float'},
    'log_int': {'action': 'log_transform', 'column': 'int_col'},
    'log_non_numeric': {'action': 'log_transform', 'column': 'text'},
    # rename / reorder
    'rename': {'action': 'rename_column', 'column': 'text', 'value': "it's text"},
    'reorder': {'action': 'reorder_columns', 'value': ['text', 'missing_col', 'int_col']},
    # extract_datetime components
    **{f'dt_{comp}_{src}': {'action': 'extract_datetime', 'column': src, 'new_column': f'{src}_{comp}', 'component': comp}
       for comp in ['year', 'month', 'day', 'day_of_week', 'hour'] for src in ['date_str', 'dt']},
    'dt_no_new_col': {'action': 'extract_datetime', 'column': 'dt', 'component': 'year'},
    'dt_unknown_comp': {'action': 'extract_datetime', 'column': 'dt', 'new_column': 'x', 'component': 'week'},
    'dt_with_stray_value': {'action': 'extract_datetime', 'column': 'dt', 'new_column': 'm', 'component': 'month', 'value': 'year'},
    # whole-frame ops
    'drop_duplicates': {'action': 'drop_duplicates'},
    'drop_empty_columns': {'action': 'drop_empty_columns'},
    'drop_empty_rows': {'action': 'drop_empty_rows'},
    # normalize_column_names modes
    **{f'ncn_{m}': {'action': 'normalize_column_names', 'value': m}
       for m in ['snake_case', 'lowercase', 'uppercase', 'remove_spaces']},
    'unknown_action': {'action': 'teleport_rows', 'column': 'text'},
}


def test_every_registered_transform_has_a_parity_case():
    covered = {s['action'] for s in STEPS.values()}
    assert set(TRANSFORM_REGISTRY) <= covered


@pytest.mark.parametrize("name", list(STEPS))
def test_single_step_parity(name):
    df = make_dup_df() if name in ('drop_duplicates', 'drop_empty_rows', 'drop_empty_columns') else make_df()
    assert_parity([STEPS[name]], df)


@pytest.mark.parametrize("mode", ['knn', 'iterative'])
def test_advanced_imputation_parity_wide(mode):
    df = make_wide_df()
    out, msgs = assert_parity([{'action': 'fill_null', 'column': 'target', 'value': mode}], df)
    assert out['target'].notnull().all()
    assert not any('Error' in m for m in msgs)


def test_multi_step_recipe_parity():
    recipe = [
        {'action': 'strip_whitespace', 'column': 'All'},
        {'action': 'normalize_text', 'column': 'text', 'value': 'lowercase'},
        {'action': 'fill_null', 'column': 'float_col', 'value': 'median'},
        {'action': 'cap_range', 'column': 'float_col', 'min': 0.0, 'max': 10.0},
        {'action': 'extract_datetime', 'column': 'date_str', 'new_column': 'yr', 'component': 'year'},
        {'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': 'num_str', 'min': 0, 'max': 3}},
        {'action': 'drop_empty_columns'},
        {'action': 'normalize_column_names', 'value': 'snake_case'},
        {'action': 'drop_duplicates'},
    ]
    assert_parity(recipe, make_dup_df())


def test_downcast_applied_in_export():
    out, _, _ = run_script([], make_df())
    assert out['int_col'].dtype == np.int8
    assert out['float_col'].dtype == np.float32


def test_unknown_action_is_flagged():
    code = generate_pipeline_code([STEPS['unknown_action']])
    assert "WARNING: Unknown action 'teleport_rows'" in code
    _, _, warns = run_script([STEPS['unknown_action']], make_df())
    assert any('teleport_rows' in w for w in warns)
    _, msgs = apply_recipe(make_df(), [STEPS['unknown_action']])
    assert any('teleport_rows' in m for m in msgs)


def test_fuzzy_dedupe_skips_high_cardinality():
    df = pd.DataFrame({'s': [f'value_{i}' for i in range(1001)]})
    step = {'action': 'normalize_text', 'column': 's', 'value': 'fuzzy_dedupe'}
    out, msgs = assert_parity([step], df)
    assert any('too many unique values' in m for m in msgs)
    _, _, warns = run_script([step], df)
    assert any('too many unique values' in w for w in warns)


def test_notebook_includes_helper_and_runs():
    import json
    recipe = [STEPS['norm_lower'], STEPS['fill_mean']]
    nb = json.loads(generate_notebook_code(recipe, []))
    ns = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            exec("".join(cell['source']), ns)
    engine_out, _ = apply_recipe(downcast_dtypes(make_df()), recipe)
    assert_frame_equal(engine_out, ns['clean_data'](make_df()))


# --------------------------------------------------------------------------- null handling (C3)

@pytest.mark.parametrize("method", ['lowercase', 'uppercase', 'titlecase', 'remove_punctuation'])
def test_normalize_text_preserves_nan(method):
    df = make_df()
    step = {'action': 'normalize_text', 'column': 'text', 'value': method}
    out, _ = assert_parity([step], df)
    assert out['text'].isnull().sum() == df['text'].isnull().sum()
    assert 'nan' not in set(out['text'].dropna().str.lower())


def test_normalize_text_preserves_pd_na_in_string_dtype():
    df = pd.DataFrame({'s': pd.array(['A b', None, 'C!'], dtype='string')})
    out, _ = assert_parity([{'action': 'normalize_text', 'column': 's', 'value': 'lowercase'}], df)
    assert out['s'].dtype == 'string'
    assert out['s'].isna().tolist() == [False, True, False]


def test_range_check_ignores_nulls_but_flags_unparseable():
    df = pd.DataFrame({'v': [5, np.nan, 50, None], 's': ['5', None, 'abc', '1']})
    assert evaluate_rule(df, {'type': 'Range Check', 'col': 'v', 'min': 0, 'max': 10}).tolist() == [False, False, True, False]
    assert evaluate_rule(df, {'type': 'Range Check', 'col': 's', 'min': 0, 'max': 10}).tolist() == [False, False, True, False]


# --------------------------------------------------------------------------- validate_data parity

RULES = [
    {'type': 'Null Check', 'col': 'float_col', 'desc': 'null float'},
    {'type': 'Null Check', 'col': 'text', 'desc': 'null text'},
    {'type': 'Range Check', 'col': 'float_col', 'min': 1.0, 'max': 50.0, 'desc': 'range float'},
    {'type': 'Range Check', 'col': 'int_col', 'min': 0, 'max': 3, 'desc': 'range int'},
    {'type': 'Range Check', 'col': 'num_str', 'min': 1, 'max': 4, 'desc': 'range numeric strings'},
    {'type': 'Range Check', 'col': 'float_col', 'min': float('nan'), 'max': 5.0, 'desc': 'range nan bound'},
    {'type': 'Relational Check', 'col_a': 'neg_float', 'op': '<', 'col_b': 'float_col', 'target_type': 'Feature', 'desc': 'rel feature'},
    {'type': 'Relational Check', 'col_a': 'int_col', 'op': '!=', 'value': 2, 'target_type': 'Value', 'desc': 'rel value'},
    {'type': 'Relational Check', 'col_a': 'text', 'op': '<=', 'value': 3, 'target_type': 'Value', 'desc': 'rel type error'},
    {'type': 'Custom Expression', 'query': 'int_col > 2 or neg_float < 0', 'desc': 'custom'},
    {'type': 'Unknown Type', 'col': 'int_col', 'desc': 'unknown'},
]


@pytest.mark.parametrize("rule", RULES, ids=[r['desc'] for r in RULES])
def test_validate_data_matches_evaluate_rule(rule):
    df = make_df()
    expected = int(evaluate_rule(df, rule).sum())
    code = generate_pipeline_code([], [rule])
    ns = {}
    exec(code, ns)
    report = ns['validate_data'](df)
    got = report.get(rule['desc'], 0)
    assert got == expected, f"{rule['desc']}: script={got!r} engine={expected}"
