"""snake_case column normalization: engine prediction, exported script parity, scout proposal."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from engine import apply_recipe, generate_pipeline_code
from engine_ops import predict_column_renames
from scout import _check_dataset_level

CASES = {
    'LotFrontage': 'lot_frontage',
    'MSSubClass': 'ms_sub_class',
    '1stFlrSF': 'col_1st_flr_sf',
    'Total Salary': 'total_salary',
    'Age-Years': 'age_years',
    'already_snake': 'already_snake',
    '  Name ': 'name',
    'HTTPResponse': 'http_response',
    'a__b': 'a_b',
    'Price($)': 'price',
    'x.y z': 'x_y_z',
}


@pytest.mark.parametrize('orig,expected', list(CASES.items()))
def test_predict_snake_case(orig, expected):
    assert predict_column_renames([orig], 'snake_case')[orig] == expected


def test_empty_name_falls_back():
    assert predict_column_renames(['  '], 'snake_case')['  '] == 'column_  '


def test_only_changed_skips_already_snake():
    out = predict_column_renames(['already_snake', 'LotFrontage'], 'snake_case', only_changed=True)
    assert out == {'LotFrontage': 'lot_frontage'}


@pytest.mark.parametrize('orig', list(CASES) + ['', '  ', 'Id-Field', 'a b.C'])
def test_exported_script_matches_engine(orig):
    df = pd.DataFrame({orig: [1, 2]})
    recipe = [{'action': 'normalize_column_names', 'value': 'snake_case'}]
    engine_out, _ = apply_recipe(df.copy(), recipe)
    ns = {}
    exec(generate_pipeline_code(recipe), ns)
    script_out = ns['clean_data'](df.copy())
    assert list(engine_out.columns) == list(script_out.columns)


def test_scout_column_names_proposal_uses_snake_case():
    df = pd.DataFrame({'LotFrontage': [1.0], 'lot_area': [2.0], '1stFlrSF': [3.0]})
    props = {p['type']: p for p in _check_dataset_level(df)}
    assert 'column_names' in props
    assert "'LotFrontage'" in props['column_names']['reason']
    assert "'lot_area'" not in props['column_names']['reason']
    assert props['column_names']['rule_data'] == {'action': 'normalize_column_names', 'value': 'snake_case'}
