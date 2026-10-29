import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scout import generate_proposals


def types(df, col):
    return [p for p in generate_proposals(df, set()) if p['column'] == col]


def test_zip_codes_with_leading_zeros_not_cast():
    df = pd.DataFrame({'zip': ['01234', '02134', '10001', '90210', '00501', 'n/a'] * 5})
    assert not [p for p in types(df, 'zip') if p['type'] == 'Type Cast']


def test_fixed_width_codes_not_cast():
    df = pd.DataFrame({'code': ['12345', '67890', '54321', '11111', 'ABCDE'] * 6})
    assert not [p for p in types(df, 'code') if p['type'] == 'Type Cast']


def test_id_named_column_not_cast():
    df = pd.DataFrame({'user_id': ['1', '22', '333', '4', 'x'] * 6})
    assert not [p for p in types(df, 'user_id') if p['type'] == 'Type Cast']


def test_real_mixed_column_still_cast_and_reports_nan_count():
    df = pd.DataFrame({'amt': ['1.5', '22', '3', '4.25', 'n/a'] * 4})
    p = [p for p in types(df, 'amt') if p['type'] == 'Type Cast']
    assert p and '4' in p[0]['reason'] and 'NaN' in p[0]['reason']


def test_constant_with_nan_not_dropped():
    df = pd.DataFrame({'flag': ['Y', np.nan, 'Y', np.nan], 'x': [1, 2, 3, 4]})
    assert not [p for p in types(df, 'flag') if p['type'] == 'Constant Value']


def test_truly_constant_still_flagged():
    df = pd.DataFrame({'c': ['Y'] * 4, 'x': [1, 2, 3, 4]})
    assert [p for p in types(df, 'c') if p['type'] == 'Constant Value']


def test_single_row_not_constant():
    df = pd.DataFrame({'a': [1], 'b': ['x']})
    assert not [p for p in generate_proposals(df, set()) if p['type'] == 'Constant Value']
