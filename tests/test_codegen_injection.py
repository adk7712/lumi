"""Regression: user-controlled text must never break out of generated code (A1)."""
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from codegen import _comment, generate_pipeline_code, generate_notebook_code

HOSTILE = [
    'status == "closed"',
    "it's",
    'a != \'");print("PWN");("\'',
    '"); print("PWN") #',
    "'); print('PWN') #",
    "x\nprint('PWN')\n",
    "x\rprint('PWN')",
    "x\r\nprint('PWN')",
    "x print('PWN')",
    "x print('PWN')",
    "x\x0bprint('PWN')",
    "x\x0cprint('PWN')",
    "x\x1cprint('PWN')",
    "x\x85print('PWN')",
    '"""; print("PWN"); """',
    "'''; print('PWN'); '''",
    "{}",
    "{0} {x} {{}} {__import__('os')}",
    "%s %d %(a)s 100%",
    "back\\slash\\",
    "\\",
    "\\n\\'",
    "café 中文 \U0001f600",
    "nul\x00byte",
    "#",
    "{print('PWN')}",
    "f'{print(1)}'",
]


def _run(code, df):
    """compile + exec the generated code, return captured stdout."""
    compile(code, "<generated>", "exec")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = {}
        exec(code, ns)
        try:
            out = ns['clean_data'](df.copy())
        except Exception:
            out = df
        try:
            ns['validate_data'](out)
        except Exception:
            pass
    return buf.getvalue()


def _notebook_code(nb_json):
    nb = json.loads(nb_json)
    return "\n".join("".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def _check(recipe, rules, df):
    code = generate_pipeline_code(recipe, rules)
    assert "PWN" not in _run(code, df)
    nb = _notebook_code(generate_notebook_code(recipe, rules))
    assert "PWN" not in _run(nb, df)


def test_issue_repro_quote_in_query_compiles():
    rule = {'type': 'Custom Expression', 'query': 'status == "closed"', 'desc': 'Violates: status == "closed"'}
    df = pd.DataFrame({'status': ['open', 'closed', 'open']})
    code = generate_pipeline_code([{'action': 'drop_violated', 'rule': rule}], [rule])
    compile(code, "<g>", "exec")
    ns = {}
    exec(code, ns)
    assert ns['clean_data'](df)['status'].tolist() == ['open', 'open']
    compile(_notebook_code(generate_notebook_code([{'action': 'drop_violated', 'rule': rule}], [rule])), "<nb>", "exec")


def test_issue_repro_crafted_query_does_not_execute():
    q = 'a != \'");print("PWN");("\''
    rule = {'type': 'Custom Expression', 'query': q, 'desc': f'Violates: {q}'}
    _check([{'action': 'drop_violated', 'rule': rule}], [rule], pd.DataFrame({'a': ['x', 'y']}))


@pytest.mark.parametrize("h", HOSTILE)
def test_comment_stays_on_one_line(h):
    c = _comment(h)
    assert len(c.splitlines()) <= 1
    for ch in "\n\r\x0b\x0c\x1c\x1d\x1e\x85  \x00":
        assert ch not in c
    compile("x = 1  # " + c + "\n", "<c>", "exec")


@pytest.mark.parametrize("h", HOSTILE)
def test_rules_with_hostile_text(h):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    rules = [
        {'type': 'Null Check', 'col': 'a', 'desc': h},
        {'type': 'Range Check', 'col': 'a', 'min': 0, 'max': 2, 'desc': h},
        {'type': 'Relational Check', 'col_a': 'a', 'op': '<', 'target_type': 'Value', 'value': h, 'desc': h},
        {'type': 'Relational Check', 'col_a': h, 'op': '<', 'target_type': 'Feature', 'col_b': h, 'desc': h},
        {'type': 'Custom Expression', 'query': h, 'desc': h},
        {'type': 'Custom Expression', 'query': 'a > 1', 'desc': h},
        {'type': 'Informational', 'desc': h},
        {'type': h, 'desc': h},
        {'type': 'Null Check', 'col': h, 'desc': h},
    ]
    recipe = [{'action': 'drop_violated', 'rule': r} for r in rules]
    _check(recipe, rules, df)


@pytest.mark.parametrize("h", HOSTILE)
def test_steps_with_hostile_text(h):
    df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': ['x ', ' y', 'z'], h: ['p', 'q', 'r']})
    recipe = [
        {'action': 'drop_column', 'column': h},
        {'action': 'drop_nulls', 'column': h},
        {'action': 'fill_null', 'column': h, 'value': h},
        {'action': 'fill_null', 'column': 'a', 'value': h},
        {'action': 'fill_null', 'column': h, 'value': 'mean'},
        {'action': 'fill_null', 'column': h, 'value': 'knn'},
        {'action': 'fill_null', 'column': h, 'value': 'iterative'},
        {'action': 'cap_range', 'column': h, 'min': 0, 'max': 1},
        {'action': 'cast_type', 'column': h, 'dtype': 'datetime64[ns]'},
        {'action': 'cast_type', 'column': h, 'dtype': 'string'},
        {'action': 'replace', 'column': h, 'find': h, 'replace': h, 'regex': False},
        {'action': 'replace', 'column': 'All', 'find': h, 'replace': h},
        {'action': 'strip_whitespace', 'column': h},
        {'action': 'normalize_text', 'column': h, 'value': 'lowercase'},
        {'action': 'normalize_text', 'column': h, 'value': 'fuzzy_dedupe'},
        {'action': 'normalize_text', 'column': 'b', 'value': h},
        {'action': 'normalize_text', 'column': h, 'value': 'remove_punctuation'},
        {'action': 'log_transform', 'column': h},
        {'action': 'rename_column', 'column': h, 'value': h},
        {'action': 'rename_column', 'column': 'b', 'value': h},
        {'action': 'reorder_columns', 'value': [h, 'a']},
        {'action': 'extract_datetime', 'column': h, 'new_column': h, 'component': 'year'},
        {'action': 'extract_datetime', 'column': 'a', 'new_column': 'n', 'component': h},
        {'action': 'normalize_column_names', 'value': h},
        {'action': h},
        {'action': 'drop_duplicates'},
    ]
    _check(recipe, [], df)


def test_hostile_column_names_in_every_position_run():
    h = 'c"\nprint(\'PWN\')\n"'
    df = pd.DataFrame({h: [1, 2, 3], 'b': ['x', 'y', 'z']})
    recipe = [{'action': 'rename_column', 'column': h, 'value': h + "2"},
              {'action': 'drop_violated', 'rule': {'type': 'Range Check', 'col': h + "2", 'min': 0, 'max': 2, 'desc': h}}]
    rules = [{'type': 'Range Check', 'col': h, 'min': 0, 'max': 2, 'desc': h}]
    code = generate_pipeline_code(recipe, rules)
    out = _run(code, df)
    assert "PWN" not in out
