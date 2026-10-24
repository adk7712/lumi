"""Custom Expression safety (query_guard) and load_data content-based cache keys."""
import io

import numpy as np
import pandas as pd
import pytest

from codegen import generate_pipeline_code
from engine import apply_recipe
from query_guard import UnsafeQueryError, validate_query_expression
from rule_utils import evaluate_rule
from lumi_state.core import _hash_file_or_buffer, CachedFileWrapper

SAFE = [
    "age > 100 or age < 0",
    "city != 'NYC'",
    "years > 100 or `years` < 0 or page == 'age' or page > 5",
    "`user age` > 1",
    "not (a == 1)",
    "a in [1, 2, 3]",
    "a + b > 10",
    "-a > 0",
    "name.isnull()",
    "name.notnull() and a > 1",
    "name.str.contains('x', case=False, na=False)",
    "name.str.startswith('A')",
]
UNSAFE = [
    "@pd.io.common.os.system('echo hi')",
    "a > @x",
    "__import__('os').system('ls')",
    "a.__class__ > 1",
    "`a` > 1 and __x__ == 1",
    "a > 1 and import_ == 2 or ().__class__",
    "name.str.contains(open('f').read())",
    "name.upper() == 'A'",
    "a.sum() > 1",
    "a ** 99999999 > 1",
    "'x' * 999999999 == name",
    "[i for i in a]",
    "lambda: 1",
    "a > 1; b",
    "a > `unbalanced",
    "",
    "a > 1" + " and a > 1" * 100,
]


@pytest.mark.parametrize("expr", SAFE)
def test_safe_expressions_pass(expr):
    assert validate_query_expression(expr) == expr


@pytest.mark.parametrize("expr", UNSAFE)
def test_unsafe_expressions_rejected(expr):
    with pytest.raises(UnsafeQueryError):
        validate_query_expression(expr)


def _df():
    return pd.DataFrame({'a': [1, 2, 3, 4], 'name': ['Ax', None, 'bx', 'C']})


def test_engine_flags_and_rejects():
    df = _df()
    assert evaluate_rule(df, {'type': 'Custom Expression', 'query': 'a > 2 or name.isnull()'}).tolist() == [False, True, True, True]
    assert evaluate_rule(df, {'type': 'Custom Expression', 'query': "name.str.contains('x', case=False, na=False)"}).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        evaluate_rule(df, {'type': 'Custom Expression', 'query': "@df"})


def _run(recipe=(), rules=None, df=None):
    ns = {}
    exec(generate_pipeline_code(list(recipe), rules), ns)
    return ns


@pytest.mark.parametrize("expr", SAFE[:0] + ["a > 2 or name.isnull()", "name.str.startswith('A', na=False)", "a in [1, 2]"])
def test_script_parity_safe(expr):
    df = _df()
    rule = {'type': 'Custom Expression', 'query': expr, 'desc': 'r'}
    expected = int(evaluate_rule(df, rule).sum())
    got = _run(rules=[rule])['validate_data'](df).get('r', 0)
    assert got == expected


@pytest.mark.parametrize("expr", ["@df", "__import__('os').getcwd() == 'x'", "a.sum() > 1"])
def test_script_parity_rejected(expr):
    df = _df()
    rule = {'type': 'Custom Expression', 'query': expr, 'desc': 'r'}
    ns = _run(rules=[rule])
    # validate_data reports the rejection instead of executing it
    assert str(ns['validate_data'](df)['r']).startswith("Error")
    # clean_data skips the drop_violated step with a warning (engine does the same, with a message)
    recipe = [{'action': 'drop_violated', 'rule': rule}]
    out_engine, msgs = apply_recipe(df.copy(), recipe)
    assert any('Error' in m for m in msgs)
    ns = _run(recipe)
    with pytest.warns(UserWarning, match="Skipped drop_violated"):
        out_script = ns['clean_data'](df.copy())
    pd.testing.assert_frame_equal(out_engine.reset_index(drop=True), out_script.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(out_script.reset_index(drop=True), df, check_dtype=False)


def test_script_embeds_validator_only_when_needed():
    assert "UnsafeQueryError" not in generate_pipeline_code([], [{'type': 'Null Check', 'col': 'a', 'desc': 'x'}])
    code = generate_pipeline_code([], [{'type': 'Custom Expression', 'query': 'a > 1', 'desc': 'x'}])
    assert "validate_query_expression" in code and "query_guard" not in code.replace("# ", "")


# ---------------------------------------------------------------- load_data cache key

def test_cache_key_is_content_based():
    a1 = CachedFileWrapper(b"x,y\n1,2\n", "data.csv")
    a2 = CachedFileWrapper(b"x,y\n1,2\n", "data.csv")   # same content, different object
    b = CachedFileWrapper(b"x,y\n9,9\n", "data.csv")    # same name, different content
    assert _hash_file_or_buffer(a1) == _hash_file_or_buffer(a2)
    assert _hash_file_or_buffer(a1) != _hash_file_or_buffer(b)
    assert _hash_file_or_buffer(a1) != _hash_file_or_buffer(CachedFileWrapper(b"x,y\n1,2\n", "other.csv"))


def test_cache_key_does_not_depend_on_id_and_restores_position():
    class Reusable(io.BytesIO):
        name = "f.csv"
    keys = set()
    for payload in (b"a\n1\n", b"a\n2\n"):
        bio = Reusable(payload)
        bio.seek(2)
        keys.add(_hash_file_or_buffer(bio))
        assert bio.tell() == 2
        del bio  # id() may be recycled by the next iteration
    assert len(keys) == 2
    w = CachedFileWrapper(b"a\n1\n", "f.csv")
    w.seek(3)
    _hash_file_or_buffer(w)
    assert w.tell() == 3


def test_deeply_nested_query_raises_unsafe_not_recursion():
    import pandas as pd
    from query_guard import safe_query_index, UnsafeQueryError
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(UnsafeQueryError):
        safe_query_index(df, '-' * 490 + 'a > 1')
