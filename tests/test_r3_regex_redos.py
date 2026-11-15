"""ReDoS defence in depth: static guard, Replace runtime timeout, engine == export parity."""
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

import query_guard as qg
import engine_ops
from codegen import generate_pipeline_code

ROOT = Path(__file__).resolve().parent.parent

BAD = {
    "a?x30+a*30": "a?" * 30 + "a" * 30,
    "a*x20 b": "a*" * 20 + "b",
    ".*x20 b": ".*" * 20 + "b",
    "[ab]*x30 c": "[ab]*" * 30 + "c",
    "a{0,2}x30 b": "a{0,2}" * 30 + "b",
    "\\w*x50 !": r"\w*" * 50 + "!",
    "a+x20 b": "a+" * 20 + "b",
    "dotstar pair": r".*.*",
    "w pair": r"\w*\w*x",
    "a pair": r"a*a*b",
    "five variable": r"a?b?c?d?e?",
}
GOOD = [r"^\d{3}-\d{4}$", r"[A-Z]{2}\d+", r"^\w+@\w+\.\w+$", "(abc|def)", r"\s+$", r"^0\d+",
        r"[\w.+-]+@[\w-]+\.[\w.-]+", r"\d{1,2}/\d{1,2}/\d{2,4}", r"\d{3}"]


@pytest.mark.parametrize("pat", list(BAD.values()), ids=list(BAD))
def test_guard_rejects(pat):
    assert qg.unsafe_regex_reason(pat)


@pytest.mark.parametrize("pat", GOOD)
def test_guard_accepts_practical(pat):
    assert qg.unsafe_regex_reason(pat) is None


def test_fixed_counts_do_not_count():
    assert qg.unsafe_regex_reason(r"\d{3}\d{3}\d{3}\d{3}\d{3}\d{3}") is None


def test_contains_guard_and_literal_allowed():
    from query_guard import validate_query_expression, UnsafeQueryError
    with pytest.raises(qg.UnsafeQueryError):
        validate_query_expression(f"a.str.contains({('a?' * 30 + 'a' * 30)!r})")
    with pytest.raises(qg.UnsafeQueryError):
        validate_query_expression(f"a.str.contains({('a*' * 20 + 'b')!r}, case=False)")
    validate_query_expression(f"a.str.contains({('a*' * 20 + 'b')!r}, regex=False)")
    validate_query_expression(f"a.str.contains({('.*' * 20)!r}, False, 0, False, False)")


def _run(df, step):
    return engine_ops._handle_replace(df.copy(), step)


def test_replace_semantics_match_pandas():
    df = pd.DataFrame({"t": ["ab12", "x", None, "9-8"], "n": [1, 2, 3, 4]})
    for find, rep in [(r"(\d)(\d)", r"\2\1"), (r"(\d)", r"<\1>"), (r"\d$", "$"), (r"b", r"\g<0>b")]:
        out, w = _run(df, {"column": "t", "find": find, "replace": rep, "regex": True})
        exp = df["t"].replace(find, rep, regex=True)
        assert w == [] and out["t"].tolist() == exp.tolist()
    out, w = _run(df, {"column": "All", "find": r"\d", "replace": "#", "regex": True})
    assert out["t"].tolist() == ["ab##", "x", None, "#-#"] and out["n"].tolist() == [1, 2, 3, 4]


@pytest.mark.skipif(qg._regex is None, reason="regex package not installed")
def test_replace_times_out_and_leaves_column_unchanged():
    pat = r"\w+a\w+a\w+!"
    assert qg.unsafe_regex_reason(pat) is None
    df = pd.DataFrame({"t": ["a" * 2000 + "b"] * 5})
    t = time.time()
    out, w = _run(df, {"column": "t", "find": pat, "replace": "X", "regex": True})
    assert time.time() - t < 2.5
    assert out["t"].tolist() == df["t"].tolist()
    assert w and "skipped" in w[0]


def test_export_matches_engine_on_safe_replace():
    df = pd.DataFrame({"t": ["ab12", "x", None, "9-8"], "k": ["a1", "b2", "c3", "d4"]})
    for col in ("t", "All"):
        step = {"action": "replace", "column": col, "find": r"(\d)", "replace": r"<\1>", "regex": True}
        eng, _ = _run(df, step)
        ns = {}
        exec(generate_pipeline_code([step]), ns)
        exp = ns["clean_data"](df.copy())
        pd.testing.assert_frame_equal(eng, exp)


@pytest.mark.skipif(qg._regex is None, reason="regex package not installed")
def test_export_times_out_with_warning():
    step = {"action": "replace", "column": "t", "find": r"\w+a\w+a\w+!", "replace": "X", "regex": True}
    ns = {}
    exec(generate_pipeline_code([step]), ns)
    df = pd.DataFrame({"t": ["a" * 2000 + "b"] * 5})
    import warnings
    t = time.time()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = ns["clean_data"](df.copy())
    assert time.time() - t < 2.5 and out["t"].tolist() == df["t"].tolist()
    assert any("skipped" in str(r.message) for r in rec)


@pytest.mark.parametrize("expr", [
    "name.str.contains('\\\\w+a\\\\w+!')",
    "name.str.contains('.*a.*b')",
    "name.str.contains('^.*a.*a.*b')",
])
def test_contains_rejects_polynomial_patterns(expr):
    # df.query cannot be timed out: two open-ended repeats take ~6 s on a 2,000-char cell
    with pytest.raises(qg.UnsafeQueryError):
        qg.validate_query_expression(expr)


@pytest.mark.parametrize("expr", [
    "name.str.contains('^\\\\w+-\\\\d+$')",  # anchored: two open-ended repeats allowed
    "name.str.contains('[A-Z]{2}\\\\d+')",
    "name.str.contains('.*a.*b', regex=False)",
])
def test_contains_allows_practical_patterns(expr):
    assert qg.validate_query_expression(expr) == expr
