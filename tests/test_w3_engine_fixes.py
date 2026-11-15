"""Wave-3 engine/export/report fixes (A21, A22, A25, A29, A30, I10, I11, Q14, Q16, Q18, N13)."""
import ast
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import codegen
from codegen import _lit, generate_notebook_code, generate_pipeline_code
from engine import apply_recipe
from report import generate_evidence_report


def _both(step, df):
    out, msgs = apply_recipe(df.copy(), [step])
    ns = {}
    exec(generate_pipeline_code([step]), ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp = ns["clean_data"](df.copy())
    return out, msgs, exp, [str(x.message) for x in w]


def _same_frame(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


# ---------------------------------------------------------------- A21 report
def test_report_rule_without_type_or_desc():
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [{"enabled": True}, {"type": "Null Check", "col": "a"}], [])
    assert "Rule Evaluation Summary" in out


def test_report_malformed_step_does_not_drop_lineage():
    recipe = [{"action": "drop_column", "column": "a"}, {"action": None, "rule": 5}, {"action": "drop_duplicates"}]
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [], recipe)
    assert "1. **Drop Column**" in out
    assert "3. **Drop Duplicates**" in out
    assert "2. " in out and "Could not describe step 2" in out or "2. **None**" in out


def test_report_malformed_step_line_when_describe_raises():
    class Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [], [{"action": "drop_duplicates"}, Boom()])
    assert "1. **Drop Duplicates**" in out
    assert "Could not describe step 2" in out


def test_report_escapes_pipes_and_backticks():
    rule = {"type": "Custom Expression", "desc": "a | b `x` c", "query": "a > 0"}
    out = generate_evidence_report(pd.DataFrame({"a": [1]}), [rule], [{"action": "rename_column", "column": "c`ol", "value": "n|ame"}])
    row = [ln for ln in out.splitlines() if ln.startswith("| Custom Expression")][0]
    assert "a \\| b" in row
    # unescaped pipes in the row: only the 5 column delimiters
    assert row.replace("\\|", "").count("|") == 5
    assert "`` c`ol ``" in out


# ---------------------------------------------------------------- A22 codegen
HOSTILE = ['we"ird', "we'ird", "a\"b'c", "back\\slash"]


def _fstring_violations(src):
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.JoinedStr):
            seg = ast.get_source_segment(src, node) or ""
            body = seg.lstrip("rRbBuUfF")
            q = body[:1]
            for fv in node.values:
                if isinstance(fv, ast.FormattedValue):
                    inner = ast.get_source_segment(src, fv.value) or ""
                    if q in inner or "\\" in inner:
                        bad.append(seg)
    return bad


@pytest.mark.parametrize("col", HOSTILE)
def test_export_compiles_for_old_pythons_with_hostile_columns(col):
    recipe = [
        {"action": "replace", "column": col, "find": "x", "replace": "y"},
        {"action": "replace", "column": "All", "find": "x", "replace": "y"},
        {"action": "strip_whitespace", "column": col},
        {"action": "normalize_text", "column": col, "value": "fuzzy_dedupe"},
        {"action": "normalize_text", "column": col, "value": "remove_punctuation"},
        {"action": "normalize_text", "column": col, "value": "titlecase"},
        {"action": "rename_column", "column": col, "value": "n'e\"w"},
        {"action": "log_transform", "column": col},
        {"action": "drop_violated", "rule": {"type": "Null Check", "col": col, "desc": col}},
    ]
    code = generate_pipeline_code(recipe, [{"type": "Null Check", "col": col, "desc": col}])
    ast.parse(code, feature_version=(3, 9))
    assert _fstring_violations(code) == []
    nb = generate_notebook_code(recipe, [])
    assert nb


def test_replace_single_column_hostile_name_runs():
    col = "we\"ird'x"
    df = pd.DataFrame({col: ["a", "b"]})
    out, _, exp, _ = _both({"action": "replace", "column": col, "find": "a", "replace": "z"}, df)
    _same_frame(out, exp)
    assert out[col].tolist() == ["z", "b"]


def test_fuzzy_export_without_thefuzz_warns_and_continues(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("thefuzz"):
            raise ImportError("no thefuzz")
        return real(name, *a, **k)

    for column in ("c", "All"):
        step = {"action": "normalize_text", "column": column, "value": "fuzzy_dedupe"}
        ns = {}
        exec(generate_pipeline_code([step, {"action": "drop_empty_rows"}]), ns)
        df = pd.DataFrame({"c": ["Jon Smith", "John Smith"], "d": ["a", "b"]})
        monkeypatch.setattr(builtins, "__import__", fake)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = ns["clean_data"](df.copy())
        finally:
            monkeypatch.setattr(builtins, "__import__", real)
        assert any("thefuzz" in str(x.message) and "not installed" in str(x.message) for x in w)
        _same_frame(res, df)


# ---------------------------------------------------------------- A25
def test_titlecase_apostrophes_engine_and_export():
    df = pd.DataFrame({"t": ["don't stop", "it’s OK", "o'brien and sons", None, "rock 'n' roll"]})
    out, msgs, exp, warns = _both({"action": "normalize_text", "column": "t", "value": "titlecase"}, df)
    assert out["t"].tolist()[:3] == ["Don't Stop", "It’s Ok", "O'Brien And Sons"]  # names keep the capital; only contraction suffixes are lowered
    assert out["t"].iloc[4] == "Rock 'N' Roll"
    assert not msgs and not warns
    _same_frame(out, exp)


def test_replace_numeric_column_string_find():
    df = pd.DataFrame({"n": [1, 5, 5, 7], "f": [1.5, 2.0, 2.0, 3.0]})
    out, msgs, exp, warns = _both({"action": "replace", "column": "n", "find": "5", "replace": "9"}, df)
    assert out["n"].tolist() == [1, 9, 9, 7] and not msgs and not warns
    _same_frame(out, exp)
    out, msgs, exp, warns = _both({"action": "replace", "column": "f", "find": "2", "replace": "2.5"}, df)
    assert out["f"].tolist() == [1.5, 2.5, 2.5, 3.0] and not msgs
    _same_frame(out, exp)


def test_replace_numeric_bad_replacement_warns_unchanged():
    df = pd.DataFrame({"n": [1, 5]})
    for rep in ("abc", "2.5"):
        out, msgs, exp, warns = _both({"action": "replace", "column": "n", "find": "5", "replace": rep}, df)
        assert out["n"].tolist() == [1, 5] and msgs and msgs[0].startswith("Warning")
        assert warns
        _same_frame(out, exp)


def test_cast_bool_to_datetime_leaves_column():
    df = pd.DataFrame({"b": [True, False, True]})
    out, msgs, exp, warns = _both({"action": "cast_type", "column": "b", "dtype": "datetime64[ns]"}, df)
    assert out["b"].dtype == bool and msgs and msgs[0].startswith("Warning")
    assert exp["b"].dtype == bool and warns
    _same_frame(out, exp)


# ---------------------------------------------------------------- A29
def test_lit_datetimes_and_scalars():
    import datetime
    assert _lit(pd.Timestamp("2020-01-02 03:04:05")) == "pd.Timestamp('2020-01-02T03:04:05')"
    assert _lit(datetime.datetime(2020, 1, 2)) == "pd.Timestamp('2020-01-02T00:00:00')"
    assert _lit(datetime.date(2020, 1, 2)) == "pd.Timestamp('2020-01-02T00:00:00')"
    assert _lit(np.int64(3)) == "3" and _lit(np.float32(1.5)) == "1.5" and _lit(np.bool_(True)) == "True"
    assert _lit(pd.NaT) == "pd.NaT"
    assert _lit([1, 2]) == "[1, 2]"  # plain literal containers round-trip (coordinator: softer than raising)
    with pytest.raises(ValueError):
        _lit([object()])
    with pytest.raises(ValueError):
        _lit(object())


def test_fill_null_timestamp_export_runs():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", None])})
    ts = pd.Timestamp("1999-12-31")
    out, msgs, exp, _ = _both({"action": "fill_null", "column": "d", "value": ts}, df)
    assert out["d"].iloc[1] == ts
    _same_frame(out, exp)


def test_fill_mode_uses_iloc_not_label():
    df = pd.DataFrame({"x": [3, 3, 7, None]}, index=[10, 11, 12, 13])
    out, msgs, exp, _ = _both({"action": "fill_null", "column": "x", "value": "mode"}, df)
    assert out["x"].iloc[3] == 3
    _same_frame(out, exp)
    assert ".iloc[0]" in generate_pipeline_code([{"action": "fill_null", "column": "x", "value": "mode"}])


# ---------------------------------------------------------------- A30
def test_getsource_failure_gives_clear_error(monkeypatch):
    monkeypatch.setattr(inspect, "getsource", lambda obj: (_ for _ in ()).throw(OSError("no source")))
    with pytest.raises(RuntimeError, match="source"):
        generate_pipeline_code([{"action": "normalize_text", "column": "a", "value": "fuzzy_dedupe"}])
    with pytest.raises(RuntimeError, match="source"):
        generate_pipeline_code([{"action": "normalize_column_names", "value": "snake_case"}])


# ---------------------------------------------------------------- I10 / I11
def test_normalize_names_non_str_headers_engine_and_export():
    df = pd.DataFrame([[1, 2, 3]], columns=[0, 1.5, None])
    for method in ("snake_case", "lowercase", "uppercase", "remove_spaces"):
        out, msgs, exp, warns = _both({"action": "normalize_column_names", "value": method}, df)
        assert not msgs and not warns, method
        assert all(isinstance(c, str) for c in out.columns)
        assert list(out.columns) == list(exp.columns)


def test_export_guards_normalize_names_and_drop_duplicates():
    for step in ({"action": "normalize_column_names", "value": "snake_case"}, {"action": "drop_duplicates"}):
        assert "except Exception" in generate_pipeline_code([step])


def test_drop_duplicates_unhashable_error_parity():
    df = pd.DataFrame({"a": [[1], [1], [2]], "b": [1, 1, 2]})
    out, msgs = apply_recipe(df.copy(), [{"action": "drop_duplicates"}])
    assert any("unexpected error" in m for m in msgs)
    ns = {}
    exec(generate_pipeline_code([{"action": "drop_duplicates"}]), ns)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp = ns["clean_data"](df.copy())
    assert any("drop_duplicates" in str(x.message) for x in w)
    assert len(out) == len(exp) == 3


def test_rename_onto_existing_refused():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out, msgs, exp, warns = _both({"action": "rename_column", "column": "a", "value": "b"}, df)
    assert list(out.columns) == ["a", "b"] and msgs and msgs[0].startswith("Warning")
    assert list(exp.columns) == ["a", "b"] and warns
    out, msgs, exp, warns = _both({"action": "rename_column", "column": "a", "value": "a"}, df)
    assert list(out.columns) == ["a", "b"] and not msgs and not warns
    out, msgs, exp, warns = _both({"action": "rename_column", "column": "a", "value": "c"}, df)
    assert list(out.columns) == ["c", "b"] == list(exp.columns)


# ---------------------------------------------------------------- Q14
def test_export_loader_example_handles_csv_and_excel():
    code = generate_pipeline_code([])
    assert "pd.read_excel" in code and 'sep=None, engine="python"' in code and ".xlsx" in code
    ast.parse(code)
    nb = generate_notebook_code([], [])
    assert "read_excel" in nb and "sep=None" in nb


# ---------------------------------------------------------------- Q16
def test_remove_punctuation_handles_backslash_and_bracket():
    df = pd.DataFrame({"t": ["a\\b]c[d^e-f", "x.y,z!"]})
    out, msgs, exp, warns = _both({"action": "normalize_text", "column": "t", "value": "remove_punctuation"}, df)
    assert out["t"].tolist() == ["abcdef", "xyz"] and not msgs
    _same_frame(out, exp)


# ---------------------------------------------------------------- Q18
def test_replace_no_match_warns_engine_and_export():
    df = pd.DataFrame({"a": ["x", "y"], "n": [1, 2]})
    for step in (
        {"action": "replace", "column": "a", "find": "zzz", "replace": "q"},
        {"action": "replace", "column": "a", "find": "zzz", "replace": "q", "regex": True},
        {"action": "replace", "column": "All", "find": "zzz", "replace": "q"},
    ):
        out, msgs, exp, warns = _both(step, df)
        assert msgs == [f"Warning: no values matched 'zzz' in '{step['column']}'; nothing was replaced."]
        assert any("no values matched 'zzz'" in w for w in warns)
        _same_frame(out, df)
        _same_frame(exp, df)
    # a match produces no message
    out, msgs, exp, warns = _both({"action": "replace", "column": "a", "find": "x", "replace": "q"}, df)
    assert not msgs and not warns
    out, msgs, exp, warns = _both({"action": "replace", "column": "All", "find": "x", "replace": "q"}, df)
    assert not msgs and not warns


# ---------------------------------------------------------------- N13
def test_drop_empty_columns_noop_message_in_export():
    df = pd.DataFrame({"a": [1, 2]})
    out, msgs, exp, warns = _both({"action": "drop_empty_columns"}, df)
    assert msgs == ["No empty columns found to drop."]
    assert "No empty columns found to drop." in warns
    _same_frame(out, exp)
    df = pd.DataFrame({"a": [1, 2], "e": [np.nan, np.nan]})
    out, msgs, exp, warns = _both({"action": "drop_empty_columns"}, df)
    assert list(out.columns) == list(exp.columns) == ["a"] and not warns
