import numpy as np
import pandas as pd

import scout


def _types(props, col):
    return {p["type"] for p in props if p["column"] == col}


def test_bool_column_no_crash_no_numeric_proposals():
    df = pd.DataFrame({"flag": [True, False, True, True, False] * 4})
    props = scout.generate_proposals(df, set())
    assert not (_types(props, "flag") & {"Range Check", "Distribution Warning"})


def test_nullable_boolean_with_na():
    df = pd.DataFrame({"flag": pd.array([True, None, False, True, False, None], dtype="boolean")})
    props = scout.generate_proposals(df, set())
    t = _types(props, "flag")
    assert "Null Check" in t
    assert not (t & {"Range Check", "Distribution Warning", "Constant Value"})


def test_constant_bool_flagged_only_when_constant():
    df = pd.DataFrame({"a": [True] * 10, "b": [True, False] * 5})
    props = scout.generate_proposals(df, set())
    assert "Constant Value" in _types(props, "a")
    assert "Constant Value" not in _types(props, "b")


def test_skewed_bool_no_crash():
    df = pd.DataFrame({"f": [True] + [False] * 99})
    props = scout.generate_proposals(df, set())
    assert not (_types(props, "f") & {"Range Check", "Distribution Warning"})


def test_mixed_frame_200_rows_numeric_unchanged():
    n = 200
    vals = list(np.arange(n - 1, dtype=float)) + [1e6]
    df = pd.DataFrame({
        "flag": [True, False] * (n // 2),
        "num": vals,
        "txt": [f"name{i}" for i in range(n)],
    })
    props = scout.generate_proposals(df, set())
    assert "Range Check" in _types(props, "num")
    assert "Range Check" not in _types(props, "flag")
    assert "High Cardinality" in _types(props, "txt")


def test_csv_true_false_text_roundtrip():
    import io
    df = pd.read_csv(io.StringIO("a,b\nTrue,1\nFalse,2\nTrue,3\n"))
    assert df["a"].dtype == bool
    scout.generate_proposals(df, set())
