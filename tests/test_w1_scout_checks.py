import os
import time

import numpy as np
import pandas as pd

import scout


def _props(df, col, typ=None):
    return [p for p in scout.generate_proposals(df, set())
            if p["column"] == col and (typ is None or p["type"] == typ)]


def test_whitespace_check_is_linear_on_long_cells():
    df = pd.DataFrame({"t": ["x" + " " * 60000 + "y", "ok"]})
    t = time.time()
    assert _props(df, "t", "Formatting Issue") == []
    assert time.time() - t < 2.0
    df2 = pd.DataFrame({"t": [" a", "b"]})
    assert len(_props(df2, "t", "Formatting Issue")) == 1


def test_constant_check_uses_full_frame():
    df = pd.DataFrame({"c": ["A"] * 15000, "n": range(15000)})
    outside = df.index.difference(df.sample(n=10000, random_state=42).index)[0]
    df.loc[outside, "c"] = "B"  # differs only outside the 10k sample
    assert _props(df, "c", "Constant Value") == []
    df["k"] = "same"
    assert len(_props(df, "k", "Constant Value")) == 1


def test_range_check_skips_id_like_columns():
    ids = list(range(1000, 1100)) + [900000]
    assert _props(pd.DataFrame({"customer_id": ids}), "customer_id", "Range Check") == []
    # Unique but spread-out integers (salary, population) are measurements, not IDs: still checked.
    sal = [41000, 52000, 47000, 39000, 61000, 45000, 50000, 48000, 990000, 43000]
    assert _props(pd.DataFrame({"salary": sal}), "salary", "Range Check") != []
    df = pd.DataFrame({"zip": [10, 10, 11, 12, 10, 11, 12, 10, 99999, 10, 11, 12]})
    assert _props(df, "zip", "Range Check") == []


def test_range_check_integer_bounds_floor_zero_and_round_outward():
    df = pd.DataFrame({"qty": [1, 2, 2, 3, 3, 3, 4, 4, 5, 2, 3, 3, 4, 2, 500]})
    p = _props(df, "qty", "Range Check")[0]["rule_data"]
    assert p["min"] == 0.0 and p["max"] == float(int(p["max"]))
    df = pd.DataFrame({"v": [-10, 10, 11, 12, 13, 14, 15, 10, 12, 13, 11, 500]})
    p = _props(df, "v", "Range Check")[0]["rule_data"]
    assert p["min"] == float(int(p["min"])) and p["min"] < 10 and p["max"] == float(int(p["max"]))


def test_thousands_separator_column_is_castable():
    df = pd.DataFrame({"pop": ["25,649", "1,234,567", "980", "12,000.50", None]})
    p = _props(df, "pop", "Type Cast")
    assert len(p) == 1
    assert "thousands" in p[0]["reason"].lower()
    assert "NaN" not in p[0]["reason"]
    assert p[0]["rule_data"]["dtype"] == "float64"


def test_iso_datetime_text_gets_cast_proposal():
    df = pd.DataFrame({"when": ["2024-01-05 10:00:00", "2024-02-06 11:30:00", "2024-03-07T01:02:03"] * 10})
    p = _props(df, "when", "Type Cast")
    assert len(p) == 1 and p[0]["rule_data"]["dtype"] == "datetime64[ns]"
    words = pd.DataFrame({"m": ["May", "June", "July"] * 10})
    assert _props(words, "m", "Type Cast") == []


def test_tiny_missing_share_is_not_zero_percent():
    df = pd.DataFrame({"a": [None, None, None] + list(range(8997))})
    p = _props(df, "a", "Null Check")[0]
    assert "<0.1%" in p["reason"] and "0.0%" not in p["reason"]


def test_sampled_reasons_are_labelled():
    df = pd.DataFrame({"a": [None] * 100 + list(range(10100))})
    p = _props(df, "a", "Null Check")[0]
    assert p["reason"].endswith("(based on a 10,000-row sample)")
    small = pd.DataFrame({"a": [None, 1, 2]})
    assert "sample" not in _props(small, "a", "Null Check")[0]["reason"]


def test_initialize_state_reset_clears_sample_info():
    import streamlit as st
    from lumi_state.core import initialize_state
    st.session_state.clear() if hasattr(st.session_state, "clear") else None
    initialize_state()
    st.session_state.sample_info = {"shown": 1}
    initialize_state(from_reset=True)
    assert st.session_state.sample_info is None


def test_hash_for_path_tracks_content(tmp_path):
    from lumi_state.core import _hash_file_or_buffer
    p = tmp_path / "f.csv"
    p.write_text("a\n1\n")
    h1 = _hash_file_or_buffer(str(p))
    p.write_text("a\n2\n")
    assert _hash_file_or_buffer(str(p)) != h1
