"""Chart caches are keyed on recipe content, not step count (A37)."""
import os
os.environ.setdefault("LUMI_TESTING", "1")

import numpy as np
import pandas as pd
import streamlit as st

import ui_utils


def _key(recipe):
    st.session_state["last_file_hash"] = "h1"
    st.session_state["cleaning_recipe"] = recipe
    return ui_utils.chart_state_key()


def test_same_step_count_different_steps_gives_different_key():
    a = _key([{"action": "drop_column", "column": "x"}])
    b = _key([{"action": "drop_column", "column": "y"}])
    assert a != b
    assert a == _key([{"action": "drop_column", "column": "x"}])


def test_undo_then_different_step_redraws_chart():
    ui_utils.plot_correlation_matrix.clear()
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 1))
    df1 = pd.DataFrame({"a": base[:, 0], "b": base[:, 0] * 2, "c": rng.normal(size=50)})
    df2 = df1.drop(columns="b")
    fig1 = ui_utils.plot_correlation_matrix(df1, (0.5, 1.0), _key([{"action": "drop_column", "column": "c"}]))
    fig2 = ui_utils.plot_correlation_matrix(df2, (0.5, 1.0), _key([{"action": "drop_column", "column": "b"}]))
    assert fig1 is not None
    assert fig2 is None  # no correlated pair left; the stale fig1 must not be returned
