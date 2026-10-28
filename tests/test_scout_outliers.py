import numpy as np
import pandas as pd

import scout


def _range_props(df, col):
    return [p for p in scout.generate_proposals(df, set()) if p["column"] == col and p["type"] == "Range Check"]


def test_binary_column_has_no_outlier_proposal():
    df = pd.DataFrame({"y": [0] * 95 + [1] * 5})
    assert _range_props(df, "y") == []


def test_zero_inflated_column_has_no_outlier_proposal():
    # Q1 == Q3 == 0, so IQR bounds collapse to [0, 0] and every non-zero value would be flagged.
    df = pd.DataFrame({"spend": [0] * 90 + [3, 7, 12, 40, 55, 8, 2, 9, 100, 4]})
    assert _range_props(df, "spend") == []


def test_real_outlier_still_flagged():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": np.append(rng.normal(50, 5, 99), 500.0)})
    props = _range_props(df, "x")
    assert len(props) == 1
    assert props[0]["rule_data"]["max"] < 500
