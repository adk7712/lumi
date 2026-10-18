import os
import numpy as np
import pandas as pd

os.environ.setdefault("LUMI_TESTING", "1")

from ui_utils import build_zscore_boxplot, zscore_candidate_columns, MAX_BOX_COLUMNS


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(10, 3, 200),
        "b": rng.normal(-5, 50, 200),
        "const": 7.0,
        "nulls": np.nan,
        "txt": ["x"] * 200,
    })


def test_zscore_columns_and_skips():
    df = _df()
    assert zscore_candidate_columns(df) == ["a", "b"]
    fig, sampled = build_zscore_boxplot(df)
    assert not sampled
    names = {t.x[0] for t in fig.data if len(t.x)}
    assert names == {"a", "b"}


def test_zscore_math():
    df = _df()
    fig, _ = build_zscore_boxplot(df, columns=["a"])
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in fig.data])
    assert abs(ys.mean()) < 1e-9
    assert abs(ys.std(ddof=1) - 1) < 1e-9


def test_none_when_nothing_plottable_and_cap():
    assert build_zscore_boxplot(pd.DataFrame({"c": [1, 1, 1]}))[0] is None
    wide = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 20)),
                        columns=[f"c{i}" for i in range(20)])
    fig, _ = build_zscore_boxplot(wide)
    assert len(fig.data) == MAX_BOX_COLUMNS


def test_insights_stub_removed():
    import views
    import views.insights as ins
    assert not hasattr(views, "render_insights_tab")
    assert not hasattr(ins, "render_insights_tab")
    root = os.path.dirname(os.path.dirname(__file__))
    assert "render_insights_tab" not in open(os.path.join(root, "app.py")).read()
    assert hasattr(ins, "render_correlation_panel") and hasattr(ins, "render_outlier_panel")
