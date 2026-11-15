"""Fuzzy dedupe keeps distinct IDs apart and picks the most frequent spelling (A38)."""
import pandas as pd

from codegen import generate_pipeline_code
from engine import apply_recipe

STEP = {"action": "normalize_text", "column": "c", "value": "fuzzy_dedupe"}


def _both(values):
    df = pd.DataFrame({"c": values})
    engine_out, _ = apply_recipe(df.copy(), [STEP])
    ns = {}
    exec(generate_pipeline_code([STEP]), ns)
    export_out = ns["clean_data"](df.copy())
    pd.testing.assert_series_equal(engine_out["c"], export_out["c"])
    return engine_out["c"].tolist()


def test_ids_with_different_numbers_are_not_merged():
    vals = ["Product 10", "Product 11", "Product 12", "Order A1001", "Order A1002"]
    assert _both(vals) == vals


def test_most_frequent_spelling_is_canonical():
    out = _both(["Aple", "Apple", "Apple", "Apple", None])
    assert out[:4] == ["Apple"] * 4 and pd.isna(out[4])


def test_typos_still_merged():
    out = _both(["New York", "New York", "new york", "New Yrok"])
    assert set(out) == {"New York"}
