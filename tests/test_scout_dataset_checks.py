import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scout import generate_proposals
from engine import apply_recipe


def _dirty():
    return pd.DataFrame({
        "Age-Years": [25, 30, 30, np.nan, 41],
        "Total Salary": [50000, 60000, 60000, np.nan, 72000],
        "city": ["a", "b", "b", None, "c"],
    })


def _clean():
    return pd.DataFrame({"age": [1, 2, 3], "total_salary": [10, 20, 30], "city": ["a", "b", "c"]})


def _ds(props):
    return {p["type"]: p for p in props if p["column"] == "__dataset__"}


def test_dirty_data_gets_dataset_proposals():
    ds = _ds(generate_proposals(_dirty(), set()))
    assert set(ds) == {"duplicates", "empty_rows", "column_names"}
    assert ds["duplicates"]["rule_data"] == {"action": "drop_duplicates"}
    assert ds["empty_rows"]["rule_data"] == {"action": "drop_empty_rows"}
    assert ds["column_names"]["rule_data"] == {"action": "normalize_column_names", "value": "snake_case"}
    assert "1 exact duplicate row " in ds["duplicates"]["reason"]


def test_clean_data_has_no_dataset_proposals():
    assert _ds(generate_proposals(_clean(), set())) == {}


def test_mixed_case_and_whitespace_headers_flagged_and_collision_noted():
    df = pd.DataFrame({" Name ": [1, 2], "name": [3, 4]})
    ds = _ds(generate_proposals(df, set()))
    assert "column_names" in ds and "collide" in ds["column_names"]["reason"]


def test_dismissal_hides_each_proposal():
    for key, t in [("__dataset__:duplicates", "duplicates"),
                   ("__dataset__:empty_rows", "empty_rows"),
                   ("__dataset__:column_names", "column_names")]:
        ds = _ds(generate_proposals(_dirty(), {key}))
        assert t not in ds and len(ds) == 2


def test_duplicates_counted_on_full_frame_beyond_sample():
    df = pd.DataFrame({"a": list(range(20000)) + [0] * 5})
    ds = _ds(generate_proposals(df, set()))
    assert "5 exact duplicate" in ds["duplicates"]["reason"]


def test_accepting_creates_expected_recipe_steps():
    df = _dirty()
    recipe = [p["rule_data"] for p in generate_proposals(df, set()) if p["column"] == "__dataset__"]
    assert {s["action"] for s in recipe} == {"drop_duplicates", "drop_empty_rows", "normalize_column_names"}
    out, _ = apply_recipe(df, recipe)
    assert list(out.columns) == ["ageyears", "total_salary", "city"] or list(out.columns) == ["age_years", "total_salary", "city"]
    assert len(out) == 3 and not out.duplicated().any()


def test_rulebook_accept_handler_adds_step_and_dismiss_key(monkeypatch):
    import types
    import views.rulebook as rb
    ss = types.SimpleNamespace(scanned_columns=set(), current_df=_dirty(), proposals=[])
    steps = []
    monkeypatch.setattr(rb.st, "session_state", ss, raising=False)
    # add_step itself syncs renamed columns (covered in tests/test_rename_sync.py)
    monkeypatch.setattr(rb, "add_step", lambda s: steps.append(s))
    props = [p for p in generate_proposals(_dirty(), set()) if p["column"] == "__dataset__"]
    ss.proposals = list(props)
    for p in props:
        rb._accept_proposal(p)
    assert {s["action"] for s in steps} == {"drop_duplicates", "drop_empty_rows", "normalize_column_names"}
    assert all(s["_source_proposal_col"] == "__dataset__" for s in steps)
    assert "__dataset__:duplicates" in ss.scanned_columns
    assert _ds(generate_proposals(_dirty(), ss.scanned_columns)) == {}
