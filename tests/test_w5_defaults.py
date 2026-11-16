"""§5 H defaults: A25 (Unicode snake_case, explicit all-columns flag) and I13 (30-day cached-upload prune)."""
import os
import time
import warnings

import pandas as pd
import pytest

from codegen import generate_pipeline_code
from engine import apply_recipe
from lumi_state import session_io


def _both(steps, df):
    out, msgs = apply_recipe(df.copy(), steps)
    ns = {}
    exec(generate_pipeline_code(steps), ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = ns["clean_data"](df.copy())
    return out, exp


# ---- A25: snake_case keeps Unicode letters ----
def test_snake_case_keeps_unicode_letters_engine_and_export():
    df = pd.DataFrame({"Café Name": [1], "naïve-Value": [2], "Größe": [3], "First Name": [4]})
    out, exp = _both([{"action": "normalize_column_names", "value": "snake_case"}], df)
    expected = ["café_name", "naïve_value", "größe", "first_name"]
    assert list(out.columns) == expected
    assert list(exp.columns) == expected


# ---- A25: a real column named "All" ----
def _all_df():
    return pd.DataFrame({"All": [" a "], "Other": [" b "]})


def test_real_column_named_all_is_targeted_alone():
    step = {"action": "strip_whitespace", "column": "All", "all_columns": False}
    out, exp = _both([step], _all_df())
    for frame in (out, exp):
        assert frame["All"].tolist() == ["a"]
        assert frame["Other"].tolist() == [" b "]


@pytest.mark.parametrize("step", [
    {"action": "strip_whitespace", "column": "All", "all_columns": True},
    {"action": "strip_whitespace", "column": "All"},  # legacy saved step: sentinel still means every column
])
def test_all_columns_flag_and_legacy_sentinel(step):
    out, exp = _both([step], _all_df())
    for frame in (out, exp):
        assert frame["All"].tolist() == ["a"] and frame["Other"].tolist() == ["b"]


def test_replace_and_normalize_respect_flag():
    df = pd.DataFrame({"All": ["x"], "Other": ["x"]})
    steps = [{"action": "replace", "column": "All", "all_columns": False, "find": "x", "replace": "y", "regex": False},
             {"action": "normalize_text", "column": "All", "all_columns": False, "value": "uppercase"}]
    out, exp = _both(steps, df)
    for frame in (out, exp):
        assert frame["All"].tolist() == ["Y"] and frame["Other"].tolist() == ["x"]


def test_ui_target_fields():
    from views.transformations import ALL_COLUMNS, _column_label, _target_fields
    assert _target_fields(ALL_COLUMNS) == {"column": "All", "all_columns": True}
    assert _target_fields("All") == {"column": "All", "all_columns": False}
    assert _column_label(ALL_COLUMNS) == "All text columns" and _column_label("Other") == "Other"


# ---- I13: prune cached uploads unused for 30 days ----
def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_prune_removes_only_old_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path)
    old, new = tmp_path / "old.dat", tmp_path / "new.dat"
    for p in (old, new, tmp_path / "old.sheet", tmp_path / "orphan.sheet", tmp_path / "keep.json"):
        p.write_bytes(b"x")
    _age(old, 31)
    _age(tmp_path / "keep.json", 90)  # resume JSON is not part of this prune
    assert session_io.prune_cached_uploads(force=True) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.json", "new.dat"]


def test_prune_skipped_in_tests_without_force(monkeypatch, tmp_path):
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LUMI_TESTING", "1")
    monkeypatch.setattr(session_io, "_pruned_cache", False)
    old = tmp_path / "old.dat"
    old.write_bytes(b"x")
    _age(old, 60)
    assert session_io.prune_cached_uploads() == 0
    assert old.exists()


def test_opening_a_cached_upload_resets_its_age(monkeypatch, tmp_path):
    monkeypatch.setattr(session_io, "CACHE_DIR", tmp_path)
    dat = tmp_path / "sid.dat"
    dat.write_bytes(b"a,b\n1,2\n")
    _age(dat, 40)
    assert session_io.load_cached_file("sid", "f.csv") is not None
    assert session_io.prune_cached_uploads(force=True) == 0
    assert dat.exists()


def test_ui_strip_targets_real_all_column_or_every_column():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload("t.csv", b"All,Other\n a , b \n", "text/csv").run()
    at.selectbox(key="trans_type_select").select("Strip Whitespace").run()
    box = at.selectbox(key="strip_target_col")
    assert box.options[:3] == ["All text columns", "All", "Other"]
    next(b for b in at.button if b.key == "btn_strip").click().run()
    at.selectbox(key="strip_target_col").select("All").run()
    next(b for b in at.button if b.key == "btn_strip").click().run()
    steps = [(s["column"], s["all_columns"]) for s in at.session_state.cleaning_recipe]
    assert steps == [("All", True), ("All", False)]
