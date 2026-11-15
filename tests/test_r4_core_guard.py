"""R4: wide bounded repeats count as open-ended; Replace "All" shares one time budget."""
import time
import warnings

import pandas as pd
import pytest

import engine_ops
import query_guard as qg
from codegen import generate_pipeline_code

WIDE = [r"(a)+?(a){2,40}[ab]$", r"[^x]{1,30}(?:a){2,}(a|ab)$", r"[^x]{1,30}a[a-z]+\d\d{30}$"]


@pytest.mark.parametrize("pat", WIDE)
def test_contains_rejects_wide_bounded_ranges(pat):
    with pytest.raises(qg.UnsafeQueryError):
        qg.validate_query_expression(f"name.str.contains({pat!r})")


@pytest.mark.parametrize("pat", [r"^\d{3}-\d{4}$", r"[A-Z]{2}\d+", r"^\w+-\d+$"])
def test_contains_practical_still_allowed(pat):
    qg.validate_query_expression(f"name.str.contains({pat!r})")


def test_replace_practical_date_pattern_allowed():
    assert qg.unsafe_regex_reason(r"\d{1,2}/\d{1,2}/\d{2,4}") is None


def test_lazy_counts_like_greedy():
    assert qg.unsafe_regex_reason(r"a+?b+?c+?d+?") is not None  # 4 open-ended > 3


@pytest.mark.skipif(qg._regex is None, reason="regex package not installed")
def test_replace_all_shares_one_budget(monkeypatch):
    monkeypatch.setattr(qg, "REGEX_TIME_BUDGET", 0.6)
    monkeypatch.setattr(qg.regex_deadline, "__defaults__", (0.6,))
    pat = r"\w+a\w+a\w+!"
    df = pd.DataFrame({f"c{i}": ["a" * 2000 + "b"] * 5 for i in range(6)})
    step = {"action": "replace", "column": "All", "find": pat, "replace": "X", "regex": True}
    t = time.time()
    out, w = engine_ops._handle_replace(df.copy(), step)
    assert time.time() - t < 2.0  # 6 columns x 0.6 s would be 3.6 s
    assert len(w) == 6 and out.equals(df)
    ns = {}
    exec(generate_pipeline_code([step]), ns)
    t = time.time()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ns["clean_data"](df.copy())
    assert time.time() - t < 4.0  # the script keeps its own 2 s budget, shared; per column is 12 s
