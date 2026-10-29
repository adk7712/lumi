"""R3 load fixes: ragged rows, trailing-delimiter columns, non-tabular uploads, dev login placeholder."""
import io
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

import pandas as pd
import pytest

from lumi_state.core import CachedFileWrapper, load_data


def _ld(raw: bytes, name="t.csv", nrows=None):
    return load_data(CachedFileWrapper(raw, name), nrows=nrows)


def test_extra_fields_refused_with_row_number():
    df = _ld(b"a,b\n1,2,3\n4,5,6\n")
    err = df.attrs.get("lumi_load_error")
    assert err and "row 2 has 3 fields" in err and "header has 2" in err
    assert df.empty


def test_extra_fields_later_row_reported():
    err = _ld(b"a,b\n1,2\n3,4\n5,6,7\n").attrs.get("lumi_load_error")
    assert err and "row 4 has 3 fields" in err


def test_empty_extra_fields_load_silently():
    df = _ld(b"id,age,name\n1,,a,\n2,3,b,\n")
    assert "lumi_load_error" not in df.attrs
    assert list(df.columns) == ["id", "age", "name"]
    assert df["name"].tolist() == ["a", "b"]


def test_trailing_delimiter_in_header_drops_unnamed_column():
    df = _ld(b"id,age,name,\n1,2,a,\n3,4,b,\n")
    assert list(df.columns) == ["id", "age", "name"]
    assert len(df) == 2


def test_real_unnamed_column_with_data_is_kept():
    df = _ld(b"id,,name\n1,x,a\n")
    assert len(df.columns) == 3 and df.iloc[0, 1] == "x"


@pytest.mark.parametrize("raw", [
    b'{"a":1}', b'{\n  "a": 1\n}', b'[{"a": 1}]',
    b"<!DOCTYPE html><html><body>x</body></html>", b"<?xml version='1.0'?><r/>",
])
def test_non_tabular_text_rejected(raw):
    df = _ld(raw)
    assert df.attrs.get("lumi_load_error")


def test_bogus_xlsx_rejected():
    df = _ld(b"PK\x03\x04not really a workbook", name="t.xlsx")
    assert "valid Excel" in df.attrs.get("lumi_load_error", "")
    df = _ld(b"PK\x03\x04not really a workbook", name="t.csv")
    assert df.attrs.get("lumi_load_error")


def test_good_files_unchanged():
    d = _ld(b"a;b\n1,5;x\n2,5;y\n")
    assert list(d.columns) == ["a", "b"] and d["a"].tolist() == [1.5, 2.5]
    d = _ld(b"a\tb\n1\t2\n")
    assert d.shape == (1, 2)
    d = _ld(b"\xef\xbb\xbfa,b\n1,2\n")
    assert list(d.columns) == ["a", "b"]
    d = _ld(b'a,b\n"x\ny",2\n3,4\n')
    assert d.shape == (2, 2) and d.iloc[0, 0] == "x\ny"
    d = _ld(b"zip,n\n01234,1\n00501,2\n")
    assert d["zip"].tolist() == ["01234", "00501"]
    d = _ld(b"a,b\n1,2\n3\n")  # short rows are fine
    assert d.shape == (2, 2)
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
    d = _ld(buf.getvalue(), name="t.xlsx")
    assert d.shape == (2, 2) and "lumi_load_error" not in d.attrs


def test_dev_login_email_uses_placeholder():
    src = (Path(__file__).parent.parent / "ui_utils.py").read_text()
    line = [l for l in src.splitlines() if "auth_dev_email_input" in l][0]
    assert 'value="user@example.com"' not in line and 'placeholder="user@example.com"' in line
