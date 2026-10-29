"""Robust file loading: headers, delimiters, encodings, quoting; load_data == load_full_data."""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault("LUMI_TESTING", "1")

from streamlit.testing.v1 import AppTest

from lumi_state.core import CachedFileWrapper, _hash_file_or_buffer, load_data, read_tabular
import lumi_state.session_io as sio

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _ld(raw: bytes, name="t.csv", nrows=None):
    return load_data(CachedFileWrapper(raw, name), nrows=nrows)


def _xlsx(rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    b = io.BytesIO()
    wb.save(b)
    return b.getvalue()


def test_numeric_xlsx_headers_become_str():
    df = _ld(_xlsx([["Name", 2019, 2020], ["a", 1, 2]]), "n.xlsx")
    assert list(df.columns) == ["Name", "2019", "2020"]
    assert all(isinstance(c, str) for c in df.columns)


def test_duplicate_after_coercion_deduped():
    df, _ = read_tabular(_xlsx([[2019, "2019", "2019"], [1, 2, 3]]), "d.xlsx")
    assert len(set(df.columns)) == 3 and df.columns[0] == "2019"
    assert all(c.startswith("2019") for c in df.columns)


def test_numeric_xlsx_upload_does_not_crash():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.session_state["_is_testing"] = True
    at.file_uploader(key="welcome_uploader").upload(
        "n.xlsx", _xlsx([["Name", 2019, 2020], ["a", 1, 2], ["b", 3, 4]]),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet").run()
    assert not at.exception
    assert list(at.session_state.raw_data.columns) == ["Name", "2019", "2020"]


@pytest.mark.parametrize("sep,label", [(";", "semicolon"), ("\t", "tab"), ("|", "pipe")])
def test_delimiters_detected(sep, label):
    raw = sep.join(["id", "name", "age"]).encode() + b"\n" + b"\n".join(
        sep.join([str(i), f"n{i}", str(20 + i)]).encode() for i in range(10))
    df = _ld(raw)
    assert list(df.columns) == ["id", "name", "age"] and len(df) == 10
    assert any(label in n for n in df.attrs["lumi_notices"])


def test_tab_file_with_empty_fields():
    df = _ld(b"a\tb\tc\n1\t\t3\n4\t5\t\n")
    assert list(df.columns) == ["a", "b", "c"]
    assert df["b"].isna().sum() == 1 and df["c"].isna().sum() == 1


def test_comma_file_with_semicolons_stays_comma():
    raw = b"id,note\n" + b"\n".join(f"{i},a;b;c;d".encode() for i in range(10))
    df = _ld(raw)
    assert list(df.columns) == ["id", "note"]
    assert "lumi_notices" not in df.attrs


def test_single_column_file_unchanged():
    df = _ld(b"name\nalice\nbob\n")
    assert list(df.columns) == ["name"] and len(df) == 2


def test_decimal_comma_unambiguous_only():
    raw = b"id;price;code\n1;1,5;a,b\n2;2,25;c\n3;3;d\n"
    df = _ld(raw)
    assert list(df.columns) == ["id", "price", "code"]
    assert df["price"].tolist() == [1.5, 2.25, 3.0]
    assert df["code"].dtype == object  # mixed text stays text
    df2 = _ld(b"id;amt\n1;1.234,5\n2;2,5\n3;7\n")
    assert df2["amt"].dtype == object  # thousand separator: ambiguous, left as text


def test_cp1252_fallback_with_notice():
    raw = "name,city,price\nJosé,Peña,5€\n".encode("cp1252")
    df = _ld(raw)
    assert df.loc[0, "name"] == "José" and df.loc[0, "city"] == "Peña" and df.loc[0, "price"] == "5€"
    assert any("Windows-1252" in n for n in df.attrs["lumi_notices"])


def test_latin1_fallback():
    raw = b"name\n\xe9t\xe9 \x81\n"  # 0x81 undefined in cp1252
    df = _ld(raw)
    assert df.loc[0, "name"] == "\xe9t\xe9 \x81"
    assert any("Latin-1" in n for n in df.attrs["lumi_notices"])


def test_utf16_le_bom():
    raw = "a,b\nÉ,2\n".encode("utf-16")  # includes BOM
    df = _ld(raw)
    assert list(df.columns) == ["a", "b"] and df.loc[0, "a"] == "É"
    assert any("UTF-16" in n for n in df.attrs["lumi_notices"])


def test_utf8_bom_header_clean_no_notice():
    df = _ld(b"\xef\xbb\xbfid,name\n1,x\n")
    assert list(df.columns) == ["id", "name"]
    assert "lumi_notices" not in df.attrs


def test_quoted_after_space():
    df = _ld(b'a, "b", "c"\n1, "x", "y"\n2, "z", "w"\n')
    assert list(df.columns) == ["a", "b", "c"]
    assert df["b"].tolist() == ["x", "z"]


def test_skipinitialspace_strips_leading_unquoted_spaces():
    df = _ld(b"a,b\n1, x\n2,y \n")
    assert df["b"].tolist() == ["x", "y "]  # leading stripped, trailing kept


def test_plain_csv_matches_pandas(tmp_path):
    for f in list(Path(__file__).parent.glob("**/*.csv"))[:5] + list(Path(__file__).parent.parent.glob("mock_data/*.csv"))[:5]:
        raw = f.read_bytes()
        try:
            expect = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
        except Exception:
            continue
        got, notices = read_tabular(raw, f.name)
        pd.testing.assert_frame_equal(got, expect, check_dtype=True)
    plain = b"id,name,score\n1,a,1.5\n2,b,\n3,c,3\n"
    got, notices = read_tabular(plain, "p.csv")
    pd.testing.assert_frame_equal(got, pd.read_csv(io.BytesIO(plain)))
    assert notices == []


def test_nrows_with_sniffing_and_encoding():
    raw = "id;nom\n".encode() + b"\n".join(f"{i};caf\xe9".encode("latin-1") for i in range(50))
    df = _ld(raw, nrows=5)
    assert len(df) == 5 and list(df.columns) == ["id", "nom"] and df.loc[0, "nom"] == "café"


VARIANTS = {
    "semi.csv": b"a;b\n1;2,5\n3;4\n",
    "tab.csv": b"a\tb\n1\t2\n",
    "cp.csv": "a,b\né,€\n".encode("cp1252"),
    "u16.csv": "a,b\nx,y\n".encode("utf-16"),
    "bom.csv": b"\xef\xbb\xbfa,b\n1,2\n",
    "q.csv": b'a, "b"\n1, "x"\n',
    "plain.csv": b"a,b\n1,2\n",
    "n.xlsx": _xlsx([["Name", 2019], ["a", 1]]),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_load_full_data_matches_load_data(name, monkeypatch):
    raw = VARIANTS[name]
    expect = _ld(raw, name)
    monkeypatch.setattr(sio.st, "session_state", {"full_source": {"name": name, "bytes": raw}})
    got = sio.load_full_data()
    pd.testing.assert_frame_equal(got, expect)


def test_cache_key_distinguishes_content():
    a = _hash_file_or_buffer(CachedFileWrapper(b"a;b\n1;2\n", "x.csv"))
    b = _hash_file_or_buffer(CachedFileWrapper(b"a;b\n1;3\n", "x.csv"))
    assert a != b
    assert _ld(b"a;b\n1;2\n")["b"].iloc[0] == 2 and _ld(b"a;b\n1;3\n")["b"].iloc[0] == 3


def test_upload_notice_toast_and_state():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.session_state["_is_testing"] = True
    raw = "name,city\nJosé,Peña\n".encode("cp1252")
    at.file_uploader(key="welcome_uploader").upload("c.csv", raw, "text/csv").run()
    assert not at.exception
    assert any("Windows-1252" in n for n in at.session_state.load_notices)
    assert any("Windows-1252" in t.value for t in at.toast)
    assert "lumi_notices" not in at.session_state.raw_data.attrs
