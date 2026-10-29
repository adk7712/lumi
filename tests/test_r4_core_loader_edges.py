"""Loader edge cases from the final fuzz pass: markup false positives, JSON arrays, odd types."""
import io
import os

os.environ.setdefault("LUMI_TESTING", "1")

from lumi_state.core import CachedFileWrapper, load_data


def _Upload(data, name, type_=None):
    if type_:
        return CachedFileWrapper(data, name)
    buf = io.BytesIO(data)  # no .type: the loader falls back to the name
    buf.name = name
    return buf


def _err(df):
    return df.attrs.get("lumi_load_error")


def test_csv_header_starting_with_angle_bracket_loads():
    df = load_data(_Upload(b"<a>,b\n1,2\n", "t.csv", "text/csv"))
    assert not _err(df) and list(df.columns) == ["<a>", "b"]


def test_json_array_is_rejected():
    df = load_data(_Upload(b'[\n{"a":1},\n{"a":2}\n]\n', "t.csv", "text/csv"))
    assert "JSON" in _err(df)


def test_html_still_rejected():
    df = load_data(_Upload(b"<html>\n<body><p>hi, there</p>\n</body></html>\n", "t.csv", "text/csv"))
    assert "HTML" in _err(df)


def test_xls_name_without_type_is_unsupported():
    df = load_data(_Upload(b"hello world\n", "t.xls"))
    assert "Unsupported file type" in _err(df)


def test_only_delimiters_is_an_error():
    df = load_data(_Upload(b",,,\n,,,\n", "t.csv", "text/csv"))
    assert "no columns" in _err(df)
