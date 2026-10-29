"""R4: friendlier loader messages for huge fields, bogus workbooks and HTML files."""
import io

from lumi_state.core import CachedFileWrapper, load_data


def _err(raw, name):
    return load_data(CachedFileWrapper(raw, name)).attrs.get("lumi_load_error", "")


def test_huge_single_field_loads():
    raw = b"a,b\n" + b"x" * 5_000_000 + b",1\n"
    df = load_data(CachedFileWrapper(raw, "big_field.csv"))
    assert not df.attrs.get("lumi_load_error") and df.shape == (1, 2)


def test_bogus_xlsx_named_xlsx_says_invalid_workbook():
    assert "not a valid Excel workbook" in _err(b"PK\x03\x04junkjunkjunk", "r4_bogus.xlsx")
    buf = io.BytesIO(b"PK\x03\x04junkjunkjunk")
    buf.name = "r4_bogus2.xlsx"  # no .type: the fallback path
    msg = load_data(buf).attrs.get("lumi_load_error", "")
    assert "not a valid Excel workbook" in msg and "Name it" not in msg


def test_multiline_html_says_markup():
    raw = b"<html>\n<body>a, b</body>\n</html>\n"
    assert "JSON or HTML/XML" in _err(raw, "r4_page.csv")
    assert "JSON or HTML/XML" in _err(b"\n  <div class='x'>\n<p>1,2,3</p>\n", "r4_page2.csv")


def test_csv_with_angle_values_still_loads():
    df = load_data(CachedFileWrapper(b"a,b\n<x>,2\n3,4\n", "r4_ok.csv"))
    assert df.shape == (2, 2) and not df.attrs.get("lumi_load_error")
