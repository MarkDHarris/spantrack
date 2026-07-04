"""Tests for the shared lxml reading primitives."""

from pathlib import Path

import pytest

from spantrack.lxml_reading import (
    XHTML_NS,
    parse_xml_file_or_none,
    qname,
    split_fragment,
)


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("OEBPS/ch1.xhtml#s01", ("OEBPS/ch1.xhtml", "s01")),
        ("OEBPS/ch1.xhtml", ("OEBPS/ch1.xhtml", "")),
        ("#frag", ("", "frag")),
        ("plain#", ("plain", "")),
        ("a#b#c", ("a", "b#c")),
        ("", ("", "")),
    ],
)
def test_split_fragment(href: str, expected: tuple[str, str]):
    assert split_fragment(href) == expected


def test_parse_xml_file_or_none_parses_valid(tmp_path: Path):
    path = tmp_path / "doc.xml"
    path.write_text(
        f'<html xmlns="{XHTML_NS}"><body><p>Hi.</p></body></html>',
        encoding="utf-8",
    )

    root = parse_xml_file_or_none(path)
    assert root is not None
    assert root.tag == qname(XHTML_NS, "html")


def test_parse_xml_file_or_none_returns_none_for_missing(tmp_path: Path):
    assert parse_xml_file_or_none(tmp_path / "ghost.xml") is None


def test_parse_xml_file_or_none_returns_none_for_malformed(tmp_path: Path):
    path = tmp_path / "bad.xml"
    path.write_text("<html><body><p>oops</wrong></body></html>", encoding="utf-8")

    assert parse_xml_file_or_none(path) is None
