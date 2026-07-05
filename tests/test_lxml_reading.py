"""Tests for the shared lxml reading primitives."""

from pathlib import Path

import pytest
from lxml import etree

from spantrack.lxml_reading import (
    XHTML_NS,
    parse_xml_file,
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


def test_hardened_parser_does_not_expand_external_entities(tmp_path: Path):
    # XXE guard: a SYSTEM entity pointing at a local file must never be expanded, so its
    # contents can't leak into the parsed tree. Locks resolve_entities=False / load_dtd=
    # False / no_network=True -- dropping any of them would leak the secret here.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    doc = tmp_path / "xxe.xml"
    doc.write_text(
        '<?xml version="1.0"?>\n'
        f'<!DOCTYPE root [ <!ENTITY xxe SYSTEM "{secret.as_uri()}"> ]>\n'
        "<root>&xxe;</root>",
        encoding="utf-8",
    )

    root = parse_xml_file(doc)

    assert "TOP-SECRET" not in etree.tostring(root, encoding="unicode")
