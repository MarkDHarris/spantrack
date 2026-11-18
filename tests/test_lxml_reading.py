"""Tests for the shared lxml reading primitives."""

from pathlib import Path

import pytest
from lxml import etree

from spantrack.lxml_reading import (
    XHTML_NS,
    join_fragment,
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


@pytest.mark.parametrize(
    ("path", "fragment", "expected"),
    [
        ("OEBPS/ch1.xhtml", "s01", "OEBPS/ch1.xhtml#s01"),
        ("OEBPS/ch1.xhtml", "", "OEBPS/ch1.xhtml"),  # empty fragment: no bare '#'
        ("", "frag", "#frag"),
    ],
)
def test_join_fragment(path: str, fragment: str, expected: str):
    assert join_fragment(path, fragment) == expected


@pytest.mark.parametrize(
    "href",
    ["OEBPS/ch1.xhtml#s01", "OEBPS/ch1.xhtml", "#frag", "a#b#c"],
)
def test_join_fragment_inverts_split_fragment(href: str):
    # join_fragment is the inverse of split_fragment for any href without a bare
    # trailing '#'; two OPF-rewriting stages lean on that round-trip.
    assert join_fragment(*split_fragment(href)) == href


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


def test_parse_xml_file_raises_on_malformed(tmp_path: Path):
    # opf, pipeline, and epub2_upgrade rely on this raising contract propagating, and
    # parse_xml_file_or_none relies on catching it; lock it directly.
    path = tmp_path / "bad.xml"
    path.write_text("<html><body><p>oops</wrong></body></html>", encoding="utf-8")

    with pytest.raises(etree.XMLSyntaxError):
        parse_xml_file(path)


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


def test_parse_xml_file_resolves_html_entities_and_preserves_xml_entities(
    tmp_path: Path,
) -> None:
    """Verifies that non-standard HTML named entities (e.g. &copy;, &oacute;)

    are decoded to their Unicode equivalents without corrupting predefined XML
    entities (&amp;, &lt;, &gt;, &quot;, &apos;).
    """
    content = """<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
      <head>
        <title>Entities &amp; Test</title>
      </head>
      <body>
        <div id="copyright-notice">
          &copy; 2026 Author Name &oacute; l&apos;auteur
        </div>
        <p class="xml-test" data-desc="&quot;quoted&quot;">
          Math test: 5 &lt; 10 &amp;&amp; 10 &gt; 5
        </p>
        <p class="mixed">
          Accents: caf&eacute; &amp; cr&egrave;me with &nbsp;space.
        </p>
      </body>
    </html>
    """
    test_file = tmp_path / "chapter_entities.xhtml"
    test_file.write_text(content, encoding="utf-8")

    # Parsing must succeed without lxml.etree.XMLSyntaxError
    root = parse_xml_file(test_file)

    # 1. Check HTML entities decoded to unicode
    copyright_div = root.find(".//{http://www.w3.org/1999/xhtml}div[@id='copyright-notice']")
    assert copyright_div is not None
    div_text = copyright_div.text or ""
    assert "©" in div_text
    assert "&copy;" not in div_text
    assert "ó" in div_text
    assert "&oacute;" not in div_text
    assert "'" in div_text

    # 2. Check XML core entities preserved proper structure and parsed values
    title_el = root.find(".//{http://www.w3.org/1999/xhtml}title")
    assert title_el is not None
    assert title_el.text == "Entities & Test"

    math_p = root.find(".//{http://www.w3.org/1999/xhtml}p[@class='xml-test']")
    assert math_p is not None
    assert math_p.get("data-desc") == '"quoted"'
    assert "5 < 10 && 10 > 5" in (math_p.text or "")

    # 3. Check accented and whitespace entities
    mixed_p = root.find(".//{http://www.w3.org/1999/xhtml}p[@class='mixed']")
    assert mixed_p is not None
    mixed_text = mixed_p.text or ""
    assert "café & crème" in mixed_text
    assert "\u00a0" in mixed_text

def test_parse_xml_file_preserves_numeric_entities(tmp_path: Path) -> None:
    """Verifies that numeric decimal and hex entities are left intact for lxml."""
    content = """<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
      <body>
        <p>© and ó should parse as unicode.</p>
      </body>
    </html>
    """
    test_file = tmp_path / "numeric_entities.xhtml"
    test_file.write_text(content, encoding="utf-8")

    root = parse_xml_file(test_file)
    p_el = root.find(".//{http://www.w3.org/1999/xhtml}p")
    assert p_el is not None
    assert "© and ó should parse as unicode." in (p_el.text or "")