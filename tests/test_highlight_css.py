"""Tests for highlight stylesheet generation and `<head>` link injection."""

from pathlib import Path

import pytest
from lxml import etree

from spantrack.highlight_css import (
    ACTIVE_CLASS,
    link_into_head,
    stylesheet,
    write_stylesheet,
)
from spantrack.lxml_reading import XmlElement, hardened_parser

_XHTML = "http://www.w3.org/1999/xhtml"
_HREF = "spantrack/highlight.css"


def _chapter() -> XmlElement:
    doc = (
        f'<html xmlns="{_XHTML}"><head><title>t</title></head>'
        "<body><p>Text.</p></body></html>"
    )
    return etree.fromstring(doc.encode(), hardened_parser())


def _head(root: XmlElement) -> XmlElement:
    head = root.find(f"{{{_XHTML}}}head")
    assert head is not None
    return head


def test_stylesheet_defines_active_class_and_dark_variant():
    css = stylesheet()

    assert f".{ACTIVE_CLASS}" in css
    assert "prefers-color-scheme: dark" in css


def test_write_stylesheet_writes_under_spantrack_and_returns_relative_path(
    tmp_path: Path,
):
    rel = write_stylesheet(tmp_path)

    assert rel == _HREF
    assert (tmp_path / rel).read_text(encoding="utf-8") == stylesheet()


def test_link_into_head_adds_one_stylesheet_link():
    root = _chapter()

    link_into_head(root, _HREF)

    links = _head(root).findall(f"{{{_XHTML}}}link")
    assert len(links) == 1
    assert links[0].get("rel") == "stylesheet"
    assert links[0].get("href") == _HREF


def test_link_into_head_is_idempotent():
    root = _chapter()

    link_into_head(root, _HREF)
    link_into_head(root, _HREF)

    assert len(_head(root).findall(f"{{{_XHTML}}}link")) == 1


def test_link_into_head_leaves_body_untouched():
    root = _chapter()
    body = root.find(f"{{{_XHTML}}}body")
    assert body is not None

    before = etree.tostring(body)

    link_into_head(root, _HREF)

    assert etree.tostring(body) == before


def test_link_into_head_raises_without_head():
    root = etree.fromstring(
        f'<html xmlns="{_XHTML}"><body><p>x</p></body></html>'.encode(),
        hardened_parser(),
    )

    with pytest.raises(ValueError, match="no <head>"):
        link_into_head(root, _HREF)
