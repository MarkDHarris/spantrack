"""Tests for Media Overlay SMIL emission (sentence and word sync units)."""

import pytest
from lxml import etree

from spantrack.lxml_reading import XmlElement
from spantrack.smil import SyncUnit, build_smil, clock_value

_SMIL = "http://www.w3.org/ns/SMIL"
_EPUB = "http://www.idpf.org/2007/ops"


def _child(element: XmlElement, tag: str) -> XmlElement:
    found = element.find(f"{{{_SMIL}}}{tag}")
    assert found is not None
    return found


def _pars(xml: bytes) -> list[XmlElement]:
    return etree.fromstring(xml).findall(f".//{{{_SMIL}}}par")


def test_one_par_per_unit_in_document_order():
    units = [
        SyncUnit("s00001", 0.0, 1.25),
        SyncUnit("s00002", 1.25, 2.5),
    ]

    pars = _pars(build_smil("chapter1.xhtml", "chapter1.mp3", units))

    assert [_child(par, "text").get("src") for par in pars] == [
        "chapter1.xhtml#s00001",
        "chapter1.xhtml#s00002",
    ]
    assert all(_child(par, "audio").get("src") == "chapter1.mp3" for par in pars)


def test_clip_bounds_serialize_as_full_clock():
    units = [
        SyncUnit("s00001", 0.0, 1.25),
        SyncUnit("s00002", 1.25, 2.5),
    ]

    pars = _pars(build_smil("c.xhtml", "c.mp3", units))

    first, second = _child(pars[0], "audio"), _child(pars[1], "audio")
    assert first.get("clipBegin") == "0:00:00.000"
    assert first.get("clipEnd") == "0:00:01.250"
    assert second.get("clipBegin") == "0:00:01.250"
    assert second.get("clipEnd") == "0:00:02.500"


def test_declares_smil_and_epub_namespaces_and_textref():
    root = etree.fromstring(
        build_smil(
            "c.xhtml",
            "c.mp3",
            [SyncUnit("s00001", 0.0, 1.0)],
        ),
    )
    assert root.tag == f"{{{_SMIL}}}smil"
    assert root.get("version") == "3.0"

    seq = root.find(f".//{{{_SMIL}}}seq")
    assert seq is not None
    assert seq.get(f"{{{_EPUB}}}textref") == "c.xhtml"


def test_empty_unit_list_is_rejected():
    # A Media Overlay must have at least one <par>; a chapter with no narration must not
    # get a SMIL at all, so this is a caller error rather than an empty <seq>.
    with pytest.raises(ValueError, match="empty"):
        build_smil("c.xhtml", "c.mp3", [])


def test_clock_value_formats_full_clock():
    assert clock_value(0.0) == "0:00:00.000"
    assert clock_value(1.25) == "0:00:01.250"
    assert clock_value(3661.5) == "1:01:01.500"


def test_clock_value_carries_at_minute_boundary():
    # A bound just under a minute must not render an invalid `:60` seconds field: the ms
    # rounding has to carry into the minutes. 1_439_999/24000s rounds up to 60.000s.
    assert clock_value(1_439_999 / 24000) == "0:01:00.000"


def test_serializes_clip_bounds_verbatim_without_enforcing_gaplessness():
    # build_smil performs no gapless validation: each unit's bounds pass through
    # clock_value unchanged. A gap between units (0.4 -> 0.5) must survive verbatim.
    units = [SyncUnit("u001", 0.0, 0.4), SyncUnit("u002", 0.5, 0.9)]

    pars = _pars(build_smil("c.xhtml", "c.mp3", units))
    first, second = _child(pars[0], "audio"), _child(pars[1], "audio")

    assert first.get("clipBegin") == clock_value(0.0)
    assert first.get("clipEnd") == clock_value(0.4)
    assert second.get("clipBegin") == clock_value(0.5)
    assert second.get("clipEnd") == clock_value(0.9)
