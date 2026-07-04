from pathlib import Path

import pytest
from narration_documents import make_document as _doc

from spantrack.content_namespace import (
    ContentNamespaceError,
    require_xhtml_namespace,
)

_XHTML = "http://www.w3.org/1999/xhtml"


def _write(root: Path, name: str, markup: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>' + markup,
        encoding="utf-8",
    )


def test_xhtml_document_passes(tmp_path: Path):
    _write(
        tmp_path,
        "ch.xhtml",
        f'<html xmlns="{_XHTML}"><head><title>t</title></head>'
        "<body><p>Hello.</p></body></html>",
    )

    require_xhtml_namespace(tmp_path, [_doc("ch.xhtml")])


def test_bare_namespace_document_raises(tmp_path: Path):
    _write(
        tmp_path,
        "ch.xhtml",
        "<html><head><title>t</title></head><body><p>Hello.</p></body></html>",
    )

    with pytest.raises(ContentNamespaceError, match=r"ch\.xhtml"):
        require_xhtml_namespace(tmp_path, [_doc("ch.xhtml")])


def test_bare_namespace_error_reports_missing_namespace(tmp_path: Path):
    # The diagnostic must distinguish "no namespace at all" from a foreign one.
    _write(tmp_path, "ch.xhtml", "<html><body><p>Hello.</p></body></html>")

    with pytest.raises(ContentNamespaceError, match="none"):
        require_xhtml_namespace(tmp_path, [_doc("ch.xhtml")])


def test_foreign_namespace_root_raises(tmp_path: Path):
    # A document rooted in a genuinely foreign namespace breaks every downstream stage
    # the same way a bare one does; the diagnostic reports the namespace it found.
    _write(
        tmp_path,
        "ch.xhtml",
        '<html xmlns="http://example.com/ns"><body><p>Hello.</p></body></html>',
    )

    with pytest.raises(ContentNamespaceError, match=r"example\.com"):
        require_xhtml_namespace(tmp_path, [_doc("ch.xhtml")])


def test_inline_foreign_namespace_does_not_raise(tmp_path: Path):
    # Root is XHTML; a re-declared foreign default namespace on a descendant (inline
    # SVG, MathML, a stray xmlns) lives under it and is excluded by the block walk, not
    # here. The guard checks the root only, so it must not fire on inline foreign nodes.
    _write(
        tmp_path,
        "ch.xhtml",
        f'<html xmlns="{_XHTML}"><body><p>Real.</p>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>x</text></svg>'
        '<p xmlns="http://example.com/ns">foreign</p>'
        "</body></html>",
    )

    require_xhtml_namespace(tmp_path, [_doc("ch.xhtml")])


def test_missing_file_is_skipped(tmp_path: Path):
    # A selected document whose file is absent is a distinct problem; this guard owns
    # only the namespace concern and leaves that one alone.
    require_xhtml_namespace(tmp_path, [_doc("ghost.xhtml")])


def test_malformed_xml_is_skipped(tmp_path: Path):
    # Not-well-formed XML is surfaced at segmentation, not here — one concern per guard.
    _write(tmp_path, "bad.xhtml", "<html><body><p>oops</wrong></body></html>")

    require_xhtml_namespace(tmp_path, [_doc("bad.xhtml")])


def test_offending_document_is_named_among_valid_ones(tmp_path: Path):
    _write(
        tmp_path,
        "ok.xhtml",
        f'<html xmlns="{_XHTML}"><body><p>Fine.</p></body></html>',
    )
    _write(tmp_path, "bad.xhtml", "<html><body><p>Bad.</p></body></html>")

    with pytest.raises(ContentNamespaceError, match=r"bad\.xhtml"):
        require_xhtml_namespace(
            tmp_path,
            [_doc("ok.xhtml", 0), _doc("bad.xhtml", 1)],
        )


def test_empty_document_list_passes(tmp_path: Path):
    require_xhtml_namespace(tmp_path, [])
