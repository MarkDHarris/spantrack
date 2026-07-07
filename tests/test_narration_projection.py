import math
from pathlib import Path

import pytest
from narration_documents import make_document as _doc
from xhtml_documents import write_xhtml

from spantrack import narration_projection


def _write_doc(root: Path, name: str, text: str) -> None:
    write_xhtml(root, name, f"<p>{text}</p>")


def test_counts_characters_across_documents(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")
    _write_doc(book, "b.xhtml", "Bye.")

    projection = narration_projection.project(
        book,
        [_doc("a.xhtml", 0), _doc("b.xhtml", 1)],
        bitrate="64k",
    )
    assert projection.total_characters == 16


def test_per_document_character_counts(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")
    _write_doc(book, "b.xhtml", "Bye.")

    projection = narration_projection.project(
        book,
        [_doc("a.xhtml", 0), _doc("b.xhtml", 1)],
        bitrate="64k",
    )

    counts = {doc.href: doc.characters for doc in projection.documents}
    assert counts == {"a.xhtml": 12, "b.xhtml": 4}


def test_collapses_markup_whitespace(tmp_path: Path):
    book = tmp_path / "book"
    write_xhtml(book, "a.xhtml", "\n  <p>Hello\n     there.</p>\n")

    projection = narration_projection.project(book, [_doc("a.xhtml")], bitrate="64k")

    assert projection.total_characters == len("Hello there.")


def test_ignores_script_and_style(tmp_path: Path):
    book = tmp_path / "book"
    write_xhtml(
        book,
        "a.xhtml",
        "<style>p{color:red}</style><p>Read me.</p><script>alert(1)</script>",
    )

    projection = narration_projection.project(book, [_doc("a.xhtml")], bitrate="64k")

    assert projection.total_characters == len("Read me.")


def test_duration_scales_with_character_count(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "short.xhtml", "abcd")
    _write_doc(book, "long.xhtml", "abcdefghijkl")

    short = narration_projection.project(book, [_doc("short.xhtml")], bitrate="64k")
    long_run = narration_projection.project(book, [_doc("long.xhtml")], bitrate="64k")

    assert math.isclose(long_run.duration_seconds, 3 * short.duration_seconds)


def test_speed_shortens_duration(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")

    at_1x = narration_projection.project(
        book,
        [_doc("a.xhtml")],
        bitrate="64k",
        speed=1.0,
    )
    at_2x = narration_projection.project(
        book,
        [_doc("a.xhtml")],
        bitrate="64k",
        speed=2.0,
    )

    assert math.isclose(at_2x.duration_seconds, at_1x.duration_seconds / 2)


def test_size_uses_bitrate(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")

    projection = narration_projection.project(book, [_doc("a.xhtml")], bitrate="64k")

    assert projection.size_bytes == round(projection.duration_seconds * 64000 / 8)


def test_higher_bitrate_larger_size(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")

    low = narration_projection.project(book, [_doc("a.xhtml")], bitrate="64k")
    high = narration_projection.project(book, [_doc("a.xhtml")], bitrate="128k")

    assert high.size_bytes > low.size_bytes


def test_empty_selection_projects_zero(tmp_path: Path):
    book = tmp_path / "book"
    book.mkdir()

    projection = narration_projection.project(book, [], bitrate="64k")

    assert projection.total_characters == 0
    assert projection.duration_seconds == 0.0
    assert projection.size_bytes == 0


def test_missing_document_counts_zero(tmp_path: Path):
    book = tmp_path / "book"
    book.mkdir()

    projection = narration_projection.project(
        book,
        [_doc("ghost.xhtml")],
        bitrate="64k",
    )

    assert projection.total_characters == 0


def test_malformed_document_counts_zero_without_crashing(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "good.xhtml", "Hello there.")
    (book / "bad.xhtml").write_text("<p>oops</wrong>", encoding="utf-8")

    projection = narration_projection.project(
        book,
        [_doc("good.xhtml", 0), _doc("bad.xhtml", 1)],
        bitrate="64k",
    )

    counts = {doc.href: doc.characters for doc in projection.documents}
    assert counts == {"good.xhtml": 12, "bad.xhtml": 0}
    assert projection.total_characters == 12


def test_bare_namespace_document_counts_zero(tmp_path: Path):
    # A chapter whose <html> omits the XHTML namespace has no {xhtml}body, so this
    # best-effort preview counts it zero. The pipeline's content_namespace guard runs
    # before projection and refuses such a document, so this zero is never surfaced.
    book = tmp_path / "book"
    book.mkdir(parents=True)
    (book / "bare.xhtml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        "<html><head><title>t</title></head>"
        "<body><p>Hello there.</p></body></html>",
        encoding="utf-8",
    )

    projection = narration_projection.project(book, [_doc("bare.xhtml")], bitrate="64k")

    assert projection.total_characters == 0


def test_rejects_malformed_bitrate(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")

    with pytest.raises(ValueError, match="bitrate"):
        narration_projection.project(book, [_doc("a.xhtml")], bitrate="fast")
