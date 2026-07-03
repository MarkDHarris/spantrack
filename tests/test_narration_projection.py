import math
from pathlib import Path

import pytest

from spantrack import narration_projection
from spantrack.narration_set import Document


def _write_doc(root: Path, name: str, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        f"<body><p>{text}</p></body></html>",
        encoding="utf-8",
    )


def _doc(href: str, index: int = 0) -> Document:
    return Document(item_id=f"id{index}", href=href, spine_index=index)


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
    (book).mkdir(parents=True)
    (book / "a.xhtml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        "<body>\n  <p>Hello\n     there.</p>\n</body></html>",
        encoding="utf-8",
    )

    projection = narration_projection.project(book, [_doc("a.xhtml")], bitrate="64k")

    assert projection.total_characters == len("Hello there.")


def test_ignores_script_and_style(tmp_path: Path):
    book = tmp_path / "book"
    book.mkdir(parents=True)
    (book / "a.xhtml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        "<body><style>p{color:red}</style><p>Read me.</p>"
        "<script>alert(1)</script></body></html>",
        encoding="utf-8",
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


def test_rejects_malformed_bitrate(tmp_path: Path):
    book = tmp_path / "book"
    _write_doc(book, "a.xhtml", "Hello there.")

    with pytest.raises(ValueError, match="bitrate"):
        narration_projection.project(book, [_doc("a.xhtml")], bitrate="fast")
