import zipfile
from pathlib import Path

import pytest
from conftest import TreeReader

from spantrack import epub_container


def test_unzip_reproduces_every_file(
    minimal_epub_path: Path,
    minimal_epub_src: Path,
    tmp_path: Path,
    read_file_tree: TreeReader,
):
    dest = tmp_path / "book"
    epub_container.unzip(minimal_epub_path, dest)

    assert read_file_tree(dest) == read_file_tree(minimal_epub_src)


def test_unzip_preserves_mimetype_bytes(minimal_epub_path: Path, tmp_path: Path):
    dest = tmp_path / "book"
    epub_container.unzip(minimal_epub_path, dest)

    assert (dest / "mimetype").read_bytes() == b"application/epub+zip"


def test_unzip_returns_member_names_in_archive_order(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    dest = tmp_path / "book"
    names = epub_container.unzip(minimal_epub_path, dest)

    with zipfile.ZipFile(minimal_epub_path) as zf:
        expected = [info.filename for info in zf.infolist() if not info.is_dir()]
    # The documented contract is archive order with directory entries excluded, not the
    # fixture's incidental mimetype-first storage.
    assert names == expected


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "/etc/passwd",
        "d:/evil.txt",  # drive letter: posixpath ignores it, Path.joinpath honors it
        "C:/evil.txt",
    ],
)
def test_unzip_rejects_unsafe_member_paths(tmp_path: Path, member: str):
    evil = tmp_path / "evil.epub"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr(member, "gotcha")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        epub_container.unzip(evil, tmp_path / "book")


def test_repack_round_trip_preserves_every_file(
    minimal_epub_path: Path,
    tmp_path: Path,
    read_file_tree: TreeReader,
):
    first = tmp_path / "first"
    names = epub_container.unzip(minimal_epub_path, first)
    out = tmp_path / "out.epub"

    epub_container.repack(first, out, order=names)

    second = tmp_path / "second"
    epub_container.unzip(out, second)

    assert read_file_tree(second) == read_file_tree(first)


def test_repack_writes_mimetype_first_stored_no_extra(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    src = tmp_path / "src"
    names = epub_container.unzip(minimal_epub_path, src)
    out = tmp_path / "out.epub"

    epub_container.repack(src, out, order=names)

    with zipfile.ZipFile(out) as zf:
        infos = zf.infolist()

    assert infos[0].filename == "mimetype"
    assert infos[0].compress_type == zipfile.ZIP_STORED
    assert infos[0].extra == b""


def test_repack_preserves_given_member_order(minimal_epub_path: Path, tmp_path: Path):
    src = tmp_path / "src"
    names = epub_container.unzip(minimal_epub_path, src)
    out = tmp_path / "out.epub"

    epub_container.repack(src, out, order=names)

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == names


def test_repack_without_order_is_deterministic(minimal_epub_path: Path, tmp_path: Path):
    src = tmp_path / "src"
    epub_container.unzip(minimal_epub_path, src)

    out_a = tmp_path / "a.epub"
    out_b = tmp_path / "b.epub"
    epub_container.repack(src, out_a)
    epub_container.repack(src, out_b)

    with zipfile.ZipFile(out_a) as za, zipfile.ZipFile(out_b) as zb:
        assert za.namelist() == zb.namelist()
        assert za.namelist()[0] == "mimetype"


def test_repack_raises_without_mimetype(tmp_path: Path):
    src = tmp_path / "src"

    (src / "OEBPS").mkdir(parents=True)
    (src / "OEBPS" / "chapter1.xhtml").write_text("<html></html>")

    with pytest.raises(ValueError, match="mimetype"):
        epub_container.repack(src, tmp_path / "out.epub")


def test_real_epub_round_trips_every_file(
    real_epub_path: Path,
    tmp_path: Path,
    read_file_tree: TreeReader,
):
    first = tmp_path / "first"
    names = epub_container.unzip(real_epub_path, first)
    out = tmp_path / "out.epub"

    epub_container.repack(first, out, order=names)

    second = tmp_path / "second"
    epub_container.unzip(out, second)

    assert read_file_tree(second) == read_file_tree(first)
