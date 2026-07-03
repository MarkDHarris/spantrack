from pathlib import Path

import pytest

from spantrack import epub_container, opf

_CONTAINER_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container version="1.0"'
    ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    "  <rootfiles>\n{rootfiles}  </rootfiles>\n"
    "</container>\n"
)


def _opf_with_manifest(items: str, spine: str = '<itemref idref="ok"/>') -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"'
        ' unique-identifier="pub-id">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="pub-id">urn:uuid:x</dc:identifier>\n'
        "    <dc:title>T</dc:title>\n"
        "    <dc:language>en</dc:language>\n"
        "  </metadata>\n"
        f"  <manifest>{items}</manifest>\n"
        f"  <spine>{spine}</spine>\n"
        "</package>\n"
    )


def _write_container(
    root: Path,
    opf_body: str,
    *,
    opf_relpath: str = "OEBPS/content.opf",
    rootfiles: str | None = None,
) -> Path:
    meta_inf = root / "META-INF"
    meta_inf.mkdir(parents=True, exist_ok=True)
    default_rootfiles = (
        f'    <rootfile full-path="{opf_relpath}"'
        ' media-type="application/oebps-package+xml"/>\n'
    )
    container_xml = _CONTAINER_TEMPLATE.format(
        rootfiles=rootfiles if rootfiles is not None else default_rootfiles,
    )
    (meta_inf / "container.xml").write_text(container_xml, encoding="utf-8")

    opf_path = root / opf_relpath
    opf_path.parent.mkdir(parents=True, exist_ok=True)
    opf_path.write_text(opf_body, encoding="utf-8")

    return root


def _read_package(epub_path: Path, tmp_path: Path) -> opf.Package:
    book = tmp_path / "book"
    epub_container.unzip(epub_path, book)

    return opf.read_package(book)


def test_locates_opf_via_container(minimal_epub_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub_path, tmp_path)

    assert package.opf_path == "OEBPS/content.opf"


def test_reads_epub3_version_and_spine(minimal_epub_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub_path, tmp_path)

    assert package.version == "3.0"
    assert [item.idref for item in package.spine] == ["ch1"]


def test_resolves_manifest_hrefs_against_opf_dir(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    package = _read_package(minimal_epub_path, tmp_path)

    hrefs = {item.item_id: item.href for item in package.manifest}
    assert hrefs["ch1"] == "OEBPS/chapter1.xhtml"
    assert hrefs["nav"] == "OEBPS/nav.xhtml"


def test_flags_nav_item_by_property(minimal_epub_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub_path, tmp_path)

    nav = package.nav_item()
    assert nav is not None
    assert nav.item_id == "nav"
    assert "nav" in nav.properties


def test_epub3_has_no_ncx_or_guide(minimal_epub_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub_path, tmp_path)

    assert package.spine_toc is None
    assert package.ncx_item() is None
    assert package.guide == ()


def test_reads_epub2_version(minimal_epub2_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub2_path, tmp_path)

    assert package.version == "2.0"
    assert package.nav_item() is None


def test_epub2_spine_toc_resolves_to_ncx(minimal_epub2_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub2_path, tmp_path)
    assert package.spine_toc == "ncx"

    ncx = package.ncx_item()
    assert ncx is not None
    assert ncx.href == "OEBPS/toc.ncx"
    assert ncx.media_type == "application/x-dtbncx+xml"


def test_epub2_spine_linear_flags(minimal_epub2_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub2_path, tmp_path)

    linear = {item.idref: item.linear for item in package.spine}
    assert linear == {"cover": False, "ch1": True}


def test_epub2_guide_references_resolved(minimal_epub2_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub2_path, tmp_path)

    guide = {ref.kind: ref.href for ref in package.guide}
    assert guide["cover"] == "OEBPS/cover.xhtml"
    assert guide["text"] == "OEBPS/chapter1.xhtml"


def test_item_by_id_returns_none_for_unknown(minimal_epub_path: Path, tmp_path: Path):
    package = _read_package(minimal_epub_path, tmp_path)

    assert package.item_by_id("does-not-exist") is None


def test_skips_manifest_item_with_empty_href(tmp_path: Path):
    body = _opf_with_manifest(
        '<item id="ok" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="empty" href="" media-type="application/xhtml+xml"/>',
    )

    package = opf.read_package(_write_container(tmp_path / "book", body))
    assert package.item_by_id("empty") is None

    ok = package.item_by_id("ok")
    assert ok is not None
    assert ok.href == "OEBPS/chapter1.xhtml"


def test_skips_spine_itemref_with_empty_idref(tmp_path: Path):
    body = _opf_with_manifest(
        '<item id="ok" href="chapter1.xhtml" media-type="application/xhtml+xml"/>',
        spine='<itemref idref=""/><itemref idref="ok"/>',
    )
    package = opf.read_package(_write_container(tmp_path / "book", body))

    assert [item.idref for item in package.spine] == ["ok"]


def test_rejects_href_escaping_container(tmp_path: Path):
    body = _opf_with_manifest(
        '<item id="ok" href="../../evil.xhtml" media-type="application/xhtml+xml"/>',
    )

    with pytest.raises(opf.OpfError, match="escapes the container"):
        opf.read_package(_write_container(tmp_path / "book", body))


def test_rejects_absolute_href(tmp_path: Path):
    body = _opf_with_manifest(
        '<item id="ok" href="/etc/passwd" media-type="application/xhtml+xml"/>',
    )

    with pytest.raises(opf.OpfError, match="absolute"):
        opf.read_package(_write_container(tmp_path / "book", body))


def test_selects_opf_rootfile_by_media_type(tmp_path: Path):
    rootfiles = (
        '    <rootfile full-path="cover.pdf" media-type="application/pdf"/>\n'
        '    <rootfile full-path="OEBPS/content.opf"'
        ' media-type="application/oebps-package+xml"/>\n'
    )
    body = _opf_with_manifest(
        '<item id="ok" href="chapter1.xhtml" media-type="application/xhtml+xml"/>',
    )
    root = _write_container(tmp_path / "book", body, rootfiles=rootfiles)

    package = opf.read_package(root)

    assert package.opf_path == "OEBPS/content.opf"
