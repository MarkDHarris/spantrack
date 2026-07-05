from pathlib import Path

import pytest
from lxml import etree

from spantrack import epub_container, opf

_OPF_NS = "http://www.idpf.org/2007/opf"
_OVERLAY_ITEMS = (
    '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
    '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml"'
    ' properties="nav"/>'
)


def _overlay() -> opf.ChapterOverlay:
    return opf.ChapterOverlay(
        content_item_id="ch1",
        audio_id="ch1-audio",
        audio_href="audio/chapter1.mp3",
        smil_id="ch1-smil",
        smil_href="chapter1.smil",
        duration_s=12.5,
    )


def _container_with_opf(tmp_path: Path) -> Path:
    return _write_container(
        tmp_path / "book",
        _opf_with_manifest(_OVERLAY_ITEMS, spine='<itemref idref="ch1"/>'),
        opf_relpath="content.opf",
    )


def _metas(opf_path: Path) -> dict[tuple[str | None, str | None], str | None]:
    root = etree.parse(str(opf_path)).getroot()
    return {
        (meta.get("property"), meta.get("refines")): meta.text
        for meta in root.iter(f"{{{_OPF_NS}}}meta")
    }


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

    with pytest.raises(opf.OpfError, match="outside the container"):
        opf.read_package(_write_container(tmp_path / "book", body))


def test_rejects_absolute_href(tmp_path: Path):
    body = _opf_with_manifest(
        '<item id="ok" href="/etc/passwd" media-type="application/xhtml+xml"/>',
    )

    with pytest.raises(opf.OpfError, match="outside the container"):
        opf.read_package(_write_container(tmp_path / "book", body))


def test_rejects_backslash_separated_href(tmp_path: Path):
    # A backslash href stays one component under posixpath but escapes once a Windows
    # Path joins it, so it is rejected like an absolute href.
    body = _opf_with_manifest(
        '<item id="ok" href="..\\evil.xhtml" media-type="application/xhtml+xml"/>',
    )

    with pytest.raises(opf.OpfError, match="outside the container"):
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


_NO_METADATA_OPF = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">\n'
    "  <manifest>"
    '<item id="ch1" href="c.xhtml" media-type="application/xhtml+xml"/>'
    "</manifest>\n"
    '  <spine><itemref idref="ch1"/></spine>\n'
    "</package>\n"
)


def test_patch_registers_overlay_and_stylesheet_manifest_items(tmp_path: Path):
    root = _container_with_opf(tmp_path)

    opf.patch_opf_media_overlays(
        root / "content.opf",
        [_overlay()],
        stylesheet=opf.ManifestEntry(
            "highlight-css", "spantrack/highlight.css", "text/css"
        ),
        narrator="Kokoro af_heart",
        active_class="-spantrack-active",
    )

    package = opf.read_package(root)
    audio = package.item_by_id("ch1-audio")
    smil = package.item_by_id("ch1-smil")
    css = package.item_by_id("highlight-css")
    assert audio is not None
    assert audio.media_type == opf.MP3_MEDIA_TYPE
    assert audio.href == "audio/chapter1.mp3"
    assert smil is not None
    assert smil.media_type == opf.SMIL_MEDIA_TYPE
    assert css is not None
    assert css.media_type == "text/css"


def test_patch_sets_media_overlay_attr_and_media_metadata(tmp_path: Path):
    root = _container_with_opf(tmp_path)
    opf_path = root / "content.opf"

    opf.patch_opf_media_overlays(
        opf_path,
        [_overlay()],
        narrator="Kokoro af_heart",
        active_class="-spantrack-active",
    )

    doc = etree.parse(str(opf_path)).getroot()
    content = next(i for i in doc.iter(f"{{{_OPF_NS}}}item") if i.get("id") == "ch1")
    assert content.get("media-overlay") == "ch1-smil"

    metas = _metas(opf_path)
    assert metas[("media:duration", "#ch1-smil")] == "0:00:12.500"
    assert metas[("media:duration", None)] == "0:00:12.500"  # total (single overlay)
    assert metas[("media:narrator", None)] == "Kokoro af_heart"
    assert metas[("media:active-class", None)] == "-spantrack-active"


def test_patch_total_duration_sums_all_overlays(tmp_path: Path):
    items = _OVERLAY_ITEMS + (
        '<item id="ch2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>'
    )
    root = _write_container(
        tmp_path / "book",
        _opf_with_manifest(items, spine='<itemref idref="ch1"/><itemref idref="ch2"/>'),
        opf_relpath="content.opf",
    )
    opf_path = root / "content.opf"
    overlays = [
        opf.ChapterOverlay("ch1", "a1", "a1.mp3", "s1", "s1.smil", 10.0),
        opf.ChapterOverlay("ch2", "a2", "a2.mp3", "s2", "s2.smil", 5.5),
    ]

    opf.patch_opf_media_overlays(
        opf_path, overlays, narrator="n", active_class="-spantrack-active"
    )

    assert _metas(opf_path)[("media:duration", None)] == "0:00:15.500"


def test_patch_raises_for_unknown_content_item(tmp_path: Path):
    root = _container_with_opf(tmp_path)
    unknown = opf.ChapterOverlay("missing", "a", "a.mp3", "s", "s.smil", 1.0)

    with pytest.raises(opf.OpfError, match="missing"):
        opf.patch_opf_media_overlays(
            root / "content.opf",
            [unknown],
            narrator="n",
            active_class="-spantrack-active",
        )


def test_patch_raises_when_metadata_missing(tmp_path: Path):
    root = _write_container(
        tmp_path / "book", _NO_METADATA_OPF, opf_relpath="content.opf"
    )

    with pytest.raises(opf.OpfError, match="metadata"):
        opf.patch_opf_media_overlays(
            root / "content.opf",
            [_overlay()],
            narrator="n",
            active_class="-spantrack-active",
        )
