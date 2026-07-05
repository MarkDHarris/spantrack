from pathlib import Path

import pytest
from lxml import etree

from spantrack import epub2_upgrade, epub_container, opf
from spantrack.lxml_reading import XmlElement

_MODIFIED = "2026-07-03T00:00:00Z"

_XHTML = "http://www.w3.org/1999/xhtml"
_EPUB = "http://www.idpf.org/2007/ops"
_OPF = "http://www.idpf.org/2007/opf"
_DC = "http://purl.org/dc/elements/1.1/"

_CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<container version="1.0"'
    ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf"'
    ' media-type="application/oebps-package+xml"/></rootfiles></container>'
)


def _make_epub2(
    root: Path,
    opf_body: str,
    *,
    ncx_body: str | None = None,
    docs: dict[str, str] | None = None,
) -> Path:
    (root / "META-INF").mkdir(parents=True)
    (root / "META-INF" / "container.xml").write_text(_CONTAINER_XML, encoding="utf-8")

    oebps = root / "OEBPS"
    oebps.mkdir(parents=True, exist_ok=True)

    (oebps / "content.opf").write_text(opf_body, encoding="utf-8")

    if ncx_body is not None:
        (oebps / "toc.ncx").write_text(ncx_body, encoding="utf-8")

    for name, text in (docs or {}).items():
        (oebps / name).write_text(text, encoding="utf-8")

    return root


def _opf2(
    manifest: str,
    spine: str,
    guide: str = "",
    *,
    metadata: str | None = None,
) -> str:
    # metadata=None uses a default <metadata> block; metadata="" omits the element
    # entirely (drives the missing-metadata negative test); any other value is inlined.
    meta = (
        metadata
        if metadata is not None
        else (
            '<dc:identifier id="pub-id">urn:uuid:2</dc:identifier>'
            "<dc:title>T</dc:title><dc:language>en</dc:language>"
        )
    )
    metadata_block = (
        ""
        if metadata == ""
        else f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">{meta}</metadata>'
    )
    manifest_block = "" if manifest == "" else f"<manifest>{manifest}</manifest>"

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0"'
        ' unique-identifier="pub-id">'
        f"{metadata_block}{manifest_block}"
        f'<spine toc="ncx">{spine}</spine>{guide}</package>'
    )


def _ncx(navmap_inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        "<head/><docTitle><text>T</text></docTitle>"
        f"<navMap>{navmap_inner}</navMap></ncx>"
    )


def _doc(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f"<title>{title}</title></head><body><p>x</p></body></html>"
    )


def _unzip(epub_path: Path, tmp_path: Path) -> Path:
    book = tmp_path / "book"
    epub_container.unzip(epub_path, book)

    return book


def _nav_root(book: Path, package: opf.Package) -> XmlElement:
    nav = package.nav_item()
    assert nav is not None

    return etree.parse(str(book / nav.href)).getroot()


def _nav_href_to_label(nav_root: XmlElement, nav_type: str) -> dict[str, str]:
    navs = [
        el
        for el in nav_root.iter(f"{{{_XHTML}}}nav")
        if el.get(f"{{{_EPUB}}}type") == nav_type
    ]
    assert len(navs) == 1

    return {
        (anchor.get("href") or ""): (anchor.text or "")
        for anchor in navs[0].iter(f"{{{_XHTML}}}a")
    }


def _landmark_type_to_href(nav_root: XmlElement) -> dict[str, str]:
    navs = [
        el
        for el in nav_root.iter(f"{{{_XHTML}}}nav")
        if el.get(f"{{{_EPUB}}}type") == "landmarks"
    ]
    assert len(navs) == 1

    return {
        (anchor.get(f"{{{_EPUB}}}type") or ""): (anchor.get("href") or "")
        for anchor in navs[0].iter(f"{{{_XHTML}}}a")
    }


def _dcterms_modified(book: Path, package: opf.Package) -> list[str]:
    root = etree.parse(str(book / package.opf_path)).getroot()

    return [
        (meta.text or "")
        for meta in root.iter(f"{{{_OPF}}}meta")
        if meta.get("property") == "dcterms:modified"
    ]


def test_upgrade_sets_version_3(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    assert package.version == "3.0"


def test_upgrade_adds_nav_manifest_item_and_file(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    nav = package.nav_item()
    assert nav is not None
    assert "nav" in nav.properties
    assert (book / nav.href).is_file()


def test_upgrade_toc_lists_ncx_entries(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    toc = _nav_href_to_label(_nav_root(book, package), "toc")
    assert toc == {"chapter1.xhtml": "Chapter 1"}


def test_upgrade_landmarks_from_guide(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    landmarks = _landmark_type_to_href(_nav_root(book, package))
    assert landmarks["cover"] == "cover.xhtml"
    assert landmarks["bodymatter"] == "chapter1.xhtml"


def test_upgrade_adds_single_dcterms_modified(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    assert _dcterms_modified(book, package) == [_MODIFIED]


def test_upgrade_retains_ncx(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)
    assert package.spine_toc == "ncx"

    ncx = package.ncx_item()
    assert ncx is not None
    assert (book / ncx.href).is_file()


def test_upgrade_nav_is_wellformed_xhtml(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    root = _nav_root(book, package)
    assert root.tag == f"{{{_XHTML}}}html"


def test_upgrade_epub3_is_noop(minimal_epub_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub_path, tmp_path)
    before = opf.read_package(book)

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)
    assert package.version == "3.0"
    assert len(package.manifest) == len(before.manifest)

    nav = package.nav_item()
    assert nav is not None
    assert nav.item_id == "nav"

    assert _dcterms_modified(book, package) == ["2026-01-01T00:00:00Z"]


def test_upgrade_is_idempotent(minimal_epub2_path: Path, tmp_path: Path):
    book = _unzip(minimal_epub2_path, tmp_path)

    once = epub2_upgrade.upgrade(book, modified=_MODIFIED)
    once_nav = once.nav_item()
    assert once_nav is not None

    nav_bytes = (book / once_nav.href).read_bytes()
    opf_bytes = (book / once.opf_path).read_bytes()

    twice = epub2_upgrade.upgrade(book, modified=_MODIFIED)
    twice_nav = twice.nav_item()
    assert twice_nav is not None

    assert len(twice.manifest) == len(once.manifest)
    assert (book / twice.opf_path).read_bytes() == opf_bytes
    assert (book / twice_nav.href).read_bytes() == nav_bytes


def test_toc_keeps_children_of_targetless_navpoint(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>'
    )
    navmap = (
        '<navPoint id="p"><navLabel><text>Part One</text></navLabel>'
        '<navPoint id="n1"><navLabel><text>One</text></navLabel>'
        '<content src="c1.xhtml"/></navPoint>'
        '<navPoint id="n2"><navLabel><text>Two</text></navLabel>'
        '<content src="c2.xhtml"/></navPoint>'
        "</navPoint>"
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/><itemref idref="c2"/>'),
        ncx_body=_ncx(navmap),
        docs={"c1.xhtml": _doc("One"), "c2.xhtml": _doc("Two")},
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    toc = _nav_href_to_label(_nav_root(book, package), "toc")
    assert toc["c1.xhtml"] == "One"
    assert toc["c2.xhtml"] == "Two"


def test_toc_falls_back_to_spine_when_navmap_empty(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/>'),
        ncx_body=_ncx(""),
        docs={"c1.xhtml": _doc("First Chapter")},
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    toc = _nav_href_to_label(_nav_root(book, package), "toc")
    assert toc == {"c1.xhtml": "First Chapter"}


def test_landmarks_normalize_acknowledgements_spelling(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="a" href="ack.xhtml" media-type="application/xhtml+xml"/>'
    )
    guide = '<guide><reference type="acknowledgements" href="ack.xhtml"/></guide>'
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="a"/>', guide),
        ncx_body=_ncx(
            '<navPoint id="a"><navLabel><text>Ack</text></navLabel>'
            '<content src="ack.xhtml"/></navPoint>',
        ),
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    assert "acknowledgments" in _landmark_type_to_href(_nav_root(book, package))


def test_landmarks_omit_unknown_guide_types(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    )
    guide = (
        "<guide>"
        '<reference type="text" href="c1.xhtml"/>'
        '<reference type="not-a-real-type" href="c1.xhtml"/>'
        "</guide>"
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/>', guide),
        ncx_body=_ncx(
            '<navPoint id="c1"><navLabel><text>One</text></navLabel>'
            '<content src="c1.xhtml"/></navPoint>',
        ),
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    landmarks = _landmark_type_to_href(_nav_root(book, package))
    assert "bodymatter" in landmarks
    assert "not-a-real-type" not in landmarks


def test_landmarks_dedupe_by_type(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>'
    )
    guide = (
        "<guide>"
        '<reference type="cover" href="c1.xhtml"/>'
        '<reference type="cover" href="c2.xhtml"/>'
        "</guide>"
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/>', guide),
        ncx_body=_ncx(
            '<navPoint id="c1"><navLabel><text>One</text></navLabel>'
            '<content src="c1.xhtml"/></navPoint>',
        ),
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    root = _nav_root(book, package)
    covers = [
        anchor
        for nav in root.iter(f"{{{_XHTML}}}nav")
        if nav.get(f"{{{_EPUB}}}type") == "landmarks"
        for anchor in nav.iter(f"{{{_XHTML}}}a")
        if anchor.get(f"{{{_EPUB}}}type") == "cover"
    ]
    assert len(covers) == 1
    assert covers[0].get("href") == "c1.xhtml"


def test_upgrade_raises_without_manifest(tmp_path: Path):
    book = _make_epub2(
        tmp_path / "book",
        _opf2("", '<itemref idref="c1"/>'),
    )

    with pytest.raises(opf.OpfError, match="manifest"):
        epub2_upgrade.upgrade(book, modified=_MODIFIED)


def test_upgrade_raises_without_metadata(tmp_path: Path):
    manifest = '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/>', metadata=""),
        docs={"c1.xhtml": _doc("One")},
    )

    with pytest.raises(opf.OpfError, match="metadata"):
        epub2_upgrade.upgrade(book, modified=_MODIFIED)


def test_free_nav_target_avoids_existing_names(tmp_path: Path):
    manifest = (
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(manifest, '<itemref idref="c1"/>'),
        ncx_body=_ncx(
            '<navPoint id="c1"><navLabel><text>One</text></navLabel>'
            '<content src="c1.xhtml"/></navPoint>',
        ),
        docs={
            "nav.xhtml": _doc("stale"),
            "c1.xhtml": _doc("One"),
        },
    )

    package = epub2_upgrade.upgrade(book, modified=_MODIFIED)

    ids = [item.item_id for item in package.manifest]
    assert len(ids) == len(set(ids))

    nav = package.nav_item()
    assert nav is not None
    assert (book / nav.href).is_file()
    # `nav` and `nav.xhtml` are both taken, so the generator steps past them to `nav-1`;
    # without pinning the stepped name a regression reusing `nav.xhtml` would silently
    # overwrite the stale file and still pass the existence check above.
    assert nav.item_id == "nav-1"
    assert nav.href.endswith("nav-1.xhtml")


_CH_MANIFEST = (
    '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
    '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
)
_CH_SPINE = '<itemref idref="ch1"/>'
_CH_NAVMAP = (
    '<navPoint id="n1"><navLabel><text>C1</text></navLabel>'
    '<content src="chapter1.xhtml"/></navPoint>'
)


def test_upgrade_migrates_opf_role_to_refining_meta(tmp_path: Path):
    opf_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0"'
        ' unique-identifier="pub-id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' xmlns:opf="http://www.idpf.org/2007/opf">'
        '<dc:identifier id="pub-id">urn:uuid:2</dc:identifier>'
        "<dc:title>T</dc:title><dc:language>en</dc:language>"
        '<dc:creator opf:role="aut" opf:file-as="Example, A.">A. Example</dc:creator>'
        "</metadata>"
        f"<manifest>{_CH_MANIFEST}</manifest>"
        f'<spine toc="ncx">{_CH_SPINE}</spine>'
        "</package>"
    )
    book = _make_epub2(
        tmp_path / "book",
        opf_body,
        ncx_body=_ncx(_CH_NAVMAP),
        docs={"chapter1.xhtml": _doc("C1")},
    )

    epub2_upgrade.upgrade(book, modified=_MODIFIED)

    root = etree.parse(str(book / "OEBPS" / "content.opf")).getroot()
    creator = next(el for el in root.iter(f"{{{_DC}}}creator"))
    assert not any(
        isinstance(key, str) and key.startswith(f"{{{_OPF}}}") for key in creator.attrib
    )
    creator_id = creator.get("id")
    assert creator_id

    metas = {
        (meta.get("property"), meta.get("refines")): (meta.text, meta.get("scheme"))
        for meta in root.iter(f"{{{_OPF}}}meta")
    }
    assert metas[("role", f"#{creator_id}")] == ("aut", "marc:relators")
    assert metas[("file-as", f"#{creator_id}")] == ("Example, A.", None)


def test_upgrade_normalizes_legacy_content_doctype(tmp_path: Path):
    legacy = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"'
        ' "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>C1</title></head>'
        "<body><p>x</p></body></html>"
    )
    book = _make_epub2(
        tmp_path / "book",
        _opf2(_CH_MANIFEST, _CH_SPINE),
        ncx_body=_ncx(_CH_NAVMAP),
        docs={"chapter1.xhtml": legacy},
    )

    epub2_upgrade.upgrade(book, modified=_MODIFIED)

    content = (book / "OEBPS" / "chapter1.xhtml").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "XHTML 1.1" not in content
    assert "PUBLIC" not in content


def test_upgrade_leaves_modern_content_doc_byte_identical(tmp_path: Path):
    modern = _doc("C1")  # already uses <!DOCTYPE html>
    book = _make_epub2(
        tmp_path / "book",
        _opf2(_CH_MANIFEST, _CH_SPINE),
        ncx_body=_ncx(_CH_NAVMAP),
        docs={"chapter1.xhtml": modern},
    )

    epub2_upgrade.upgrade(book, modified=_MODIFIED)

    assert (book / "OEBPS" / "chapter1.xhtml").read_text(encoding="utf-8") == modern
