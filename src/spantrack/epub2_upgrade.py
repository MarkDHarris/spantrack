"""Upgrade an EPUB 2 container in place to EPUB 3.

Downstream stages assume EPUB 3: a navigation document with `epub:type` landmarks and
`version="3.0"`. This stage is the sole producer of that shape from EPUB 2 inputs — it
synthesizes `nav.xhtml` from the NCX navMap and the OPF `<guide>`, bumps the package
version, stamps `dcterms:modified`, and leaves the NCX in place for reader fallback.
"""

import posixpath
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lxml import etree

from spantrack import opf
from spantrack.lxml_reading import (
    EPUB_NS,
    OPF_NS,
    XHTML_NS,
    XmlElement,
    document_ids,
    hardened_parser,
    join_fragment,
    parse_xml_file,
    parse_xml_file_or_none,
    qname,
    split_fragment,
)
from spantrack.opf import XHTML_MEDIA_TYPE, Package

__all__ = ["upgrade"]

_NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"

# Base name for the synthesized nav document and its manifest id.
_NAV_BASE_NAME = "nav"

# Pre-built tag names for the XHTML nav document this stage synthesizes.
_XHTML_HTML = qname(XHTML_NS, "html")
_XHTML_HEAD = qname(XHTML_NS, "head")
_XHTML_TITLE = qname(XHTML_NS, "title")
_XHTML_BODY = qname(XHTML_NS, "body")
_XHTML_NAV = qname(XHTML_NS, "nav")
_XHTML_OL = qname(XHTML_NS, "ol")
_XHTML_LI = qname(XHTML_NS, "li")
_XHTML_A = qname(XHTML_NS, "a")
_XHTML_SPAN = qname(XHTML_NS, "span")
_EPUB_TYPE = qname(EPUB_NS, "type")

# Translate OPF 2.0 <guide> reference types to EPUB 3 structural-semantics epub:type
# landmark values. Guide types absent from this map are omitted from the landmarks nav
# rather than passed through, since an unrecognized epub:type is not epubcheck-clean.
_GUIDE_TO_LANDMARK = {
    "cover": "cover",
    "title-page": "titlepage",
    "titlepage": "titlepage",
    "toc": "toc",
    "text": "bodymatter",
    "start": "bodymatter",
    "index": "index",
    "glossary": "glossary",
    "bibliography": "bibliography",
    "colophon": "colophon",
    "copyright-page": "copyright-page",
    "dedication": "dedication",
    "epigraph": "epigraph",
    "foreword": "foreword",
    "preface": "preface",
    "acknowledgements": "acknowledgments",
    "acknowledgments": "acknowledgments",
    "loi": "loi",
    "lot": "lot",
}


@dataclass(frozen=True, slots=True)
class _NavEntry:
    """One toc entry.

    `href` is None for a target-less structural heading (a navPoint with no `<content>`
    src), which `_append_list` renders as a `<span>` rather than an `<a>`.
    """

    label: str
    href: str | None
    children: tuple["_NavEntry", ...] = ()


@dataclass(frozen=True, slots=True)
class _Landmark:
    epub_type: str
    label: str
    href: str


def upgrade(container_root: Path, *, modified: str) -> Package:
    """Upgrade an unzipped EPUB 2 container to EPUB 3 in place.

    A container already at version 3.x is returned unchanged.

    Args:
        container_root: The container root directory (holds `META-INF/`).
        modified: The `dcterms:modified` timestamp (`CCYY-MM-DDThh:mm:ssZ`), supplied by
            the caller so upgrade output stays deterministic.

    Returns:
        The re-parsed package, reflecting the upgrade when one was applied.
    """
    package = opf.read_package(container_root)
    if not package.version.startswith("2"):
        return package

    base_dir = posixpath.dirname(package.opf_path)
    toc = _toc_entries(container_root, package, base_dir)
    landmarks = _landmark_entries(package, base_dir)
    opf_root = parse_xml_file(container_root / package.opf_path)
    nav_id, nav_filename = _free_nav_target(package, base_dir, opf_root)

    nav_path = container_root / posixpath.join(base_dir, nav_filename)
    nav_path.write_bytes(_render_nav(toc, landmarks))

    _patch_opf(
        container_root / package.opf_path,
        nav_id=nav_id,
        nav_href=nav_filename,
        modified=modified,
    )
    _normalize_content_doctypes(container_root, package)

    return opf.read_package(container_root)


def _toc_entries(
    container_root: Path,
    package: Package,
    base_dir: str,
) -> tuple[_NavEntry, ...]:
    ncx = package.ncx_item()
    entries: tuple[_NavEntry, ...] = ()
    if ncx is not None:
        nav_map = parse_xml_file(container_root / ncx.href).find(
            qname(_NCX_NS, "navMap"),
        )
        if nav_map is not None:
            entries = tuple(_nav_points(nav_map, posixpath.dirname(ncx.href), base_dir))

    # An absent, empty, or all-targetless navMap yields nothing; fall back to the spine
    # so the toc nav never renders an empty <ol> (which epubcheck rejects).
    return entries or _spine_entries(container_root, package, base_dir)


def _nav_points(
    parent: XmlElement,
    ncx_dir: str,
    base_dir: str,
) -> Iterator[_NavEntry]:
    for point in parent.iterfind(qname(_NCX_NS, "navPoint")):
        children = tuple(_nav_points(point, ncx_dir, base_dir))
        content = point.find(qname(_NCX_NS, "content"))
        src = content.get("src") if content is not None else None
        if not src and not children:
            continue

        label = point.find(f"{qname(_NCX_NS, 'navLabel')}/{qname(_NCX_NS, 'text')}")
        yield _NavEntry(
            label=(label.text or "").strip() if label is not None else "",
            href=(
                _relativize(_resolve_ncx_src(ncx_dir, src), base_dir) if src else None
            ),
            children=children,
        )


def _spine_entries(
    container_root: Path,
    package: Package,
    base_dir: str,
) -> tuple[_NavEntry, ...]:
    # NCX is mandatory in EPUB 2, but tolerate its absence: derive a flat TOC from the
    # linear spine, labelling each entry by its document <title>.
    entries: list[_NavEntry] = []
    for spine_item in package.spine:
        if not spine_item.linear:
            continue

        item = package.item_by_id(spine_item.idref)
        if item is None:
            continue

        stem = posixpath.splitext(posixpath.basename(item.href))[0]
        entries.append(
            _NavEntry(
                label=_document_title(container_root / item.href) or stem,
                href=_relativize(item.href, base_dir),
            ),
        )

    return tuple(entries)


def _landmark_entries(package: Package, base_dir: str) -> tuple[_Landmark, ...]:
    landmarks: list[_Landmark] = []
    seen_types: set[str] = set()
    for ref in package.guide:
        epub_type = _GUIDE_TO_LANDMARK.get(ref.kind)
        # An epub:type is unique per landmarks nav; keep the first reference of each and
        # drop guide types with no valid EPUB 3 equivalent.
        if epub_type is None or epub_type in seen_types:
            continue

        seen_types.add(epub_type)
        landmarks.append(
            _Landmark(
                epub_type=epub_type,
                label=ref.title or ref.kind.replace("-", " ").title(),
                href=_relativize(ref.href, base_dir),
            ),
        )

    return tuple(landmarks)


def _render_nav(
    toc: tuple[_NavEntry, ...],
    landmarks: tuple[_Landmark, ...],
) -> bytes:
    # lxml uses a None key for the default namespace; lxml-stubs types nsmap as
    # Mapping[str, str] and rejects it, so cast past the stub gap.
    nsmap = cast("dict[str, str]", {None: XHTML_NS, "epub": EPUB_NS})

    html = etree.Element(_XHTML_HTML, nsmap=nsmap)
    head = etree.SubElement(html, _XHTML_HEAD)
    etree.SubElement(head, _XHTML_TITLE).text = "Contents"
    body = etree.SubElement(html, _XHTML_BODY)

    _append_list(_nav_section(body, "toc"), toc)

    if landmarks:
        landmarks_list = etree.SubElement(
            _nav_section(body, "landmarks"),
            _XHTML_OL,
        )
        for landmark in landmarks:
            item = etree.SubElement(landmarks_list, _XHTML_LI)
            anchor = etree.SubElement(item, _XHTML_A)
            anchor.set("href", landmark.href)
            anchor.set(_EPUB_TYPE, landmark.epub_type)
            anchor.text = landmark.label

    body_xml = etree.tostring(html, encoding="utf-8", pretty_print=True)

    return b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n' + body_xml


def _nav_section(body: XmlElement, epub_type: str) -> XmlElement:
    # Each EPUB 3 nav carries its epub:type as both the type and the element id.
    nav = etree.SubElement(body, _XHTML_NAV)
    nav.set(_EPUB_TYPE, epub_type)
    nav.set("id", epub_type)

    return nav


def _append_list(parent: XmlElement, entries: tuple[_NavEntry, ...]) -> None:
    ordered = etree.SubElement(parent, _XHTML_OL)
    for entry in entries:
        item = etree.SubElement(ordered, _XHTML_LI)

        # A navPoint without a target (a structural heading) becomes a <span> label; the
        # EPUB 3 nav grammar allows it and it keeps the child hierarchy intact.
        label_tag = _XHTML_A if entry.href is not None else _XHTML_SPAN
        label = etree.SubElement(item, label_tag)
        label.text = entry.label

        if entry.href is not None:
            label.set("href", entry.href)

        if entry.children:
            _append_list(item, entry.children)


def _patch_opf(path: Path, *, nav_id: str, nav_href: str, modified: str) -> None:
    tree = etree.parse(str(path), hardened_parser())

    root = tree.getroot()
    root.set("version", "3.0")

    # EPUB 3 mandates a dcterms:modified meta and a nav manifest item; without these
    # host elements the OPF cannot be upgraded, so fail loudly instead of emitting
    # invalid output.
    metadata = root.find(qname(OPF_NS, "metadata"))
    if metadata is None:
        err_msg = "OPF has no <metadata> element to stamp dcterms:modified"
        raise opf.OpfError(err_msg)

    manifest = root.find(qname(OPF_NS, "manifest"))
    if manifest is None:
        err_msg = "OPF has no <manifest> element to register the nav document"
        raise opf.OpfError(err_msg)

    _migrate_dc_opf_attributes(root, metadata)

    meta = etree.SubElement(metadata, qname(OPF_NS, "meta"))
    meta.set("property", "dcterms:modified")
    meta.text = modified

    item = etree.SubElement(manifest, qname(OPF_NS, "item"))
    item.set("id", nav_id)
    item.set("href", nav_href)
    item.set("media-type", XHTML_MEDIA_TYPE)
    item.set("properties", "nav")

    tree.write(str(path), xml_declaration=True, encoding="utf-8")


_MARC_RELATORS = "marc:relators"

# A simple XHTML PUBLIC/SYSTEM DOCTYPE, no internal subset. EPUB 2 docs carry XHTML
# 1.0/1.1 doctypes invalid in EPUB 3; `[^\[>]*` stops at an internal subset, so an
# entity-declaring doctype is left alone rather than broken.
_LEGACY_DOCTYPE = re.compile(
    rb"<!DOCTYPE\s+html\s+(?:PUBLIC|SYSTEM)\b[^\[>]*>",
    re.IGNORECASE,
)


def _migrate_dc_opf_attributes(root: XmlElement, metadata: XmlElement) -> None:
    """Convert EPUB 2 `opf:*` metadata attributes to a valid EPUB 3 shape.

    EPUB 3 forbids `opf:*` attributes on `dc:*` elements. `opf:role`/`opf:file-as` move
    to `<meta refines>` children (role carries the `marc:relators` scheme); every other
    `opf:*` attribute (e.g. `opf:scheme`, `opf:event`) is dropped as invalid EPUB 3.
    """
    prefix = f"{{{OPF_NS}}}"
    existing_ids = document_ids(root)
    for element in list(metadata):
        opf_keys = [
            key for key in cast("list[str]", element.keys()) if key.startswith(prefix)
        ]
        if not opf_keys:
            continue

        element_id = element.get("id")
        if element_id is None:
            element_id = _unused(
                etree.QName(element).localname,
                lambda candidate: candidate in existing_ids,
            )
            existing_ids.add(element_id)
            element.set("id", element_id)

        for key in opf_keys:
            value = element.get(key) or ""
            del element.attrib[key]

            localname = key[len(prefix) :]
            if localname == "role":
                _add_refining_meta(
                    metadata, element_id, "role", value, scheme=_MARC_RELATORS
                )
            elif localname == "file-as":
                _add_refining_meta(metadata, element_id, "file-as", value)


def _add_refining_meta(
    metadata: XmlElement,
    refines_id: str,
    prop: str,
    value: str,
    *,
    scheme: str | None = None,
) -> None:
    meta = etree.SubElement(metadata, qname(OPF_NS, "meta"))
    meta.set("refines", f"#{refines_id}")
    meta.set("property", prop)
    if scheme is not None:
        meta.set("scheme", scheme)

    meta.text = value


def _normalize_content_doctypes(container_root: Path, package: Package) -> None:
    """Rewrite legacy XHTML DOCTYPEs in content documents to `<!DOCTYPE html>`.

    Only a simple `PUBLIC`/`SYSTEM` DOCTYPE declaration is rewritten, and only that
    declaration -- every other byte is preserved. A document with no legacy doctype, or
    one with an internal subset, stays untouched. Fixes epubcheck HTM-004 on the
    non-narrated content documents the pipeline never re-serializes.
    """
    for item in package.manifest:
        if item.media_type != XHTML_MEDIA_TYPE:
            continue

        path = container_root / item.href
        if not path.is_file():
            continue

        data = path.read_bytes()
        rewritten = _LEGACY_DOCTYPE.sub(b"<!DOCTYPE html>", data, count=1)
        if rewritten != data:
            path.write_bytes(rewritten)


def _document_title(path: Path) -> str:
    # Missing or malformed spine document degrades to the filename stem via the caller's
    # `... or stem`, never aborting the whole upgrade.
    root = parse_xml_file_or_none(path)
    if root is None:
        return ""

    title = root.find(f"{_XHTML_HEAD}/{_XHTML_TITLE}")

    return (title.text or "").strip() if title is not None else ""


def _free_nav_target(
    package: Package,
    base_dir: str,
    opf_root: XmlElement,
) -> tuple[str, str]:
    # XML ids are document-unique and _patch_opf injects the nav item into the OPF tree,
    # so the nav id must dodge every id in the OPF, not just manifest item ids: a
    # non-manifest <dc:identifier id="nav"> collides with the injected item too.
    ids = {item.item_id for item in package.manifest} | document_ids(opf_root)
    hrefs = {item.href for item in package.manifest}

    nav_id = _unused(_NAV_BASE_NAME, lambda candidate: candidate in ids)
    stem = _unused(
        _NAV_BASE_NAME,
        lambda candidate: posixpath.join(base_dir, f"{candidate}.xhtml") in hrefs,
    )

    return nav_id, f"{stem}.xhtml"


def _unused(base: str, taken: Callable[[str], bool]) -> str:
    if not taken(base):
        return base

    counter = 1
    while taken(f"{base}-{counter}"):
        counter += 1

    return f"{base}-{counter}"


def _resolve_ncx_src(ncx_dir: str, src: str) -> str:
    """Resolve an NCX `content/@src` to a container-relative path (fragment kept).

    Unlike `opf.resolve_within_container` this deliberately skips the container-escape
    check: the result is only re-relativized into the generated nav, never used for
    filesystem access.
    """
    path, fragment = split_fragment(src)
    resolved = posixpath.normpath(posixpath.join(ncx_dir, path))

    return join_fragment(resolved, fragment)


def _relativize(resolved_href: str, base_dir: str) -> str:
    path, fragment = split_fragment(resolved_href)
    relative = posixpath.relpath(path, base_dir or ".")

    return join_fragment(relative, fragment)
