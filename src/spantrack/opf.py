"""Read the EPUB package document (OPF): version, manifest, spine, and guide.

The parser is read-only and version-agnostic: it exposes the raw package structure of
both EPUB 2 and EPUB 3 containers so downstream stages (upgrade, narration-set) can
decide what to do. Media-overlay patching extends this module.
"""

import posixpath
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from spantrack.lxml_reading import (
    OPF_NS,
    XmlElement,
    parse_xml_file,
    qname,
    split_fragment,
)

__all__ = [
    "XHTML_MEDIA_TYPE",
    "GuideReference",
    "ManifestItem",
    "OpfError",
    "Package",
    "SpineItem",
    "read_package",
]

_CONTAINER_PATH = "META-INF/container.xml"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_MEDIA_TYPE = "application/oebps-package+xml"

# The EPUB 3 content-document media type. Shared with narration-set (which gates
# narration on this exact string) and the EPUB 2 upgrade (which stamps it onto the
# generated nav item), so one source keeps the selector and the emitter from drifting.
XHTML_MEDIA_TYPE = "application/xhtml+xml"


class OpfError(Exception):
    """A container is missing or malformed at the package-document level."""


@dataclass(frozen=True, slots=True)
class ManifestItem:
    """One `<manifest><item>` entry.

    Attributes:
        item_id: The manifest `id`.
        href: Path resolved relative to the container root (POSIX), e.g.
            `OEBPS/chapter1.xhtml`.
        media_type: The declared `media-type`.
        properties: The parsed `properties` token set (e.g. `{"nav"}`).
    """

    item_id: str
    href: str
    media_type: str
    properties: frozenset[str]


@dataclass(frozen=True, slots=True)
class SpineItem:
    """One `<spine><itemref>` entry.

    Attributes:
        idref: The referenced manifest `id`.
        linear: False when `linear="no"` (skippable auxiliary content).
    """

    idref: str
    linear: bool


@dataclass(frozen=True, slots=True)
class GuideReference:
    """One EPUB 2 `<guide><reference>` entry.

    Attributes:
        kind: The reference `type` (e.g. `cover`, `toc`, `text`).
        href: Path resolved relative to the container root (POSIX).
        title: The human-readable `title`, when present.
    """

    kind: str
    href: str
    title: str | None


@dataclass(frozen=True, slots=True)
class Package:
    """A parsed package document.

    Attributes:
        opf_path: Container-relative path to the OPF (POSIX).
        version: The `package/@version` string (e.g. `3.0`, `2.0`).
        manifest: Manifest items in document order.
        spine: Spine items in reading order.
        spine_toc: The `spine/@toc` idref (the NCX), when present.
        guide: EPUB 2 guide references; empty for EPUB 3 packages without a guide.
    """

    opf_path: str
    version: str
    manifest: tuple[ManifestItem, ...]
    spine: tuple[SpineItem, ...]
    spine_toc: str | None
    guide: tuple[GuideReference, ...]

    # id -> item index, built once so id lookups (one per spine item, in several stages)
    # never rescan the manifest. Excluded from eq/repr; first-writer-wins matches the
    # original linear search when two items share an id.
    _id_index: dict[str, ManifestItem] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the id -> item index once the manifest is set."""
        index: dict[str, ManifestItem] = {}
        for item in self.manifest:
            index.setdefault(item.item_id, item)

        object.__setattr__(self, "_id_index", index)

    def item_by_id(self, item_id: str) -> ManifestItem | None:
        """Return the manifest item with the given id, or None."""
        return self._id_index.get(item_id)

    def nav_item(self) -> ManifestItem | None:
        """Return the EPUB 3 navigation document (`properties="nav"`), or None."""
        for item in self.manifest:
            if "nav" in item.properties:
                return item

        return None

    def ncx_item(self) -> ManifestItem | None:
        """Return the NCX referenced by `spine/@toc`, or None."""
        if self.spine_toc is None:
            return None

        return self.item_by_id(self.spine_toc)


def read_package(container_root: Path) -> Package:
    """Parse the package document of an unzipped EPUB container.

    Args:
        container_root: The container root directory (holds `META-INF/`).

    Returns:
        The parsed package.

    Raises:
        OpfError: When `container.xml` or the OPF is missing or malformed.
    """
    opf_path = _locate_opf(container_root)
    root = _parse_xml(container_root / opf_path)
    base_dir = posixpath.dirname(opf_path)

    spine, spine_toc = _read_spine(root.find(qname(OPF_NS, "spine")))

    return Package(
        opf_path=opf_path,
        version=root.get("version", ""),
        manifest=_read_manifest(root, base_dir),
        spine=spine,
        spine_toc=spine_toc,
        guide=_read_guide(root, base_dir),
    )


def _locate_opf(container_root: Path) -> str:
    container_xml = container_root / _CONTAINER_PATH
    if not container_xml.is_file():
        err_msg = f"missing {_CONTAINER_PATH}"
        raise OpfError(err_msg)

    rootfiles = _parse_xml(container_xml).iterfind(
        f".//{qname(_CONTAINER_NS, 'rootfile')}",
    )
    full_path = _select_opf_rootfile(rootfiles)
    if not full_path:
        err_msg = "no OPF rootfile with a full-path in container.xml"
        raise OpfError(err_msg)

    return full_path


def _select_opf_rootfile(rootfiles: Iterator[XmlElement]) -> str | None:
    # Prefer the OPF by media type; fall back to the first rootfile with a full-path so
    # a stray non-OPF rootfile ordered first does not derail ingestion.
    fallback: str | None = None
    for rootfile in rootfiles:
        full_path = rootfile.get("full-path")
        if not full_path:
            continue

        if rootfile.get("media-type") == _OPF_MEDIA_TYPE:
            return full_path

        fallback = fallback or full_path

    return fallback


def _read_manifest(root: XmlElement, base_dir: str) -> tuple[ManifestItem, ...]:
    manifest = root.find(qname(OPF_NS, "manifest"))
    if manifest is None:
        return ()

    items: list[ManifestItem] = []
    for item in manifest.iterfind(qname(OPF_NS, "item")):
        item_id = item.get("id")
        href = item.get("href")
        if not item_id or not href:
            continue

        items.append(
            ManifestItem(
                item_id=item_id,
                href=_require_within_container(base_dir, href),
                media_type=item.get("media-type", ""),
                properties=frozenset(item.get("properties", "").split()),
            ),
        )

    return tuple(items)


def _read_spine(
    spine: XmlElement | None,
) -> tuple[tuple[SpineItem, ...], str | None]:
    if spine is None:
        return (), None

    items = [
        SpineItem(idref=idref, linear=itemref.get("linear", "yes") != "no")
        for itemref in spine.iterfind(qname(OPF_NS, "itemref"))
        if (idref := itemref.get("idref"))
    ]

    return tuple(items), spine.get("toc")


def _read_guide(root: XmlElement, base_dir: str) -> tuple[GuideReference, ...]:
    guide = root.find(qname(OPF_NS, "guide"))
    if guide is None:
        return ()

    references: list[GuideReference] = []
    for ref in guide.iterfind(qname(OPF_NS, "reference")):
        kind = ref.get("type")
        href = ref.get("href")
        if not kind or not href:
            continue

        references.append(
            GuideReference(
                kind=kind,
                href=_require_within_container(base_dir, href),
                title=ref.get("title"),
            ),
        )

    return tuple(references)


def _parse_xml(path: Path) -> XmlElement:
    try:
        return parse_xml_file(path)
    except etree.XMLSyntaxError as exc:
        err_msg = f"malformed XML in {path.name}"
        raise OpfError(err_msg) from exc


def resolve_within_container(base_dir: str, href: str) -> str | None:
    """Resolve an href to a container-relative POSIX path, or None if it escapes.

    `base_dir` is the directory the href resolves against — usually the OPF's own
    directory, but also a nav document's directory when resolving landmark hrefs.
    Manifest/guide/landmark hrefs are a distinct injection surface from archive member
    names (which `epub_container` already guards): a benign archive can carry an OPF
    pointing outside the container. Returns None for an href that is empty, absolute,
    backslash-separated, or escapes the container root; the fragment is dropped.
    """
    path = split_fragment(href)[0]
    if not path or "\\" in path or posixpath.isabs(path):
        return None

    resolved = posixpath.normpath(posixpath.join(base_dir, path))
    if resolved == ".." or resolved.startswith("../"):
        return None

    return resolved


def _require_within_container(base_dir: str, href: str) -> str:
    """Resolve an OPF-relative href within the container, or raise on escape.

    Unlike `resolve_within_container`, this fails loud with `OpfError` rather than
    returning None, and preserves the href fragment.
    """
    resolved = resolve_within_container(base_dir, href)
    if resolved is None:
        err_msg = f"href outside the container: {href!r}"
        raise OpfError(err_msg)

    fragment = split_fragment(href)[1]

    return f"{resolved}#{fragment}" if fragment else resolved
