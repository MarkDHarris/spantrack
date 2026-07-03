"""Read the EPUB package document (OPF): version, manifest, spine, and guide.

The parser is read-only and version-agnostic: it exposes the raw package structure of
both EPUB 2 and EPUB 3 containers so downstream stages (upgrade, narration-set) can
decide what to do. Media-overlay patching extends this module.
"""

import posixpath
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

# lxml names its element type with a leading underscore by convention; alias it once so
# annotations reference a public name instead of tripping reportPrivateUsage repeatedly.
type XmlElement = etree._Element  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "GuideReference",
    "ManifestItem",
    "OpfError",
    "Package",
    "SpineItem",
    "read_package",
]

_CONTAINER_PATH = "META-INF/container.xml"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_NS = "http://www.idpf.org/2007/opf"
_OPF_MEDIA_TYPE = "application/oebps-package+xml"


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
        unique_identifier: The `unique-identifier` id attribute, when present.
        manifest: Manifest items in document order.
        spine: Spine items in reading order.
        spine_toc: The `spine/@toc` idref (the NCX), when present.
        guide: EPUB 2 guide references; empty for EPUB 3 packages without a guide.
    """

    opf_path: str
    version: str
    unique_identifier: str | None
    manifest: tuple[ManifestItem, ...]
    spine: tuple[SpineItem, ...]
    spine_toc: str | None
    guide: tuple[GuideReference, ...]

    def item_by_id(self, item_id: str) -> ManifestItem | None:
        """Return the manifest item with the given id, or None."""
        for item in self.manifest:
            if item.item_id == item_id:
                return item

        return None

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

    spine, spine_toc = _read_spine(root.find(_qn(_OPF_NS, "spine")))

    return Package(
        opf_path=opf_path,
        version=root.get("version", ""),
        unique_identifier=root.get("unique-identifier"),
        manifest=tuple(_read_manifest(root, base_dir)),
        spine=spine,
        spine_toc=spine_toc,
        guide=tuple(_read_guide(root, base_dir)),
    )


def _locate_opf(container_root: Path) -> str:
    container_xml = container_root / _CONTAINER_PATH
    if not container_xml.is_file():
        err_msg = f"missing {_CONTAINER_PATH}"
        raise OpfError(err_msg)

    rootfiles = _parse_xml(container_xml).iterfind(
        f".//{_qn(_CONTAINER_NS, 'rootfile')}",
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


def _read_manifest(root: XmlElement, base_dir: str) -> Iterator[ManifestItem]:
    manifest = root.find(_qn(_OPF_NS, "manifest"))
    if manifest is None:
        return

    for item in manifest.iterfind(_qn(_OPF_NS, "item")):
        item_id = item.get("id")
        href = item.get("href")
        if not item_id or not href:
            continue

        yield ManifestItem(
            item_id=item_id,
            href=_resolve(base_dir, href),
            media_type=item.get("media-type", ""),
            properties=frozenset(item.get("properties", "").split()),
        )


def _read_spine(
    spine: XmlElement | None,
) -> tuple[tuple[SpineItem, ...], str | None]:
    if spine is None:
        return (), None

    items = [
        SpineItem(idref=idref, linear=itemref.get("linear", "yes") != "no")
        for itemref in spine.iterfind(_qn(_OPF_NS, "itemref"))
        if (idref := itemref.get("idref"))
    ]

    return tuple(items), spine.get("toc")


def _read_guide(root: XmlElement, base_dir: str) -> Iterator[GuideReference]:
    guide = root.find(_qn(_OPF_NS, "guide"))
    if guide is None:
        return

    for ref in guide.iterfind(_qn(_OPF_NS, "reference")):
        kind = ref.get("type")
        href = ref.get("href")
        if not kind or not href:
            continue

        yield GuideReference(
            kind=kind,
            href=_resolve(base_dir, href),
            title=ref.get("title"),
        )


def _parse_xml(path: Path) -> XmlElement:
    # Disable entity resolution and network access to neutralize XXE in EPUB inputs.
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        return etree.fromstring(path.read_bytes(), parser)
    except etree.XMLSyntaxError as exc:
        err_msg = f"malformed XML in {path.name}"
        raise OpfError(err_msg) from exc


def _qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def _resolve(base_dir: str, href: str) -> str:
    """Resolve an OPF-relative href to a container-relative POSIX path.

    Manifest hrefs are a distinct injection surface from archive member names (which
    `epub_container` already guards): a benign archive can carry an OPF pointing outside
    the container. Reject anything that is absolute or escapes the root.

    Raises:
        OpfError: When the href is absolute, backslash-separated, or escapes the root.
    """
    path, _, fragment = href.partition("#")
    if not path or "\\" in path or posixpath.isabs(path):
        err_msg = f"absolute or non-relative href outside the container: {href!r}"
        raise OpfError(err_msg)

    resolved = posixpath.normpath(posixpath.join(base_dir, path))
    if resolved == ".." or resolved.startswith("../"):
        err_msg = f"href escapes the container root: {href!r}"
        raise OpfError(err_msg)

    return f"{resolved}#{fragment}" if fragment else resolved
