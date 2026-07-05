"""Shared lxml reading helpers: XXE-hardened parsing and namespaced tag names.

The parser configuration is the security contract for every stage that reads untrusted
EPUB XML; keeping one authoritative factory here means a dropped flag can never regress
one module's hardening while the others stay safe.
"""

from pathlib import Path

from lxml import etree

__all__ = [
    "EPUB_NS",
    "OPF_NS",
    "XHTML_NS",
    "XmlElement",
    "document_ids",
    "hardened_parser",
    "parse_xml_file",
    "parse_xml_file_or_none",
    "qname",
    "split_fragment",
]

# lxml names its element type with a leading underscore by convention; alias it once so
# every XML-reading stage annotates against a public name instead of tripping
# reportPrivateUsage in each module.
type XmlElement = etree._Element  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

# The namespace URIs the content and package stages match in Clark notation. One
# authoritative copy so a downstream `{xhtml}body` lookup can never drift from the tags
# the walk actually emits.
XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"
OPF_NS = "http://www.idpf.org/2007/opf"


def hardened_parser() -> etree.XMLParser:
    """Return a fresh lxml parser hardened against XXE for untrusted EPUB input."""
    # Disable entity resolution, DTD loading, and network access so a malicious EPUB
    # cannot trigger XXE or SSRF while being parsed. A new parser per call keeps callers
    # independent (lxml parsers are not safe to share concurrently).
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)


def parse_xml_file(path: Path) -> XmlElement:
    """Read and parse an untrusted EPUB XML file with the hardened parser.

    Single-sources the read-and-parse of untrusted container XML so a change to the
    security posture (a size cap, streaming) lands in one place. Callers own their error
    policy; malformed input raises lxml's `etree.XMLSyntaxError`.
    """
    return etree.fromstring(path.read_bytes(), hardened_parser())


def parse_xml_file_or_none(path: Path) -> XmlElement | None:
    """Parse an untrusted EPUB XML file, or None when it is missing or malformed.

    The graceful counterpart to `parse_xml_file` for stages that treat an absent or
    not-well-formed document as a soft, skippable condition rather than an error.
    """
    if not path.is_file():
        return None

    try:
        return parse_xml_file(path)
    except etree.XMLSyntaxError:
        return None


def qname(namespace: str, tag: str) -> str:
    """Build a Clark-notation qualified tag name, e.g. `{namespace}tag`."""
    return f"{{{namespace}}}{tag}"


def document_ids(root: XmlElement) -> set[str]:
    """Collect every `id` attribute present anywhere in `root`'s tree.

    Both span-id minting (`sentence_spans`) and nav-id minting (`epub2_upgrade`) skip
    these so a generated id can never collide with a publisher's.
    """
    return {value for element in root.iter() if (value := element.get("id"))}


def split_fragment(href: str) -> tuple[str, str]:
    """Split an href into its path and its fragment (the part after `#`).

    One home for the `partition("#")` that manifest, guide, landmark, and NCX href
    handling would otherwise each repeat.

    Returns:
        A `(path, fragment)` pair; the fragment drops the leading `#`, and either part
        is an empty string when absent (`split_fragment('#top')` is `('', 'top')`).
    """
    path, _, fragment = href.partition("#")

    return path, fragment
