"""Refuse a content document whose root is not in the XHTML namespace.

Every XML-reading stage downstream of narration-set selection assumes content documents
live in the XHTML namespace (`{http://www.w3.org/1999/xhtml}`): the block walk and the
projection match Clark-notation XHTML tags. A malformed EPUB 2 content document whose
`<html>` omits `xmlns=".../1999/xhtml"` puts every element in no namespace, so those
stages match nothing and silently narrate the chapter as empty audio.

Namespaces inherit down the tree, so the root element's namespace decides the whole
document. This guard inspects the root once, after selection and before segmentation,
and fails loud rather than emit a silently-empty audiobook. Fixing the block walk alone
was rejected: it would narrate a document that projection still counts as zero
characters, trading one inconsistency for another.

Genuinely foreign content (inline SVG/MathML, or a descendant that re-declares a
non-XHTML default namespace) lives under an XHTML root and is correctly excluded by the
block walk, so it never reaches this guard. Only a document whose root is non-XHTML —
the shape that breaks every downstream stage — is rejected here.
"""

from collections.abc import Sequence
from pathlib import Path

from lxml import etree

from spantrack.lxml_reading import XHTML_NS, parse_xml_file_or_none
from spantrack.narration_set import Document

__all__ = ["ContentNamespaceError", "require_xhtml_namespace"]


class ContentNamespaceError(Exception):
    """A selected content document's root element is not in the XHTML namespace."""


def require_xhtml_namespace(
    container_root: Path,
    documents: Sequence[Document],
) -> None:
    """Reject any selected content document whose root is not XHTML-namespaced.

    Args:
        container_root: The container root directory holding the documents.
        documents: The selected content documents, in narration order.

    Raises:
        ContentNamespaceError: When a document's root element is in a namespace other
            than XHTML — bare (no namespace) or genuinely foreign — which every
            downstream XML stage would otherwise narrate as empty audio.
    """
    for document in documents:
        root = parse_xml_file_or_none(container_root / document.href)
        if root is None:
            # Missing file or not-well-formed XML: a distinct malformation surfaced by
            # its own stage (segmentation hard-errors; projection counts zero). This
            # guard owns exactly one concern, so it leaves those alone.
            continue

        root_name = etree.QName(root)
        if root_name.namespace != XHTML_NS:
            found = root_name.namespace or "none"
            msg = (
                f"{document.href}: root <{root_name.localname}> is not in the XHTML "
                f"namespace (found: {found}); cannot narrate"
            )
            raise ContentNamespaceError(msg)
