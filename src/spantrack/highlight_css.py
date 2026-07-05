"""Highlight stylesheet: the active-class CSS and its injection into content docs.

The reader adds the `media:active-class` class to the sentence `<span>` being read;
this stylesheet is the only thing that styles it. It ships a light rule plus a
`prefers-color-scheme: dark` variant so the highlight stays legible in either theme.

Adding the stylesheet file and a `<head>` `<link>` to each chapter is the one markup
change the layout-preservation guarantee permits beyond the injected spans. Placement of
the file and the per-chapter relative href are the pipeline's job; this module owns the
CSS itself and the idempotent `<head>` link.
"""

from pathlib import Path

from lxml import etree

from spantrack.lxml_reading import XHTML_NS, XmlElement, qname

__all__ = [
    "ACTIVE_CLASS",
    "link_into_head",
    "stylesheet",
    "write_stylesheet",
]

# The class a reader toggles on the active sentence span; also the media:active-class.
ACTIVE_CLASS = "-spantrack-active"

_SPANTRACK_DIR = "spantrack"
_STYLESHEET_NAME = "highlight.css"

_STYLESHEET = f""".{ACTIVE_CLASS} {{
  background-color: #fff2a8;
  color: #1a1a1a;
}}

@media (prefers-color-scheme: dark) {{
  .{ACTIVE_CLASS} {{
    background-color: #6b5800;
    color: #f5f5f5;
  }}
}}
"""


def stylesheet() -> str:
    """Return the highlight stylesheet (light rule + dark-scheme variant)."""
    return _STYLESHEET


def write_stylesheet(base_dir: Path) -> str:
    """Write the highlight stylesheet under `base_dir/spantrack/`.

    Returns:
        The stylesheet's path relative to `base_dir`, as a POSIX href for the OPF
        manifest (the per-chapter `<link>` href is re-relativized by the caller).
    """
    path = base_dir / _SPANTRACK_DIR / _STYLESHEET_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STYLESHEET, encoding="utf-8")

    return f"{_SPANTRACK_DIR}/{_STYLESHEET_NAME}"


def link_into_head(root: XmlElement, href: str) -> None:
    """Add a `<link rel="stylesheet">` for `href` to the document's `<head>`.

    Idempotent: a re-run with the same href adds nothing, so re-narrating a chapter
    never duplicates the link.

    Raises:
        ValueError: If the document has no `<head>` element.
    """
    head = root.find(qname(XHTML_NS, "head"))
    if head is None:
        msg = "content document has no <head> to link the highlight stylesheet into"
        raise ValueError(msg)

    for link in head.iterfind(qname(XHTML_NS, "link")):
        if link.get("rel") == "stylesheet" and link.get("href") == href:
            return

    link = etree.SubElement(head, qname(XHTML_NS, "link"))
    link.set("rel", "stylesheet")
    link.set("type", "text/css")
    link.set("href", href)
