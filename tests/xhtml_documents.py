"""Build XHTML content documents for tests from a `<body>` inner fragment.

One canonical wrapper (XML prolog + `<!DOCTYPE html>` + XHTML-namespaced `<html>`) so a
parser-config or wrapper change lands in a single place instead of each test module's
own private copy.
"""

from pathlib import Path

from lxml import etree

from spantrack.lxml_reading import XHTML_NS, XmlElement, hardened_parser

_PROLOG = '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'


def xhtml_document(body: str, *, title: str = "t") -> str:
    """Return a well-formed XHTML document wrapping `body` as its `<body>` contents."""
    return (
        f"{_PROLOG}"
        f'<html xmlns="{XHTML_NS}"><head><title>{title}</title></head>'
        f"<body>{body}</body></html>"
    )


def parse_body(body: str) -> XmlElement:
    """Parse a `<body>` inner fragment into a hardened XHTML element tree."""
    return etree.fromstring(xhtml_document(body).encode(), hardened_parser())


def write_xhtml(root: Path, name: str, body: str, *, title: str = "t") -> Path:
    """Write `xhtml_document(body)` to `root/name`, creating `root`; return the path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(xhtml_document(body, title=title), encoding="utf-8")
    return path
