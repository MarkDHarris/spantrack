"""Shared lxml reading helpers: XXE-hardened parsing and namespaced tag names.

The parser configuration is the security contract for every stage that reads untrusted
EPUB XML; keeping one authoritative factory here means a dropped flag can never regress
one module's hardening while the others stay safe.
"""

from lxml import etree

__all__ = ["hardened_parser", "qname"]


def hardened_parser() -> etree.XMLParser:
    """Return a fresh lxml parser hardened against XXE for untrusted EPUB input."""
    # Disable entity resolution, DTD loading, and network access so a malicious EPUB
    # cannot trigger XXE or SSRF while being parsed. A new parser per call keeps callers
    # independent (lxml parsers are not safe to share concurrently).
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)


def qname(namespace: str, tag: str) -> str:
    """Build a Clark-notation qualified tag name, e.g. `{namespace}tag`."""
    return f"{{{namespace}}}{tag}"
