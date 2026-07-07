"""Media Overlay SMIL emission, one document per content document.

`build_smil` is the shared, granularity-agnostic emitter: it ties each injected `<span>`
-- a sentence or a word -- to a clip in the chapter audio, regardless of how the caller
derived the clip bounds. This module only serializes what it is handed; it enforces no
gaplessness of its own.

In sentence mode the assembler supplies gapless clips (`clipEnd_n == clipBegin_{n+1}`),
so the overlay never flickers and never trips epubcheck's overlay-gap checks. In word
mode the bounds come from per-token timestamps and the caller drops zero-duration clips,
so gaplessness is not guaranteed there.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from lxml import etree

from spantrack.lxml_reading import EPUB_NS, qname

__all__ = ["SMIL_NS", "SyncUnit", "build_smil", "clock_value"]

SMIL_NS = "http://www.w3.org/ns/SMIL"


@dataclass(frozen=True, slots=True)
class SyncUnit:
    """One `<par>`: a span id (sentence or word) with its gapless clip bounds."""

    span_id: str
    clip_begin_s: float
    clip_end_s: float


def clock_value(seconds: float) -> str:
    """Format seconds as a SMIL full-clock value `H:MM:SS.mmm`.

    The same grammar serves `<audio>` clip bounds here and OPF `media:duration`.
    """
    # Round to whole milliseconds first, then split integer units, so a bound just under
    # a boundary carries (59.9996s -> 0:01:00.000, never the invalid 0:00:60.000 that
    # rounding a float seconds field after the divmod would emit).
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"


def build_smil(
    text_doc_href: str,
    audio_href: str,
    units: Sequence[SyncUnit],
) -> bytes:
    """Serialize a Media Overlay SMIL document: one `<par>` per sync unit.

    Args:
        text_doc_href: The content document href, used as the `<par>` text `src` base
            and the `<seq>` `epub:textref`.
        audio_href: The chapter audio href every `<par>`'s `<audio>` points at.
        units: The sync units in document order; their bounds are serialized verbatim,
            with no gaplessness check.

    Returns:
        The serialized SMIL document (UTF-8, XML-declared).

    Raises:
        ValueError: If `units` is empty -- a Media Overlay must carry at least one
            `<par>`, so a chapter with no narration must not get a SMIL at all.
    """
    if not units:
        msg = "cannot build a SMIL overlay from an empty sync-unit list"
        raise ValueError(msg)

    nsmap = cast("dict[str, str]", {None: SMIL_NS, "epub": EPUB_NS})
    smil = etree.Element(qname(SMIL_NS, "smil"), nsmap=nsmap)
    smil.set("version", "3.0")

    body = etree.SubElement(smil, qname(SMIL_NS, "body"))
    seq = etree.SubElement(body, qname(SMIL_NS, "seq"))
    seq.set(qname(EPUB_NS, "textref"), text_doc_href)

    for unit in units:
        par = etree.SubElement(seq, qname(SMIL_NS, "par"))
        text = etree.SubElement(par, qname(SMIL_NS, "text"))
        text.set("src", f"{text_doc_href}#{unit.span_id}")

        audio = etree.SubElement(par, qname(SMIL_NS, "audio"))
        audio.set("src", audio_href)
        audio.set("clipBegin", clock_value(unit.clip_begin_s))
        audio.set("clipEnd", clock_value(unit.clip_end_s))

    return etree.tostring(
        smil,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )
