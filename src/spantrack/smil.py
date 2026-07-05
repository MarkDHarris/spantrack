"""Media Overlay SMIL emission, one document per content document.

`build_smil` is the shared, granularity-agnostic emitter: it ties each injected `<span>`
-- a sentence or a word -- to a clip in the chapter audio, regardless of how the caller
derived the clip bounds. This module only serializes what it is handed and requires the
clips be gapless (`clipEnd_n == clipBegin_{n+1}`).

In sentence mode the assembler's gapless intervals meet that requirement, so the overlay
never flickers and never trips epubcheck's overlay-gap checks. In word mode the bounds
come from per-token timestamps and the caller drops zero-duration clips, so exact
gaplessness is not guaranteed there (the accepted experimental word-mode drift).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from lxml import etree

from spantrack.lxml_reading import EPUB_NS, qname

__all__ = ["SMIL_NS", "SmilSentence", "build_smil", "clock_value"]

SMIL_NS = "http://www.w3.org/ns/SMIL"


@dataclass(frozen=True, slots=True)
class SmilSentence:
    """One `<par>`'s input: its span id and gapless clip bounds, in seconds."""

    span_id: str
    clip_begin_s: float
    clip_end_s: float


def clock_value(seconds: float) -> str:
    """Format seconds as a SMIL full-clock value `H:MM:SS.mmm`.

    The same grammar serves `<audio>` clip bounds here and OPF `media:duration`.
    """
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:06.3f}"


def build_smil(
    text_doc_href: str,
    audio_href: str,
    sentences: Sequence[SmilSentence],
) -> bytes:
    """Serialize a sentence-granularity Media Overlay SMIL document.

    Args:
        text_doc_href: The content document href, used as the `<par>` text `src` base
            and the `<seq>` `epub:textref`.
        audio_href: The chapter audio href every `<par>`'s `<audio>` points at.
        sentences: The sync units in document order; their clip bounds must be gapless.

    Returns:
        The serialized SMIL document (UTF-8, XML-declared).

    Raises:
        ValueError: If `sentences` is empty -- a Media Overlay must carry at least one
            `<par>`, so a chapter with no narration must not get a SMIL at all.
    """
    if not sentences:
        msg = "cannot build a SMIL overlay from an empty sentence list"
        raise ValueError(msg)

    nsmap = cast("dict[str, str]", {None: SMIL_NS, "epub": EPUB_NS})
    smil = etree.Element(qname(SMIL_NS, "smil"), nsmap=nsmap)
    smil.set("version", "3.0")

    body = etree.SubElement(smil, qname(SMIL_NS, "body"))
    seq = etree.SubElement(body, qname(SMIL_NS, "seq"))
    seq.set(qname(EPUB_NS, "textref"), text_doc_href)

    for sentence in sentences:
        par = etree.SubElement(seq, qname(SMIL_NS, "par"))
        text = etree.SubElement(par, qname(SMIL_NS, "text"))
        text.set("src", f"{text_doc_href}#{sentence.span_id}")

        audio = etree.SubElement(par, qname(SMIL_NS, "audio"))
        audio.set("src", audio_href)
        audio.set("clipBegin", clock_value(sentence.clip_begin_s))
        audio.set("clipEnd", clock_value(sentence.clip_end_s))

    return etree.tostring(
        smil, encoding="utf-8", xml_declaration=True, pretty_print=True
    )
