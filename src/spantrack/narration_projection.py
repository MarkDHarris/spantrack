"""Project narration duration and output size from the selected documents.

This is a crude pre-synthesis estimate for `--dry-run`: it counts whitespace-collapsed
body text (not the precise block walk) and maps characters to audio seconds via the
performance-envelope rate, then audio seconds to encoded bytes via the bitrate. It
deliberately over-counts slightly (it does not yet honour the table/code skip rules) —
the figure is a ballpark shown before the user commits to a full run.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from spantrack.narration_set import Document
from spantrack.opf import XmlElement

__all__ = ["DocumentProjection", "SelectionProjection", "project"]

# The main assumption we make is that ~1.8m characters result in about 30 hours of
# narration (perf envelope) and we can calculate characters per audio second from that.
_CHARS_PER_AUDIO_SECOND = 16.7
_XHTML_NS = "http://www.w3.org/1999/xhtml"
_WHITESPACE = re.compile(r"\s+")
_BITRATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([km]?)\s*$", re.IGNORECASE)
_BITRATE_SUFFIXES = {"": 1, "k": 1_000, "m": 1_000_000}


@dataclass(frozen=True, slots=True)
class DocumentProjection:
    """The projected narratable character count for one document."""

    href: str
    characters: int


@dataclass(frozen=True, slots=True)
class SelectionProjection:
    """The projected scale of a narration run before synthesis."""

    documents: tuple[DocumentProjection, ...]
    total_characters: int
    duration_seconds: float
    size_bytes: int


def project(
    container_root: Path,
    documents: Sequence[Document],
    *,
    bitrate: str,
    speed: float = 1.0,
) -> SelectionProjection:
    """Estimate narration duration and encoded size for a selection.

    Args:
        container_root: The container root directory holding the documents.
        documents: The selected content documents, in narration order.
        bitrate: The audio bitrate as passed to the encoder (e.g. `64k`, `96000`).
        speed: The speech-speed multiplier; higher speeds shorten the duration.

    Returns:
        The projected character totals, audio duration, and byte size.

    Raises:
        ValueError: When `bitrate` is not a recognizable rate.
    """
    bitrate_bps = _parse_bitrate(bitrate)

    per_document: list[DocumentProjection] = []
    total = 0
    for document in documents:
        characters = _character_count(container_root / document.href)
        per_document.append(DocumentProjection(document.href, characters))
        total += characters

    duration = total / (_CHARS_PER_AUDIO_SECOND * speed) if total else 0.0
    size_bytes = round(duration * bitrate_bps / 8)

    return SelectionProjection(tuple(per_document), total, duration, size_bytes)


def _character_count(path: Path) -> int:
    if not path.is_file():
        return 0

    try:
        root = _parse(path)
    except etree.XMLSyntaxError:
        # A malformed document counts as zero here; the projection is a best-effort
        # preview, and segmentation is where a broken document hard-errors.
        return 0

    etree.strip_elements(
        root,
        _qn("script"),
        _qn("style"),
        with_tail=False,
    )
    body = root.find(_qn("body"))
    if body is None:
        return 0

    text = "".join(chunk for chunk in body.itertext() if isinstance(chunk, str))

    return len(_WHITESPACE.sub(" ", text).strip())


def _parse_bitrate(bitrate: str) -> int:
    match = _BITRATE.match(bitrate)
    if match is None:
        msg = f"invalid bitrate: {bitrate!r}"
        raise ValueError(msg)

    value, suffix = match.groups()

    return int(float(value) * _BITRATE_SUFFIXES[suffix.lower()])


def _parse(path: Path) -> XmlElement:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.fromstring(path.read_bytes(), parser)


def _qn(tag: str) -> str:
    return f"{{{_XHTML_NS}}}{tag}"
