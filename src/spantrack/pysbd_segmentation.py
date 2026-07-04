"""Hardened sentence segmentation over pysbd.

pysbd is quarantined here because it needs handling no caller should repeat:

- It emits `SyntaxWarning`s at import (invalid escape sequences in its regex literals),
  suppressed around the import below.
- Under `clean=False` its `segment()` usually partitions the text into verbatim
  substrings, but on degenerate runs (tabs/NBSPs between repeated marks) it shuffles
  characters so the pieces no longer partition the source. `split_sentences` detects
  that (the pieces fail to reconstruct the input) and narrates the block whole rather
  than drop or misplace text.
- Its `char_span` offsets come from a first-match `finditer`, so a sentence that repeats
  earlier in the block gets the wrong span; we pass `char_span=False` and rebuild
  offsets from the segment lengths instead.
- Nothing in pysbd bounds its own runtime: dense punctuation/number patterns drive
  superlinear regex backtracking (pysbd #79). `_MAX_SEGMENTER_CHARS` is the sole guard
  against that — `char_span=False` does NOT avoid the backtracking path (pysbd runs the
  same processor pipeline regardless of the flag).

The single entry point returns char-offset ranges, so no other stage imports pysbd.
"""

import warnings
from typing import Protocol, cast

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from pysbd import Segmenter  # pyright: ignore[reportMissingTypeStubs]

__all__ = ["split_sentences"]

# Bound the input handed to pysbd. pysbd #79: dense punctuation/number patterns drive
# superlinear regex backtracking. A real prose paragraph is far below this ceiling, so
# anything above it is degenerate markup (or adversarial) and is safer narrated whole
# than fed to pysbd — a single sync unit never mis-splits and never hangs.
_MAX_SEGMENTER_CHARS = 10_000


class _SentenceSegmenter(Protocol):
    def segment(self, text_to_segment: str) -> list[str]: ...


# clean=False so pysbd never rewrites the text (offsets must map back onto the DOM).
# char_span=False so segment() returns plain strings and we rebuild contiguous offsets
# from their lengths; pysbd's own char_span offsets misplace a repeated sentence.
_segmenter: _SentenceSegmenter = cast(
    "_SentenceSegmenter",
    Segmenter(language="en", clean=False, char_span=False),
)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split text into sentence char-offset ranges, tight around non-whitespace.

    Args:
        text: The block's plain text to segment.

    Returns:
        Forward, non-overlapping `(start, end)` pairs where `text[start:end]` is one
        sentence with surrounding whitespace stripped; inter-sentence whitespace falls
        in the gaps between ranges. Empty or whitespace-only input yields no ranges.
    """
    if not text.strip():
        return []

    if len(text) > _MAX_SEGMENTER_CHARS:
        return _whole(text)

    segments = _segmenter.segment(text)
    if "".join(segments) != text:
        # pysbd shuffled characters, so its segments no longer partition the source;
        # mapping them would drop or misplace text. Narrate the block whole instead.
        return _whole(text)

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for sentence in segments:
        start, cursor = cursor, cursor + len(sentence)
        trimmed = _trim(text, start, cursor)
        if trimmed is not None:
            ranges.append(trimmed)

    return ranges


def _whole(text: str) -> list[tuple[int, int]]:
    """Return the whole text as one range, trimmed of surrounding whitespace."""
    trimmed = _trim(text, 0, len(text))
    return [trimmed] if trimmed is not None else []


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink `[start, end)` past leading/trailing whitespace; None if it empties."""
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return (start, end) if start < end else None
