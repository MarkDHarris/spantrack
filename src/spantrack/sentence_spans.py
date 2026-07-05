"""Sentence-span injection: walk a content document and wrap sentences in `<span>`s.

`narratable_blocks` walks a content document for the leaf block elements whose text
becomes narration. Only leaves are yielded — a `blockquote` or `li` that wraps its own
`p` contributes the inner `p`, never both, so no text is narrated twice. Subtrees that
are not prose (`table`, `pre`, `code`, etc.) are pruned entirely.

`extract_block_text` flattens one block into its raw narratable text plus a map from a
character offset back to the DOM string it came from (an element's `.text` or a child's
`.tail`). Segmentation runs on the raw text so its offsets map straight onto the DOM, so
the next stage can wrap sentence runs in `<span>`s without disturbing inline markup.
"""

import html
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count, pairwise

from lxml import etree

from spantrack.audio_assembly import BreakAfter
from spantrack.lxml_reading import XHTML_NS, XmlElement, qname
from spantrack.pysbd_segmentation import split_sentences
from spantrack.text_normalization import collapse_whitespace
from spantrack.word_alignment import segment_words

__all__ = [
    "BlockText",
    "InjectedSpan",
    "InjectedWord",
    "TextPiece",
    "WordSentence",
    "extract_block_text",
    "inject",
    "inject_block",
    "inject_words",
    "narratable_blocks",
]

_BLOCK_TAGS = frozenset(
    qname(XHTML_NS, tag)
    for tag in (
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "figcaption",
        "dt",
        "dd",
    )
)
_SKIP_CONTAINERS = frozenset(
    qname(XHTML_NS, tag) for tag in ("table", "pre", "code", "script", "style", "nav")
)


@dataclass(frozen=True, slots=True)
class TextPiece:
    """One contiguous run of a block's raw text, tied to the DOM string it came from.

    `is_tail` selects whether `string` is `node.text` (False) or `node.tail` (True);
    `offset` is where the run starts in the block's raw text. An entity-reference node
    yields an atomic piece whose `string` is the resolved character(s), identified by
    `node.tag is etree.Entity and not is_tail` (a later stage wraps the node rather than
    slicing text); the entity's own trailing text is an ordinary `is_tail` piece.
    """

    node: XmlElement
    is_tail: bool
    offset: int
    string: str


@dataclass(frozen=True, slots=True)
class BlockText:
    """A block's flattened narratable text with an offset-to-DOM map."""

    text: str
    pieces: tuple[TextPiece, ...]

    def locate(self, offset: int) -> tuple[TextPiece, int]:
        """Resolve a character offset to its piece and the local offset within it.

        Args:
            offset: A character offset into `text`, in `range(len(text))`.

        Returns:
            The piece covering `offset` and the offset within that piece's string.

        Raises:
            IndexError: When `offset` is outside `range(len(text))`.
        """
        for piece in self.pieces:
            if piece.offset <= offset < piece.offset + len(piece.string):
                return piece, offset - piece.offset

        msg = f"offset {offset} is outside the block text"
        raise IndexError(msg)


def narratable_blocks(root: XmlElement) -> list[XmlElement]:
    """Return the leaf narratable block elements under root, in document order."""
    return _blocks_under(root)


def _blocks_under(element: XmlElement) -> list[XmlElement]:
    if element.tag in _SKIP_CONTAINERS:
        return []

    nested = [block for child in element for block in _blocks_under(child)]
    if nested:
        # A container of blocks (e.g. blockquote > p) is never itself a sync unit; its
        # leaf descendants are, so its own direct text is intentionally not narrated.
        return nested

    if element.tag in _BLOCK_TAGS and _has_narratable_text(element):
        return [element]

    return []


def extract_block_text(block: XmlElement) -> BlockText:
    """Flatten a block into its raw narratable text and an offset-to-DOM map."""
    pieces: list[TextPiece] = []
    offset = 0
    for node, is_tail, string in _text_pieces(block):
        pieces.append(
            TextPiece(node=node, is_tail=is_tail, offset=offset, string=string)
        )
        offset += len(string)

    return BlockText(
        text="".join(piece.string for piece in pieces), pieces=tuple(pieces)
    )


def _has_narratable_text(element: XmlElement) -> bool:
    # A block whose only text sits inside a skipped subtree (an inline `code`, a nested
    # `table`) carries no prose of its own, so it is not a sync unit.
    return any(string.strip() for _, _, string in _text_pieces(element))


def _text_pieces(element: XmlElement) -> Iterator[tuple[XmlElement, bool, str]]:
    """Yield `(node, is_tail, string)` for the element's prose text and tails, in order.

    Skips non-prose subtrees so a block's narratable text and its DOM offset map come
    from the same traversal — see the module docstring for what counts as prose.
    """
    if element.text:
        yield (element, False, element.text)

    for child in element:
        tag = child.tag
        if tag is etree.Entity:
            # A named entity (e.g. &rsquo;) is a leaf node whose narration is its
            # resolved character; the node stays untouched so output bytes are kept.
            char = _entity_char(child)
            if char:
                yield (child, False, char)
        elif not callable(tag) and tag not in _SKIP_CONTAINERS:
            # Comment/PI bodies are invisible metadata (callable tag), so recurse only
            # into real prose elements outside skipped subtrees.
            yield from _text_pieces(child)

        # A tail is real prose in the parent's flow even after a skipped or non-element
        # child (comment, entity, table), so it is always kept.
        if child.tail:
            yield (child, True, child.tail)


def _entity_char(entity: XmlElement) -> str:
    """Resolve a named-entity reference to its character(s), or "" if it cannot be.

    `html.unescape` expands the standard HTML named entities a real EPUB declares in its
    internal DTD subset (`rsquo`, `mdash`, `hellip`, ...) — atomic single characters. A
    custom or entity-aliased reference stays its literal `&name;` (libxml2 does not
    expand a nested reference inside a declaration's content); rather than narrate that
    raw markup it is dropped, returning "".
    """
    resolved = html.unescape(entity.text or "")
    return resolved if "&" not in resolved else ""


@dataclass(frozen=True, slots=True)
class _TextRun:
    """A sliceable top-level text run — the block's `.text` or a child's `.tail`."""

    text: str
    start: int

    @property
    def end(self) -> int:
        """Exclusive end offset of this run in the block's raw text."""
        return self.start + len(self.text)


@dataclass(frozen=True, slots=True)
class _ChildRun:
    """An atomic top-level child element spanning `[start, end)` of the raw text."""

    element: XmlElement
    start: int
    end: int


type _Segment = _TextRun | _ChildRun


def inject_block(block: XmlElement, id_gen: Iterator[str]) -> list[tuple[str, str]]:
    """Wrap each sentence run in the block in a `<span id>`, preserving inline markup.

    Sentences whose boundary falls inside an inline element merge into one sync unit, so
    a span never splits an element. The block is mutated in place: its inline elements
    and entity-reference nodes are detached and re-emitted (moved, not copied) into the
    span containers, so a caller must not reuse the old child structure afterward. A
    block with no narratable sentence is left untouched.

    Args:
        block: A leaf block element (from `narratable_blocks`).
        id_gen: Yields the span ids to assign, in order (e.g. `s00001`, `s00002`, ...);
            it must yield at least one id per sync unit.

    Returns:
        `(span_id, display_text)` for each injected span, in document order, with
        `display_text` whitespace-collapsed.

    Raises:
        StopIteration: If `id_gen` runs out before every span id is pulled. The pull
            happens before any mutation, so the block is left untouched (atomic no-op).
    """
    raw_text = extract_block_text(block).text
    ranges = split_sentences(raw_text)
    if not ranges:
        return []

    segments = _segments(block)
    units = _merge_units(ranges, segments)

    # Pull every span id up front so an exhausted id_gen is a no-op, never a partly
    # rewritten block; every unit gets exactly one span.
    span_ids = [next(id_gen) for _ in units]
    _rebuild(block, segments, units, span_ids)

    return [
        (span_ids[index], collapse_whitespace(raw_text[start:end]))
        for index, (start, end) in enumerate(units)
    ]


_HEADING_TAGS = frozenset(
    qname(XHTML_NS, tag) for tag in ("h1", "h2", "h3", "h4", "h5", "h6")
)


@dataclass(frozen=True, slots=True)
class InjectedSpan:
    """One injected sentence span with the break that follows it.

    This is the partial `SentenceUnit` (`doc_href`/`normalized_text` are attached
    downstream): `span_id` addresses the `<span>`, `display_text` is whitespace-
    collapsed, and `break_after` — `heading` when the span ends a heading block,
    `paragraph` when it ends any other block, `sentence` for a non-final span within a
    block — drives the assembler's inter-sentence gap.
    """

    span_id: str
    display_text: str
    break_after: BreakAfter


def inject(root: XmlElement) -> list[InjectedSpan]:
    """Inject sentence spans across a whole content document, in document order.

    Walks the document's narratable blocks, wraps each sentence run in a
    `<span id="sNNNNN">` with an id unique and sequential within the document, and
    classifies each span's following break from its block. Mutates the tree in place.

    Args:
        root: The parsed content-document root (e.g. its `<html>` element).

    Returns:
        The injected spans in document order, each with its display text and
        `break_after`.
    """
    # Skip any id already used in the document so the injected ids stay unique document-
    # wide (a hard XHTML requirement, and media overlays resolve spans by id).
    id_gen = _fresh_ids(_document_ids(root))
    injected: list[InjectedSpan] = []
    for block in narratable_blocks(root):
        spans = inject_block(block, id_gen)
        # Only the block's last span carries the block-level break: a heading closes
        # with a heading gap, any other block a paragraph gap, interior spans plainly.
        block_break = (
            BreakAfter.HEADING if block.tag in _HEADING_TAGS else BreakAfter.PARAGRAPH
        )

        for index, (span_id, display_text) in enumerate(spans):
            last = index == len(spans) - 1
            injected.append(
                InjectedSpan(
                    span_id=span_id,
                    display_text=display_text,
                    break_after=block_break if last else BreakAfter.SENTENCE,
                ),
            )

    return injected


def _document_ids(root: XmlElement) -> set[str]:
    return {value for element in root.iter() if (value := element.get("id"))}


def _fresh_ids(existing_ids: set[str]) -> Iterator[str]:
    for number in count(1):
        candidate = f"s{number:05d}"
        if candidate not in existing_ids:
            yield candidate


def _segments(block: XmlElement) -> list[_Segment]:
    segments: list[_Segment] = []
    cursor = 0
    if block.text:
        segments.append(_TextRun(text=block.text, start=cursor))
        cursor += len(block.text)

    for child in block:
        content = _child_narratable_text(child)
        segments.append(
            _ChildRun(element=child, start=cursor, end=cursor + len(content))
        )
        cursor += len(content)
        if child.tail:
            segments.append(_TextRun(text=child.tail, start=cursor))
            cursor += len(child.tail)

    return segments


def _child_narratable_text(child: XmlElement) -> str:
    # Match how _text_pieces counts this child at block level: an entity is its resolved
    # character, a comment/PI/skip-container contributes nothing, else its own subtree.
    tag = child.tag
    if tag is etree.Entity:
        return _entity_char(child)

    if callable(tag) or tag in _SKIP_CONTAINERS:
        return ""

    return "".join(string for _, _, string in _text_pieces(child))


def _merge_units(
    ranges: list[tuple[int, int]],
    segments: list[_Segment],
) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    start, end = ranges[0]
    for next_start, next_end in ranges[1:]:
        left = _segment_at(end - 1, segments)
        right = _segment_at(next_start, segments)
        if left is right and isinstance(left, _ChildRun):
            # The boundary falls inside a single inline element; splitting would break
            # it, so the two sentences become one sync unit.
            end = next_end
        else:
            units.append((start, end))
            start, end = next_start, next_end

    units.append((start, end))
    return units


def _segment_at(offset: int, segments: list[_Segment]) -> _Segment | None:
    for segment in segments:
        if segment.start <= offset < segment.end:
            return segment

    return None


def _rebuild(
    block: XmlElement,
    segments: list[_Segment],
    units: list[tuple[int, int]],
    span_ids: list[str],
) -> None:
    # Detach the original content, then re-emit it into spans (sentence runs) and loose
    # slots (inter-sentence whitespace, non-narratable children), preserving order.
    for child in list(block):
        child.tail = None  # tails are re-emitted from their _TextRun segments
        block.remove(child)

    block.text = None

    block_emitter = _Emitter(block)
    span_emitters: dict[int, _Emitter] = {}

    def span_for(index: int) -> _Emitter:
        if index not in span_emitters:
            span = block.makeelement(qname(XHTML_NS, "span"), {"id": span_ids[index]})
            block_emitter.element(span)
            span_emitters[index] = _Emitter(span)

        return span_emitters[index]

    for segment in segments:
        if isinstance(segment, _TextRun):
            for run_start, run_end, index in _classify_text(segment, units):
                piece = segment.text[
                    run_start - segment.start : run_end - segment.start
                ]
                (block_emitter if index is None else span_for(index)).text(piece)
        else:
            index = _child_unit(segment, units)
            (block_emitter if index is None else span_for(index)).element(
                segment.element
            )


def _classify_text(
    segment: _TextRun,
    units: list[tuple[int, int]],
) -> Iterator[tuple[int, int, int | None]]:
    start = segment.start
    end = segment.end

    cuts = sorted({start, end} | {b for unit in units for b in unit if start < b < end})
    for run_start, run_end in pairwise(cuts):
        yield run_start, run_end, _unit_at(run_start, units)


def _child_unit(segment: _ChildRun, units: list[tuple[int, int]]) -> int | None:
    # The child joins the one unit its content overlaps. A child's own leading/trailing
    # whitespace can reach past the trimmed sentence range, so classify by overlap of
    # `[start, end)` with a unit — not by an endpoint, which may be whitespace in an
    # inter-sentence gap. A zero-width child reduces to strict interior; a whitespace-
    # only child overlaps no unit and stays loose.
    for index, (u_start, u_end) in enumerate(units):
        if segment.start < u_end and u_start < segment.end:
            return index

    return None


def _unit_at(offset: int, units: list[tuple[int, int]]) -> int | None:
    for index, (u_start, u_end) in enumerate(units):
        if u_start <= offset < u_end:
            return index

    return None


class _Emitter:
    """Append text and elements to a parent, routing text to `.text` or last tail."""

    def __init__(self, parent: XmlElement) -> None:
        self._parent = parent
        self._last: XmlElement | None = None

    def text(self, string: str) -> None:
        if not string:
            return

        if self._last is None:
            self._parent.text = (self._parent.text or "") + string
        else:
            self._last.tail = (self._last.tail or "") + string

    def element(self, element: XmlElement) -> None:
        self._parent.append(element)
        self._last = element


@dataclass(frozen=True, slots=True)
class InjectedWord:
    """One injected word span: its id, display text, and range in the sentence text.

    `start`/`end` index into the owning `WordSentence.display_text`, so they are exactly
    the `source_range` a token mapped to this word reports.
    """

    span_id: str
    display_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class WordSentence:
    """One sentence's word spans and the sentence-level break that follows it.

    Word mode keeps the sentence as the synthesis/assembly unit (so `break_after` still
    drives the inter-sentence gap) while each source word gets its own `<span>` for
    highlighting. `span_id` is the sentence base id `sNNNNN`; word ids are `sNNNNN-wK`.
    Words inside a single inline element are wrapped together as one unit -- they cannot
    be split without breaking the element, an experimental-mode limitation.
    """

    span_id: str
    display_text: str
    break_after: BreakAfter
    words: tuple[InjectedWord, ...]


def inject_words(root: XmlElement) -> list[WordSentence]:
    """Inject word-level spans across a content document, in document order.

    Like `inject`, but wraps each source word in its own `<span id="sNNNNN-wK">` (K the
    word index within sentence `sNNNNN`) for word-granularity highlighting. The sentence
    stays the synthesis unit and carries the `break_after`. Mutates the tree in place.
    """
    used = _document_ids(root)
    counter = count(1)
    injected: list[WordSentence] = []
    for block in narratable_blocks(root):
        block_break = (
            BreakAfter.HEADING if block.tag in _HEADING_TAGS else BreakAfter.PARAGRAPH
        )
        injected.extend(_inject_block_words(block, counter, used, block_break))

    return injected


def _reserve_base(counter: Iterator[int], used: set[str], word_count: int) -> str:
    # Pick the next `sNNNNN` whose whole word-id family `sNNNNN-wK` is free of ids
    # already in the document -- suffixed word ids would otherwise escape the base
    # uniqueness check and could duplicate a pre-existing id.
    while True:
        base = f"s{next(counter):05d}"
        family = {base, *(f"{base}-w{index}" for index in range(word_count))}
        if used.isdisjoint(family):
            used |= family
            return base


def _inject_block_words(
    block: XmlElement,
    counter: Iterator[int],
    used: set[str],
    block_break: BreakAfter,
) -> list[WordSentence]:
    raw_text = extract_block_text(block).text
    sentence_ranges = split_sentences(raw_text)
    if not sentence_ranges:
        return []

    segments = _segments(block)
    sentence_units = _merge_units(sentence_ranges, segments)

    all_units: list[tuple[int, int]] = []
    all_ids: list[str] = []
    sentences: list[WordSentence] = []
    for sentence_index, (start, end) in enumerate(sentence_units):
        display_text = collapse_whitespace(raw_text[start:end])
        raw_words = [
            (start + span.start, start + span.end)
            for span in segment_words(raw_text[start:end])
        ]

        # Merge words that straddle one inline element into a single wrappable unit,
        # mirroring how sentences merge across inline boundaries.
        word_units = _merge_units(raw_words, segments) if raw_words else []
        base_id = _reserve_base(counter, used, len(word_units))

        words: list[InjectedWord] = []
        cursor = 0
        for word_index, (word_start, word_end) in enumerate(word_units):
            word_display = collapse_whitespace(raw_text[word_start:word_end])

            # Locate the word in the sentence's collapsed text; the cursor keeps a
            # repeated word landing on its own occurrence.
            display_start = display_text.index(word_display, cursor)
            cursor = display_start + len(word_display)
            word_id = f"{base_id}-w{word_index}"
            words.append(
                InjectedWord(
                    span_id=word_id,
                    display_text=word_display,
                    start=display_start,
                    end=cursor,
                ),
            )
            all_units.append((word_start, word_end))
            all_ids.append(word_id)

        last = sentence_index == len(sentence_units) - 1
        sentences.append(
            WordSentence(
                span_id=base_id,
                display_text=display_text,
                break_after=block_break if last else BreakAfter.SENTENCE,
                words=tuple(words),
            ),
        )

    _rebuild(block, segments, all_units, all_ids)
    return sentences
