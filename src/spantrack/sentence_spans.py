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

from spantrack.lxml_reading import XHTML_NS, XmlElement, qname
from spantrack.pysbd_segmentation import split_sentences
from spantrack.text_normalization import collapse_whitespace

__all__ = [
    "BlockText",
    "TextPiece",
    "extract_block_text",
    "inject",
    "inject_block",
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


def inject(root: XmlElement) -> dict[str, str]:
    """Inject sentence spans across a whole content document, in document order.

    Walks the document's narratable blocks and wraps each sentence run in a
    `<span id="sNNNNN">` with an id unique and sequential within the document. Mutates
    the tree in place.

    Args:
        root: The parsed content-document root (e.g. its `<html>` element).

    Returns:
        The injected spans' `{span_id: display_text}`, ordered by document position.
    """
    # Skip any id already used in the document so the injected ids stay unique document-
    # wide (a hard XHTML requirement, and media overlays resolve spans by id).
    id_gen = _fresh_ids(_document_ids(root))
    emitted: dict[str, str] = {}
    for block in narratable_blocks(root):
        emitted.update(inject_block(block, id_gen))

    return emitted


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
