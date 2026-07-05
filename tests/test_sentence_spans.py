from collections.abc import Iterator
from copy import deepcopy
from itertools import count, pairwise

import pytest
from lxml import etree
from segmentation_oracle import syntok_sentence_count

from spantrack.audio_assembly import BreakAfter
from spantrack.lxml_reading import XHTML_NS, XmlElement, hardened_parser, qname
from spantrack.sentence_spans import (
    BlockText,
    extract_block_text,
    inject,
    inject_block,
    inject_words,
    narratable_blocks,
)

_XHTML = "http://www.w3.org/1999/xhtml"


def _parse(body: str) -> XmlElement:
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        f'<html xmlns="{_XHTML}"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )
    return etree.fromstring(doc.encode(), hardened_parser())


def _local(element: XmlElement) -> str:
    return etree.QName(element).localname


def _tags(body: str) -> list[str]:
    return [_local(block) for block in narratable_blocks(_parse(body))]


def test_yields_leaf_blocks_in_document_order():
    body = (
        "<h1>Title</h1><p>First.</p>"
        "<ul><li>one</li><li>two</li></ul>"
        "<blockquote><p>quoted</p></blockquote>"
    )

    assert _tags(body) == ["h1", "p", "li", "li", "p"]


def test_recognizes_every_block_tag_type():
    body = (
        "<h1>a</h1><h2>b</h2><h3>c</h3><h4>d</h4><h5>e</h5><h6>f</h6>"
        "<p>g</p><ul><li>h</li></ul><blockquote>i</blockquote>"
        "<figure><figcaption>j</figcaption></figure>"
        "<dl><dt>k</dt><dd>l</dd></dl>"
    )

    assert _tags(body) == [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "blockquote",
        "figcaption",
        "dt",
        "dd",
    ]


def test_leaf_block_wins_over_wrapping_block():
    # A blockquote/li that wraps its own p contributes the inner p only, never both.
    blocks = narratable_blocks(_parse("<blockquote><p>inner</p></blockquote>"))

    assert [_local(b) for b in blocks] == ["p"]
    assert blocks[0].text == "inner"


def test_container_own_text_dropped_when_nested_block_present():
    # A container with both its own text and a nested block yields only the nested leaf;
    # its direct text ("lead-in") is intentionally not narrated.
    blocks = narratable_blocks(_parse("<blockquote>lead-in<p>quoted</p></blockquote>"))

    assert [(_local(b), b.text) for b in blocks] == [("p", "quoted")]


def test_returns_live_dom_nodes():
    # Later stages inject <span> by mutating these nodes in place, so the returned
    # elements must be the live tree nodes, not detached copies.
    root = _parse("<p>hi</p><p>bye</p>")
    blocks = narratable_blocks(root)
    assert blocks[0] is root.find(f".//{{{_XHTML}}}p")

    etree.SubElement(blocks[0], f"{{{_XHTML}}}span").text = "MARK"
    assert b"MARK" in etree.tostring(root)


@pytest.mark.parametrize(
    "container",
    ["table", "pre", "code", "script", "style", "nav"],
)
def test_blocks_inside_skip_containers_are_pruned(container: str):
    body = f"<p>keep</p><{container}><p>drop</p></{container}><p>after</p>"
    blocks = narratable_blocks(_parse(body))

    assert [b.text for b in blocks] == ["keep", "after"]


def test_skips_whitespace_only_blocks():
    blocks = narratable_blocks(_parse("<p>   </p><p></p><p><br/></p><p>real</p>"))

    assert [b.text for b in blocks] == ["real"]


def test_block_with_only_skipped_inline_text_is_dropped():
    # A paragraph whose only content is inline code (skipped by default) has no prose.
    assert narratable_blocks(_parse("<p><code>x = 1</code></p>")) == []


def test_block_keeps_prose_around_inline_code():
    blocks = narratable_blocks(_parse("<p>Run <code>x</code> now.</p>"))

    assert [_local(b) for b in blocks] == ["p"]
    assert blocks[0].text == "Run "


def test_block_kept_by_tail_after_skipped_child():
    # The block's only prose can be the tail after a skipped child (inline code, a
    # nested table); it must still be narrated.
    assert _tags("<p><code>ignored</code> real prose.</p>") == ["p"]
    assert _tags("<p><table><tr><td>z</td></tr></table> after table.</p>") == ["p"]


def test_comment_and_pi_only_blocks_are_dropped():
    # Comment/PI bodies are invisible metadata, not prose — they never make a sync unit.
    assert narratable_blocks(_parse("<p><!-- editor note --></p>")) == []
    assert narratable_blocks(_parse("<p><!--realwords--></p>")) == []
    assert narratable_blocks(_parse("<p><?pi data?></p>")) == []


def test_prose_around_a_comment_is_narrated():
    blocks = narratable_blocks(_parse("<p>Hello <!-- c --> world.</p>"))

    assert [_local(b) for b in blocks] == ["p"]


def test_foreign_namespace_block_is_not_narrated():
    body = '<p xmlns="http://example.com/ns">foreign</p><p>real</p>'
    blocks = narratable_blocks(_parse(body))

    assert [b.text for b in blocks] == ["real"]


def test_descends_through_non_block_wrappers():
    blocks = narratable_blocks(_parse("<div><section><p>deep</p></section></div>"))

    assert [(_local(b), b.text) for b in blocks] == [("p", "deep")]


def test_bare_text_outside_blocks_is_not_narrated():
    assert narratable_blocks(_parse("<div>loose text</div>")) == []


def test_empty_body_yields_nothing():
    assert narratable_blocks(_parse("")) == []


def _extract(inner: str) -> BlockText:
    block = narratable_blocks(_parse(f"<p>{inner}</p>"))[0]
    return extract_block_text(block)


def _child(block_text: BlockText, local: str) -> XmlElement:
    node = block_text.pieces[0].node.find(f".//{{{_XHTML}}}{local}")
    assert node is not None
    return node


def test_extract_plain_block_maps_to_its_own_text():
    block_text = _extract("Hello there.")
    assert block_text.text == "Hello there."
    assert len(block_text.pieces) == 1

    piece = block_text.pieces[0]
    assert piece.is_tail is False
    assert piece.offset == 0
    assert piece.string == "Hello there."


def test_extract_concatenates_text_and_tails_in_document_order():
    block_text = _extract("a<em>b</em>c<a>d</a>e")

    assert block_text.text == "abcde"
    assert [(p.is_tail, p.string) for p in block_text.pieces] == [
        (False, "a"),
        (False, "b"),
        (True, "c"),
        (False, "d"),
        (True, "e"),
    ]


def test_extract_flattens_nested_inline():
    block_text = _extract("a<em><strong>x</strong></em>b")

    assert block_text.text == "axb"
    assert [(p.is_tail, p.string) for p in block_text.pieces] == [
        (False, "a"),
        (False, "x"),
        (True, "b"),
    ]


def test_offsets_resolve_to_the_correct_node_and_kind():
    block_text = _extract("Go <em>deep</em> now")
    em = _child(block_text, "em")

    assert block_text.text == "Go deep now"
    # Every character offset maps back to the DOM string it came from.
    assert block_text.locate(0)[0].is_tail is False  # 'G' in the block's own text
    assert block_text.locate(3) == (block_text.pieces[1], 0)  # 'd', em.text[0]
    node, local = block_text.locate(7)  # ' ' in em.tail
    assert node.node is em
    assert node.is_tail is True
    assert local == 0
    # Each piece's recorded string matches the live DOM string it points at.
    for piece in block_text.pieces:
        source = piece.node.tail if piece.is_tail else piece.node.text
        assert source == piece.string
        assert (
            block_text.text[piece.offset : piece.offset + len(piece.string)]
            == piece.string
        )


def test_extract_skips_non_prose_subtree_but_keeps_its_tail():
    block_text = _extract("before <code>ignored</code> after")

    assert block_text.text == "before  after"
    assert [(p.is_tail, p.string) for p in block_text.pieces] == [
        (False, "before "),
        (True, " after"),
    ]


def test_extract_ignores_comment_body_keeps_tail():
    block_text = _extract("x<!-- note -->y")

    assert block_text.text == "xy"


@pytest.mark.parametrize("offset", [2, 7, -1])
def test_locate_rejects_out_of_range_offset(offset: int):
    block_text = _extract("hi")  # text "hi": only offsets 0 and 1 are in range

    with pytest.raises(IndexError):
        block_text.locate(offset)


def test_bare_namespace_document_is_not_narrated():
    # The walk matches Clark-notation XHTML tags, so a document whose <html> omits the
    # XHTML namespace puts every element in no namespace and yields nothing. This is
    # deliberately NOT patched piecemeal here; content_namespace.require_xhtml_namespace
    # refuses such a document upstream (see the Decision Log). Narrating it here while
    # projection still counts it zero would trade one inconsistency for another.
    doc = etree.fromstring(
        b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        b"<html><head><title>t</title></head>"
        b"<body><h1>Title</h1><p>Bare.</p></body></html>",
        hardened_parser(),
    )

    assert narratable_blocks(doc) == []


def _assert_pieces_tile_and_map(block_text: BlockText) -> None:
    pieces = block_text.pieces

    # Pieces tile [0, len(text)) contiguously — no gaps, no overlaps, none zero-length.
    assert pieces[0].offset == 0
    for prev, nxt in pairwise(pieces):
        assert nxt.offset == prev.offset + len(prev.string)
    assert pieces[-1].offset + len(pieces[-1].string) == len(block_text.text)
    assert all(piece.string for piece in pieces)

    # Every offset resolves to the right piece, local index, and live DOM string.
    for offset in range(len(block_text.text)):
        piece, local = block_text.locate(offset)
        assert piece.string[local] == block_text.text[offset]
        if piece.node.tag is etree.Entity and not piece.is_tail:
            continue  # entity char piece: string is the resolved char, not node.text
        source = piece.node.tail if piece.is_tail else piece.node.text
        assert source == piece.string


@pytest.mark.parametrize(
    "inner",
    [
        "a<em>b</em>c<a>d</a>e",
        "café <em>x</em> y",  # multibyte: offsets are code points, not bytes
        "x\U0001f600<em>y</em>z",  # astral char is one code point
        "before <code>skip</code> after",  # only prose is a tail after a skipped child
        "p<!-- c -->q",  # tail piece points at the comment node
    ],
)
def test_pieces_tile_block_text_and_map_every_offset(inner: str):
    _assert_pieces_tile_and_map(_extract(inner))


def test_extract_omits_empty_string_nodes():
    # Empty CDATA parses to `.text == ""` (not None); it must not become a zero-length
    # piece that locate() can never resolve.
    block_text = _extract("before <em><![CDATA[]]></em> after")

    assert all(piece.string for piece in block_text.pieces)
    assert [(p.is_tail, p.string) for p in block_text.pieces] == [
        (False, "before "),
        (True, " after"),
    ]


_RSQUO_DTD = '<!DOCTYPE html [ <!ENTITY rsquo "&#8217;"> ]>'
_RSQUO = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK
_MDASH = chr(0x2014)  # EM DASH


def _parse_with_dtd(dtd: str, body: str) -> XmlElement:
    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>{dtd}'
        f'<html xmlns="{_XHTML}"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )
    return etree.fromstring(doc.encode(), hardened_parser())


def _extract_with_dtd(dtd: str, inner: str) -> BlockText:
    return extract_block_text(
        narratable_blocks(_parse_with_dtd(dtd, f"<p>{inner}</p>"))[0]
    )


def test_entity_reference_resolves_into_narration_text():
    # A named entity (declared in the internal DTD subset) contributes its resolved
    # character to the narration text; numeric refs and predefined entities already
    # expand into `.text` during parsing.
    block_text = _extract_with_dtd(_RSQUO_DTD, "don&rsquo;t")

    assert block_text.text == "don" + _RSQUO + "t"


def test_entity_piece_is_atomic_and_leaves_the_dom_untouched():
    block_text = _extract_with_dtd(_RSQUO_DTD, "a&rsquo;b")

    assert block_text.text == "a" + _RSQUO + "b"
    entity_piece, tail_piece = block_text.pieces[1], block_text.pieces[2]
    # The atomic entity-char piece is `Entity tag and not is_tail`; its DOM text stays
    # the literal reference, so the output document's bytes are unchanged.
    assert entity_piece.string == _RSQUO
    assert entity_piece.node.tag is etree.Entity
    assert entity_piece.is_tail is False
    assert entity_piece.node.text == "&rsquo;"
    # The text after the entity is an ordinary tail piece on the same node.
    assert (tail_piece.string, tail_piece.is_tail) == ("b", True)
    assert tail_piece.node is entity_piece.node


def test_entity_offsets_stay_aligned_across_following_markup():
    block_text = _extract_with_dtd(_RSQUO_DTD, "x&rsquo;<em>y</em>z")

    assert block_text.text == "x" + _RSQUO + "yz"
    entity_piece, entity_local = block_text.locate(1)
    assert entity_piece.node.tag is etree.Entity
    assert entity_local == 0
    following, _ = block_text.locate(2)  # 'y' in em.text, not shifted by the entity
    assert following.string == "y"


def test_entity_only_block_is_narrated_not_dropped():
    # Previously a block whose only content was entity references reported no narratable
    # text and was dropped entirely; the resolved characters now make it a sync unit.
    root = _parse_with_dtd(
        '<!DOCTYPE html [ <!ENTITY mdash "&#8212;"> ]>',
        "<p>&mdash;&mdash;</p>",
    )
    blocks = narratable_blocks(root)

    assert [_local(b) for b in blocks] == ["p"]
    assert extract_block_text(blocks[0]).text == _MDASH + _MDASH


def test_nested_or_custom_entity_is_dropped_not_leaked():
    # libxml2 does not expand a nested reference inside a declaration, and a custom text
    # entity is not a standard HTML name; rather than narrate literal `&name;` markup an
    # unresolvable entity is dropped from the narration text (the DOM stays untouched).
    root = _parse_with_dtd(
        '<!DOCTYPE html [ <!ENTITY b "BB"> <!ENTITY a "&b;"> ]>',
        "<p>x&a;y</p>",
    )
    block_text = extract_block_text(narratable_blocks(root)[0])

    assert block_text.text == "xy"
    assert "&" not in block_text.text
    assert b"&a;" in etree.tostring(
        root
    )  # the entity reference survives in the document


def test_entity_block_tiles_and_maps_every_offset():
    # Same tiling + offset-map invariant as the plain case, but with an atomic entity
    # char piece in the mix (the shared helper retains the entity-skip conditional).
    _assert_pieces_tile_and_map(_extract_with_dtd(_RSQUO_DTD, "a&rsquo;<em>b</em>c"))


def test_entity_with_no_tail_is_a_single_non_tail_char_piece():
    block_text = _extract_with_dtd(_RSQUO_DTD, "end&rsquo;")

    assert block_text.text == "end" + _RSQUO
    last = block_text.pieces[-1]
    assert last.node.tag is etree.Entity
    assert last.is_tail is False
    assert last.string == _RSQUO


def _block(inner: str) -> XmlElement:
    node = _parse(f"<p>{inner}</p>").find(f".//{{{_XHTML}}}p")
    assert node is not None
    return node


def _ids() -> Iterator[str]:
    return (f"s{n:05d}" for n in count(1))


def _inner(block: XmlElement) -> str:
    # Serialize the block with the default xmlns stripped, for readable asserts.
    return etree.tostring(block, encoding="unicode").replace(f' xmlns="{_XHTML}"', "")


def _strip_spans(element: XmlElement, injected_ids: set[str]) -> str:
    # Unwrap ONLY the injected spans, matched by their exact emitted ids (not a pattern,
    # so a publisher <span id="s99"> survives). Retag to a sentinel first, since
    # strip_tags cannot filter by attribute.
    clone = deepcopy(element)
    for span in clone.iter(f"{{{_XHTML}}}span"):
        if span.get("id") in injected_ids:
            span.tag = f"{{{_XHTML}}}_injected_"
    etree.strip_tags(clone, f"{{{_XHTML}}}_injected_")
    return _inner(clone)


def test_single_sentence_wraps_all_content_in_one_span():
    block = _block("Hello there.")

    emitted = inject_block(block, _ids())

    assert emitted == [("s00001", "Hello there.")]
    assert _inner(block) == '<p><span id="s00001">Hello there.</span></p>'


def test_two_sentences_get_two_spans_with_whitespace_left_between():
    block = _block("First. Second.")

    emitted = inject_block(block, _ids())

    assert emitted == [("s00001", "First."), ("s00002", "Second.")]
    assert _inner(block) == (
        '<p><span id="s00001">First.</span> <span id="s00002">Second.</span></p>'
    )


def test_inline_element_wholly_in_one_sentence_is_kept_intact():
    block = _block('Hi <em class="x">there</em>. Bye.')

    emitted = inject_block(block, _ids())

    assert [d for _, d in emitted] == ["Hi there.", "Bye."]
    assert _inner(block) == (
        '<p><span id="s00001">Hi <em class="x">there</em>.</span>'
        ' <span id="s00002">Bye.</span></p>'
    )


def test_sentence_boundary_inside_inline_merges_into_one_sync_unit():
    block = _block("<em>One. Two</em> three.")

    emitted = inject_block(block, _ids())

    # The boundary after "One." falls inside <em>; splitting would break the element, so
    # the two sentences merge into a single span wrapping the whole <em> and its tail.
    assert emitted == [("s00001", "One. Two three.")]
    assert _inner(block) == ('<p><span id="s00001"><em>One. Two</em> three.</span></p>')


def test_inline_element_straddling_two_sentences_merges():
    block = _block("Done. <a href='u'>Look. Here</a> now.")

    emitted = inject_block(block, _ids())

    # <a> straddles the boundary between "Look." and "Here now." -> merge those two.
    assert [d for _, d in emitted] == ["Done.", "Look. Here now."]
    assert 'href="u"' in _inner(block)


def test_display_text_is_whitespace_collapsed():
    block = _block("Hello    there.\n  General Kenobi.")

    emitted = inject_block(block, _ids())

    assert [d for _, d in emitted] == ["Hello there.", "General Kenobi."]


def test_entity_reference_is_wrapped_atomically():
    root = _parse_with_dtd(_RSQUO_DTD, "<p>don&rsquo;t stop.</p>")
    block = root.find(f".//{{{_XHTML}}}p")
    assert block is not None

    emitted = inject_block(block, _ids())

    assert [d for _, d in emitted] == ["don" + _RSQUO + "t stop."]
    assert b"&rsquo;" in etree.tostring(
        root
    )  # DOM entity node preserved inside the span


def test_non_narratable_child_at_a_boundary_stays_loose_between_spans():
    block = _block('One. <img src="a.png"/> Two.')

    inject_block(block, _ids())

    # The image is at a sentence boundary, so it survives loose between the two spans.
    assert _inner(block) == (
        '<p><span id="s00001">One.</span>'
        ' <img src="a.png"/> <span id="s00002">Two.</span></p>'
    )


def test_non_narratable_child_interior_to_a_sentence_stays_in_the_span():
    block = _block('Hello <img src="a"/> world. Bye.')

    inject_block(block, _ids())

    # The image sits inside the first sentence, so it belongs to that sentence's span.
    assert _inner(block) == (
        '<p><span id="s00001">Hello <img src="a"/> world.</span>'
        ' <span id="s00002">Bye.</span></p>'
    )


@pytest.mark.parametrize(
    "inner",
    [
        "<em>A.</em><em>B.</em>",  # adjacent inline elements, no separating text
        "<em>Hi.</em> Bye.",  # leading inline element
        "Hi. <em>Bye.</em>",  # trailing inline element
        "A.  \n  B.   C.",  # multi-space / newline gaps
        'Hi <em id="e" data-x="1">there</em>. Bye.',  # inline attributes
        "One. <img src='a.png'/> Two.",  # non-narratable child between sentences
        "One.<!-- c --> Two.",  # comment between sentences
        "A <b><i>deep. nest</i></b> b. C.",  # deeply nested straddle
        "<em>One. Two. Three</em> four.",  # three sentences inside one element
        "Pre <table><tr><td>x. y</td></tr></table> post. End.",  # skip-container child
        "  Padded.  Text.  ",  # leading/trailing block whitespace
        'A <span class="k">source</span> span. And more.',  # publisher's own <span>
        'A <span id="s99">source</span> span. And more.',  # publisher <span>, s-like id
    ],
)
def test_layout_is_byte_identical_outside_injected_spans(inner: str):
    block = _block(inner)
    before = _inner(block)

    emitted = inject_block(block, _ids())

    assert _strip_spans(block, {span_id for span_id, _ in emitted}) == before


def test_no_sentences_leaves_block_unchanged_and_returns_empty():
    block = _block("   ")
    before = _inner(block)

    emitted = inject_block(block, _ids())

    assert emitted == []
    assert _inner(block) == before


def test_ids_come_from_the_generator_in_order():
    block = _block("One. Two. Three.")

    emitted = inject_block(block, _ids())

    assert [span_id for span_id, _ in emitted] == ["s00001", "s00002", "s00003"]


def test_child_with_leading_whitespace_as_whole_sentence():
    # The <em>'s .text opens with whitespace and carries the entire second sentence; its
    # _ChildRun.start lands in the inter-sentence gap, which must not misroute it (was a
    # KeyError crash before classifying children by overlap).
    block = _block("One. <em>  Two.</em>")

    emitted = inject_block(block, _ids())

    assert emitted == [("s00001", "One."), ("s00002", "Two.")]
    assert _inner(block) == (
        '<p><span id="s00001">One.</span> <span id="s00002"><em>  Two.</em></span></p>'
    )


def test_child_with_leading_whitespace_stays_inside_its_span():
    # The <em> opens the second sentence but the sentence continues past it; the <em>
    # must be wrapped inside that span, not left loose outside it.
    block = _block("One. <em>  Two</em> three.")

    emitted = inject_block(block, _ids())

    assert emitted == [("s00001", "One."), ("s00002", "Two three.")]
    assert _inner(block) == (
        '<p><span id="s00001">One.</span>'
        ' <span id="s00002"><em>  Two</em> three.</span></p>'
    )


def test_child_with_trailing_whitespace_stays_inside_its_span():
    # Trailing whitespace pushes the child's end past the trimmed sentence range;
    # overlap classification keeps it in its span (an end check alone would misroute).
    block = _block("One. <em>Two.  </em> three end.")

    emitted = inject_block(block, _ids())

    assert [d for _, d in emitted] == ["One.", "Two.", "three end."]
    assert '<span id="s00002"><em>Two.  </em></span>' in _inner(block)


def test_adjacent_inline_elements_split_into_separate_spans():
    # Two narratable inline children at a real sentence boundary land in two spans, each
    # wrapping its own element intact (no span splits an element, none is left loose).
    block = _block("<b>Stop right now! </b><b>Go away please.</b>")

    emitted = inject_block(block, _ids())

    assert [d for _, d in emitted] == ["Stop right now!", "Go away please."]
    assert _inner(block) == (
        '<p><span id="s00001"><b>Stop right now! </b></span>'
        '<span id="s00002"><b>Go away please.</b></span></p>'
    )


def test_source_span_survives_injection():
    block = _block('Hi <span class="k">there</span>. Bye.')

    inject_block(block, _ids())

    # The publisher's own <span> is preserved intact, nested inside the injected span.
    assert '<span class="k">there</span>' in _inner(block)


def test_block_with_only_a_non_narratable_child_is_unchanged():
    block = _block('<img src="a.png"/>')
    before = _inner(block)

    emitted = inject_block(block, _ids())

    assert emitted == []
    assert _inner(block) == before


def test_id_gen_exhaustion_is_an_atomic_no_op():
    block = _block("One. Two. Three.")
    before = _inner(block)

    with pytest.raises(StopIteration):
        inject_block(block, iter(["s1", "s2"]))  # only two ids for three units

    # The pull happens before any mutation, so the block is left completely untouched.
    assert _inner(block) == before


_FICTION = [
    (
        "The door opened. She stepped inside.",
        ("The door opened.", "She stepped inside."),
    ),
    (
        '"Where are you going?" he asked. She left.',
        ('"Where are you going?" he asked.', "She left."),
    ),
    ("He ran. He jumped. He fell.", ("He ran.", "He jumped.", "He fell.")),
]


def test_inject_assigns_sequential_document_scoped_ids():
    root = _parse("<h1>Title here.</h1><p>First. Second.</p>")

    emitted = inject(root)

    assert [span.span_id for span in emitted] == ["s00001", "s00002", "s00003"]


def test_inject_returns_ordered_injected_spans():
    root = _parse("<p>One. Two.</p><p>Three.</p>")

    emitted = inject(root)

    assert [(span.span_id, span.display_text) for span in emitted] == [
        ("s00001", "One."),
        ("s00002", "Two."),
        ("s00003", "Three."),
    ]


def test_inject_classifies_break_after_by_block_structure():
    root = _parse(
        "<h1>Title here.</h1><p>First. Second.</p><ul><li>Item one.</li></ul>"
    )

    emitted = inject(root)

    # The block-final span carries the block's break (heading/paragraph); an interior
    # span breaks as a plain sentence.
    assert [(span.span_id, span.break_after) for span in emitted] == [
        ("s00001", BreakAfter.HEADING),
        ("s00002", BreakAfter.SENTENCE),
        ("s00003", BreakAfter.PARAGRAPH),
        ("s00004", BreakAfter.PARAGRAPH),
    ]


def test_inject_breaks_heading_only_after_its_last_sentence():
    root = _parse("<h2>One. Two.</h2>")

    emitted = inject(root)

    assert [span.break_after for span in emitted] == [
        BreakAfter.SENTENCE,
        BreakAfter.HEADING,
    ]


def test_inject_skips_non_narratable_containers():
    root = _parse(
        "<nav><ol><li>skip</li></ol></nav>"
        "<table><tr><td>x. y</td></tr></table>"
        "<p>Real prose.</p>"
    )

    emitted = inject(root)

    assert [span.display_text for span in emitted] == ["Real prose."]


@pytest.mark.parametrize(
    "body",
    [
        "<h1>A Heading.</h1><p>First. <em>Second</em> point.</p>",
        "<p>Lead <a href='u'>x. y</a> tail.</p><blockquote>Quote here.</blockquote>",
        "<p>Pre <img src='i.png'/> post. End.</p><ul><li>one. two.</li></ul>",
        "<nav><ol><li>toc</li></ol></nav><p>Body. Text.</p>",
        '<p class="CN"><span id="s3">2</span></p><p>A source span. Kept.</p>',
    ],
)
def test_inject_preserves_document_layout_outside_spans(body: str):
    root = _parse(body)
    before = _inner(root)

    emitted = inject(root)

    assert _strip_spans(root, {span.span_id for span in emitted}) == before


def test_inject_over_a_fiction_document_matches_expected_and_syntok():
    root = _parse("".join(f"<p>{text}</p>" for text, _ in _FICTION))

    emitted = inject(root)

    # The strong check: exact per-span display texts across the document, in order.
    expected = [sentence for _, sentences in _FICTION for sentence in sentences]
    assert [span.display_text for span in emitted] == expected
    # Coarse independent oracle: on plain prose (no inline merges) syntok agrees.
    assert len(emitted) == sum(syntok_sentence_count(text) for text, _ in _FICTION)


def test_inject_ids_never_collide_with_source_ids():
    root = _parse(
        '<p>Hello there world today. <span id="s00001">kept</span></p><p>Next line.</p>'
    )

    emitted = inject(root)

    all_ids = [element.get("id") for element in root.iter() if element.get("id")]
    assert len(all_ids) == len(set(all_ids))  # every id in the document is unique
    injected_ids = {span.span_id for span in emitted}
    assert "s00001" not in injected_ids  # the injected ids skipped the taken source id


def test_inject_ids_stay_sequential_across_many_blocks():
    root = _parse("".join(f"<p>Item {n}.</p>" for n in range(10)))

    emitted = inject(root)

    assert [span.span_id for span in emitted] == [f"s{n:05d}" for n in range(1, 11)]


def test_inject_empty_document_returns_no_spans():
    root = _parse("<div>loose text</div><table><tr><td>x. y</td></tr></table>")

    assert inject(root) == []


def _text(element: XmlElement) -> str:
    # lxml stubs type itertext() as bytes-or-str; a str-parsed tree only yields str.
    return "".join(part for part in element.itertext() if isinstance(part, str))


def _word_spans(root: XmlElement) -> list[tuple[str, str]]:
    return [
        (span.get("id") or "", _text(span))
        for span in root.iter(qname(XHTML_NS, "span"))
    ]


def test_inject_words_wraps_each_word_in_its_own_span():
    root = _parse("<p>The cat sat.</p>")

    sentences = inject_words(root)
    assert len(sentences) == 1

    sentence = sentences[0]
    assert sentence.span_id == "s00001"
    assert sentence.break_after == BreakAfter.PARAGRAPH
    assert [word.display_text for word in sentence.words] == ["The", "cat", "sat"]
    assert _word_spans(root) == [
        ("s00001-w0", "The"),
        ("s00001-w1", "cat"),
        ("s00001-w2", "sat"),
    ]


def test_word_ranges_index_into_the_sentence_display_text():
    sentence = inject_words(_parse("<p>The cat sat.</p>"))[0]

    for word in sentence.words:
        assert sentence.display_text[word.start : word.end] == word.display_text


def test_word_ids_and_breaks_are_scoped_per_sentence():
    sentences = inject_words(_parse("<p>One two. Three four.</p>"))

    assert [sentence.span_id for sentence in sentences] == ["s00001", "s00002"]
    assert [word.span_id for word in sentences[0].words] == ["s00001-w0", "s00001-w1"]
    assert [word.span_id for word in sentences[1].words] == ["s00002-w0", "s00002-w1"]
    assert sentences[0].break_after == BreakAfter.SENTENCE
    assert sentences[1].break_after == BreakAfter.PARAGRAPH


def test_single_word_inline_element_is_wrapped_and_preserved():
    root = _parse("<p>a <em>big</em> cat</p>")

    sentence = inject_words(root)[0]
    assert [word.display_text for word in sentence.words] == ["a", "big", "cat"]

    em = next(root.iter(qname(XHTML_NS, "em")))
    assert _text(em) == "big"

    parent = em.getparent()
    assert parent is not None
    assert parent.get("id") == "s00001-w1"  # a span wraps the <em>


def test_multi_word_inline_element_merges_into_one_unit():
    # Experimental limitation: words inside one inline element highlight together.
    root = _parse("<p><em>big cat</em></p>")

    sentence = inject_words(root)[0]

    assert [word.display_text for word in sentence.words] == ["big cat"]
    assert [word.span_id for word in sentence.words] == ["s00001-w0"]


def test_heading_block_breaks_heading():
    sentence = inject_words(_parse("<h1>Chapter One</h1>"))[0]

    assert sentence.break_after == BreakAfter.HEADING


def test_punctuation_and_whitespace_stay_outside_word_spans():
    root = _parse("<p>Hello, world!</p>")

    sentence = inject_words(root)[0]
    block = narratable_blocks(root)[0]

    assert [word.display_text for word in sentence.words] == ["Hello", "world"]
    # The comma and exclamation are loose text, not swallowed by a span.
    assert _text(block) == "Hello, world!"


def test_inject_words_empty_block_injects_nothing():
    assert inject_words(_parse("<p>   </p>")) == []


def _all_ids(root: XmlElement) -> list[str]:
    return [value for element in root.iter() if (value := element.get("id"))]


def test_word_ids_avoid_a_pre_existing_word_scheme_id():
    # A source id already shaped like a word id must not be duplicated by injection.
    root = _parse('<p id="s00001-w0">Hello world.</p>')

    inject_words(root)

    ids = _all_ids(root)
    assert len(ids) == len(set(ids))  # no duplicate id attributes
    assert "s00001-w0" in ids  # the pre-existing id is untouched


def test_reprocessing_injected_output_keeps_ids_unique():
    root = _parse("<p>Hello world.</p>")

    inject_words(root)
    inject_words(root)  # feed spantrack's own word-mode output back in

    ids = _all_ids(root)
    assert len(ids) == len(set(ids))
