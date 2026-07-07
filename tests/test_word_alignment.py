"""Tests for word-level source segmentation and token alignment (word mode)."""

import math
from itertools import pairwise

import pytest

from spantrack.tts.backend import TokenTimestamp
from spantrack.word_alignment import AlignedSentence, align, segment_words


def _triples(text: str) -> list[tuple[str, int, int]]:
    spans = segment_words(text)
    for span in spans:
        assert text[span.start : span.end] == span.text  # every range slices back

    return [(span.text, span.start, span.end) for span in spans]


def test_segments_plain_sentence_into_words():
    assert _triples("The cat sat.") == [("The", 0, 3), ("cat", 4, 7), ("sat", 8, 11)]


def test_keeps_apostrophe_and_hyphen_intra_word():
    assert _triples("I don't like well-known cliches.") == [
        ("I", 0, 1),
        ("don't", 2, 7),
        ("like", 8, 12),
        ("well-known", 13, 23),
        ("cliches", 24, 31),
    ]


def test_typographic_apostrophe_is_intra_word():
    # A curly right-single-quote (U+2019) joins a contraction like a straight one.
    assert _triples("it’s here") == [("it’s", 0, 4), ("here", 5, 9)]


def test_excludes_surrounding_punctuation():
    assert _triples("(Hello!)") == [("Hello", 1, 6)]


def test_trailing_apostrophe_is_not_part_of_the_word():
    # A possessive plural's trailing apostrophe has no following word char, so it drops.
    assert _triples("the dogs' bowls") == [
        ("the", 0, 3),
        ("dogs", 4, 8),
        ("bowls", 10, 15),
    ]


def test_interior_punctuation_splits_words():
    assert _triples("cat,dog") == [("cat", 0, 3), ("dog", 4, 7)]


def test_numbers_are_words():
    assert _triples("It was 1984 then.") == [
        ("It", 0, 2),
        ("was", 3, 6),
        ("1984", 7, 11),
        ("then", 12, 16),
    ]


def test_empty_and_non_word_inputs_yield_no_spans():
    assert segment_words("") == []
    assert segment_words("    ") == []
    assert segment_words("!!! ... ---") == []


def test_spans_are_ordered_and_non_overlapping():
    spans = segment_words("one two three four")
    for previous, current in pairwise(spans):
        assert previous.end <= current.start


def _tok(text: str, start: float, end: float) -> TokenTimestamp:
    return TokenTimestamp(text=text, start_s=start, end_s=end, whitespace=" ")


def _ranges(result: AlignedSentence) -> list[tuple[int, int]]:
    return [(clip.span.start, clip.span.end) for clip in result.words]


def test_aligns_one_token_per_word_with_gapless_clips():
    display = "The cat sat"
    tokens = [_tok("The", 0.0, 0.4), _tok("cat", 0.4, 0.8), _tok("sat", 0.8, 1.2)]

    result = align(segment_words(display), tokens)

    assert _ranges(result) == [(0, 3), (4, 7), (8, 11)]
    bounds = [(clip.clip_begin_s, clip.clip_end_s) for clip in result.words]
    for (begin, end), (exp_begin, exp_end) in zip(
        bounds,
        [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2)],
        strict=True,
    ):
        assert math.isclose(begin, exp_begin)
        assert math.isclose(end, exp_end)
    # Every token maps to the word it spoke.
    assert result.source_ranges == ((0, 3), (4, 7), (8, 11))


def test_expansion_tokens_group_onto_one_source_word():
    # A number a normalizer expands ("1984" -> three spoken tokens) stays one span.
    tokens = [
        _tok("nineteen", 0.0, 0.2),
        _tok("eighty", 0.2, 0.5),
        _tok("four", 0.5, 0.8),
    ]

    result = align(segment_words("1984"), tokens)
    assert len(result.words) == 1
    assert result.source_ranges == ((0, 4), (0, 4), (0, 4))

    clip = result.words[0]
    assert (clip.span.start, clip.span.end) == (0, 4)
    assert math.isclose(clip.clip_begin_s, 0.0)
    assert math.isclose(clip.clip_end_s, 0.8)  # spans first begin to last end


def test_expansion_in_the_middle_of_a_sentence():
    display = "It was 1984 then"
    tokens = [
        _tok("It", 0.0, 0.2),
        _tok("was", 0.2, 0.5),
        _tok("nineteen", 0.5, 0.7),
        _tok("eighty", 0.7, 0.9),
        _tok("four", 0.9, 1.1),
        _tok("then", 1.1, 1.4),
    ]

    result = align(segment_words(display), tokens)
    assert _ranges(result) == [(0, 2), (3, 6), (7, 11), (12, 16)]
    assert result.source_ranges == (
        (0, 2),
        (3, 6),
        (7, 11),
        (7, 11),
        (7, 11),
        (12, 16),
    )

    number = result.words[2]
    assert math.isclose(number.clip_begin_s, 0.5)
    assert math.isclose(number.clip_end_s, 1.1)


def test_word_without_a_token_gets_a_zero_width_clip_at_the_boundary():
    # "world" was dropped (untimed), so it collapses to a point at its neighbour's end.
    result = align(segment_words("hi world"), [_tok("hi", 0.0, 0.3)])
    assert result.source_ranges == ((0, 2),)
    assert _ranges(result) == [(0, 2), (3, 8)]

    world = result.words[1]
    assert math.isclose(world.clip_begin_s, 0.3)
    assert math.isclose(world.clip_end_s, 0.3)


def test_word_clips_are_gapless_and_monotonic():
    tokens = [_tok("one", 0.0, 0.3), _tok("two", 0.5, 0.9), _tok("three", 0.9, 1.5)]

    result = align(segment_words("one two three"), tokens)

    for previous, current in pairwise(result.words):
        assert math.isclose(previous.clip_end_s, current.clip_begin_s)  # gapless
        assert previous.clip_begin_s <= current.clip_begin_s  # monotonic


def test_align_fails_loud_without_tokens():
    with pytest.raises(ValueError, match="at least one token"):
        align(segment_words("anything"), [])


def test_adjacent_identical_words_each_keep_their_own_token():
    # Regression: the second "the" must not steal the first "the"'s token and leave it a
    # zero-length clip.
    result = align(
        segment_words("the the"), [_tok("the", 0.0, 0.3), _tok("the", 0.3, 0.6)]
    )

    assert _ranges(result) == [(0, 3), (4, 7)]
    assert math.isclose(result.words[0].clip_begin_s, 0.0)
    assert math.isclose(result.words[0].clip_end_s, 0.3)
    assert math.isclose(result.words[1].clip_begin_s, 0.3)
    assert math.isclose(result.words[1].clip_end_s, 0.6)
    assert result.source_ranges == ((0, 3), (4, 7))


def test_real_word_after_an_expansion_is_not_starved():
    # "3" is voiced only by expansion tokens (none equal "3"), so the walk must still
    # release onto "approx" rather than swallowing it into the number.
    display = "pi is 3.14 approx"
    tokens = [
        _tok("pi", 0.0, 0.3),
        _tok("is", 0.3, 0.5),
        _tok("three", 0.5, 0.8),
        _tok("point", 0.8, 1.0),
        _tok("one", 1.0, 1.2),
        _tok("four", 1.2, 1.5),
        _tok("approx", 1.5, 2.0),
    ]

    result = align(segment_words(display), tokens)

    assert [clip.span.text for clip in result.words] == [
        "pi",
        "is",
        "3",
        "14",
        "approx",
    ]
    assert math.isclose(result.words[4].clip_end_s, 2.0)
    assert result.source_ranges[6] == (11, 17)  # the approx token maps to "approx"
    assert result.source_ranges[2] == (6, 7)  # expansion tokens map to "3"


def test_punctuation_only_token_does_not_advance_the_word():
    # A timed comma token has an empty key, so it must stay on the current word rather
    # than consuming "there" (exercises the `if key` guards in _assign).
    tokens = [
        _tok("hi", 0.0, 0.3),
        _tok(",", 0.3, 0.35),
        _tok("there", 0.35, 0.7),
    ]

    result = align(segment_words("hi there"), tokens)

    assert _ranges(result) == [(0, 2), (3, 8)]
    assert result.source_ranges == ((0, 2), (0, 2), (3, 8))


def test_out_of_order_token_end_never_inverts_a_clip():
    # Defensive: a regressing token end (outside the spoken-order contract) must not
    # yield a negative-duration clip, which would be invalid SMIL.
    result = align(
        segment_words("go now"),
        [
            _tok("go", 0.0, 1.5),
            _tok("now", 0.2, 0.8),
        ],
    )

    for clip in result.words:
        assert clip.clip_end_s >= clip.clip_begin_s
