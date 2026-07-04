import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
from segmentation_oracle import syntok_sentence_count

from spantrack.pysbd_segmentation import (
    _MAX_SEGMENTER_CHARS,  # pyright: ignore[reportPrivateUsage]  # boundary test needs it
    split_sentences,
)

_SRC = Path(__file__).resolve().parents[1] / "src"

# Fiction-dialogue cases with their linguistically correct sentence splits. These lock
# the hardened primitive's output and double as the syntok cross-check corpus below.
CASES: list[tuple[str, tuple[str, ...]]] = [
    (
        "The door opened. She stepped inside.",
        ("The door opened.", "She stepped inside."),
    ),
    (
        "Mr. Darcy bowed. Elizabeth curtsied.",
        ("Mr. Darcy bowed.", "Elizabeth curtsied."),
    ),
    (
        '"Where are you going?" he asked. She did not answer.',
        ('"Where are you going?" he asked.', "She did not answer."),
    ),
    (
        '"Run!" she screamed. The crowd scattered.',
        ('"Run!" she screamed.', "The crowd scattered."),
    ),
    (
        "The vial held 3.5 milliliters. It glowed faintly.",
        ("The vial held 3.5 milliliters.", "It glowed faintly."),
    ),
    ("He ran. He jumped. He fell.", ("He ran.", "He jumped.", "He fell.")),
]


def _slices(text: str) -> list[str]:
    return [text[start:end] for start, end in split_sentences(text)]


def _covered(ranges: list[tuple[int, int]]) -> set[int]:
    return {index for start, end in ranges for index in range(start, end)}


def _non_whitespace(text: str) -> set[int]:
    return {index for index, char in enumerate(text) if not char.isspace()}


def test_splits_into_ordered_offset_ranges():
    text = "The door opened. She stepped inside."
    ranges = split_sentences(text)

    assert ranges == [(0, 16), (17, 36)]
    assert [text[s:e] for s, e in ranges] == ["The door opened.", "She stepped inside."]


def test_ranges_are_forward_nonoverlapping_and_in_bounds():
    text = "One. Two. Three."
    ranges = split_sentences(text)
    assert ranges, "expected at least one range"

    for start, end in ranges:
        assert 0 <= start < end <= len(text)

    for (_, prev_end), (next_start, _) in pairwise(ranges):
        assert prev_end <= next_start


def test_ranges_are_tight_around_non_whitespace():
    text = "First.   Second."
    ranges = split_sentences(text)
    assert len(ranges) == 2

    for start, end in ranges:
        assert not text[start].isspace()
        assert not text[end - 1].isspace()


@pytest.mark.parametrize(
    "text",
    [
        "One.   Two.  Three.",
        "The door opened. She stepped inside.",
        "First para ends here.\nSecond para starts.",
        "Café awaited. Naïve fun—yes. Done.",
    ],
)
def test_ranges_cover_non_whitespace_with_whitespace_only_gaps(text: str):
    ranges = split_sentences(text)
    assert ranges

    # No non-whitespace char is dropped, and everything outside the ranges (gaps plus
    # leading and trailing edges) is whitespace only — so no real text falls in a gap.
    assert _non_whitespace(text) <= _covered(ranges)
    assert text[: ranges[0][0]].strip() == ""
    assert text[ranges[-1][1] :].strip() == ""

    for (_, prev_end), (next_start, _) in pairwise(ranges):
        assert text[prev_end:next_start].strip() == ""


def test_empty_and_whitespace_yield_no_ranges():
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_repeated_sentences_get_distinct_forward_ranges():
    # Guards against pysbd char_span's first-match offset bug on identical substrings.
    ranges = split_sentences("Go. Go. Go.")
    assert ranges == [(0, 3), (4, 7), (8, 11)]


def test_unicode_and_newline_slices_are_codepoint_exact():
    # Offsets are Python str code points; a byte/normalization regression would shift
    # them past multibyte letters. The real input domain is accented EPUB prose.
    assert _slices("Café awaited. Naïve fun—yes. Done.") == [
        "Café awaited.",
        "Naïve fun—yes.",
        "Done.",
    ]

    text = "One paragraph.\nAnother one."
    ranges = split_sentences(text)

    assert [text[s:e] for s, e in ranges] == ["One paragraph.", "Another one."]
    assert text[ranges[0][1] : ranges[1][0]] == "\n"


def test_reproduces_curated_fiction_splits():
    for text, expected in CASES:
        assert tuple(_slices(text)) == expected


def test_agrees_with_syntok_oracle():
    # Independent cross-check: pysbd's sentence count matches syntok's on every curated
    # case, and both match the hand-authored split. An unexpected divergence fails here.
    for text, expected in CASES:
        assert (
            len(split_sentences(text)) == syntok_sentence_count(text) == len(expected)
        ), text


def test_concat_failure_falls_back_to_single_unit_without_dropping_text():
    # Degenerate punctuation/NBSP runs make pysbd shuffle characters across boundaries;
    # the block is narrated whole rather than losing the un-findable fragment.
    for text in ("Stop\xa0!\xa0!\xa0!", ")?\xa0?\xa0?5"):
        ranges = split_sentences(text)
        assert len(ranges) == 1
        assert _non_whitespace(text) <= _covered(ranges)


def test_over_length_input_returns_single_unit():
    # Above the #79 length backstop: narrated whole rather than fed to pysbd. A single
    # range over 700 sentences proves the guard engaged (pysbd would return ~700).
    text = "This is a sentence. " * 700
    ranges = split_sentences(text)
    assert len(ranges) == 1

    start, end = ranges[0]
    assert text[start:end] == text.strip()


def test_length_backstop_engages_strictly_above_the_ceiling():
    unit = "This is a sentence. "
    count = _MAX_SEGMENTER_CHARS // len(unit)

    # At the ceiling pysbd still segments each sentence; one char over collapses to one.
    at_ceiling = unit * count
    assert len(at_ceiling) == _MAX_SEGMENTER_CHARS
    assert len(split_sentences(at_ceiling)) == count
    assert len(split_sentences(at_ceiling + "X")) == 1


def test_import_is_silent_on_cold_compile(tmp_path: Path):
    # pysbd emits SyntaxWarnings at COMPILE time (invalid escape sequences); the module
    # suppresses them around its import. Force a cold recompile via a fresh cache prefix
    # and promote SyntaxWarning to an error, so dropping the suppression fails here.
    env = {
        **os.environ,
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        "PYTHONPATH": str(_SRC),
    }
    # Fixed argv, trusted interpreter — no untrusted input reaches the shell.
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::SyntaxWarning",
            "-c",
            "import spantrack.pysbd_segmentation",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
