"""Word-level source segmentation and token-to-word alignment for word mode.

`segment_words` splits a sentence's display text into source words, fixing the one
word-boundary rule that word-span injection and SMIL regrouping both consume so the
injected `<span>`s and the overlay `<par>`s agree on what a word is. Apostrophes and
hyphens count as part of a word only between word characters, so "don't" and
"well-known" stay single words while surrounding punctuation is excluded; a leading or
trailing apostrophe/hyphen is not part of the word.

`align` regroups a sentence's chapter-absolute token timestamps back onto those source
words. Kokoro speaks the normalized text, so a normalizer-expanded number ("1984" ->
"nineteen eighty four") arrives as several tokens that must collapse onto the single
source word -- the alignment is a monotonic walk that keeps expansion tokens on the word
they voice and advances only when a token clearly opens the next word. Word clips are
made gapless (each word begins where the previous ended) so the overlay never flickers,
matching the sentence-mode invariant. It is pure: no synthesis, no DOM, no re-alignment
of audio -- the timings come straight from the tokens.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from spantrack.tts.backend import TokenTimestamp

__all__ = [
    "AlignedSentence",
    "WordClip",
    "WordSpan",
    "align",
    "segment_words",
]

# A word is a run of word characters, extended across an internal apostrophe or hyphen
# (one flanked by word characters). The straight apostrophe and the typographic right
# single quote U+2019 both join contractions; `\uXXXX` is an `re` escape even in a raw
# pattern, so the source stays ASCII.
_WORD = re.compile(r"\w+(?:['\u2019\-]\w+)*")


@dataclass(frozen=True, slots=True)
class WordSpan:
    """One source word and its `[start, end)` char offsets into the segmented text."""

    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class WordClip:
    """A source word paired with the chapter-absolute clip that voices it."""

    span: WordSpan
    clip_begin_s: float
    clip_end_s: float


@dataclass(frozen=True, slots=True)
class AlignedSentence:
    """One sentence's word clips plus each token's source range into `display_text`.

    `words` holds one `WordClip` per `segment_words` source word, in order.

    `source_ranges` is parallel to the input tokens: entry `i` is the `[start, end)`
    char range of the word token `i` was mapped to, or `None` when the sentence has no
    source words at all.
    """

    words: tuple[WordClip, ...]
    source_ranges: tuple[tuple[int, int] | None, ...]


def segment_words(text: str) -> list[WordSpan]:
    """Split display text into source words on the shared word-boundary rule."""
    return [
        WordSpan(text=match.group(), start=match.start(), end=match.end())
        for match in _WORD.finditer(text)
    ]


def align(
    words: Sequence[WordSpan],
    tokens: Sequence[TokenTimestamp],
) -> AlignedSentence:
    """Regroup a sentence's timed tokens onto its source words.

    Args:
        words: The sentence's source words, in order -- from `segment_words`, or the
            exact word units the injector wrapped, so the SMIL and the DOM spans agree.
            Their `start`/`end` are the offsets `source_ranges` reports.
        tokens: The sentence's chapter-absolute token timestamps, in spoken order.

    Returns:
        The per-word clips and per-token source ranges (see `AlignedSentence`).

    Raises:
        ValueError: If `tokens` is empty. A sentence the backend produced no timestamps
            for cannot be word-aligned; the caller decides whether to fail or fall back
            to sentence granularity.
    """
    if not tokens:
        msg = "align needs at least one token to map onto source words"
        raise ValueError(msg)

    if not words:
        # Nothing to highlight (punctuation-only text); no token maps anywhere.
        return AlignedSentence(words=(), source_ranges=(None,) * len(tokens))

    assignment = _assign(tokens, [_key(word.text) for word in words])
    clips = _clips(words, tokens, assignment)
    ranges = tuple((words[index].start, words[index].end) for index in assignment)

    return AlignedSentence(words=tuple(clips), source_ranges=ranges)


def _key(text: str) -> str:
    # Compare tokens to words on their bare alphanumerics so punctuation, apostrophes,
    # and case never block a match ("don't" <-> "dont").
    return "".join(char.lower() for char in text if char.isalnum())


def _assign(tokens: Sequence[TokenTimestamp], keys: Sequence[str]) -> list[int]:
    # Walk tokens left to right. A token stays on the current word while that word still
    # wants it -- an exact match it has not yet received -- so a repeated word keeps its
    # own token; otherwise the token opens a later word.
    assignment: list[int] = []
    index = 0
    matched = [False] * len(keys)
    last = len(keys) - 1

    for token in tokens:
        key = _key(token.text)
        if key and index < last:
            wants_current = keys[index] == key and not matched[index]
            if not wants_current:
                index = _next_word(index, key, keys, matched=matched, last=last)

        assignment.append(index)
        if key and keys[index] == key:
            matched[index] = True

    return assignment


def _next_word(
    index: int,
    key: str,
    keys: Sequence[str],
    *,
    matched: Sequence[bool],
    last: int,
) -> int:
    # Jump to the nearest later word this token exactly voices: rescues a real word
    # after an expansion (e.g. "approx" after a "3.14" no token equals) and routes a
    # repeated word to its own slot. Failing that, step once past a word already
    # finished by an exact match; otherwise stay put (an expansion token).
    for later in range(index + 1, last + 1):
        if keys[later] == key:
            return later

    if matched[index] and keys[index] != key:
        return index + 1

    return index


def _clips(
    words: Sequence[WordSpan],
    tokens: Sequence[TokenTimestamp],
    assignment: Sequence[int],
) -> list[WordClip]:
    token_indices: list[list[int]] = [[] for _ in words]
    for token_index, word_index in enumerate(assignment):
        token_indices[word_index].append(token_index)

    clips: list[WordClip] = []
    cursor = tokens[0].start_s
    for word, owned in zip(words, token_indices, strict=True):
        if owned:
            # Begin where the previous word ended (absorbing any inter-token gap) so the
            # overlay stays gapless; end at this word's last token, never before begin.
            begin = cursor
            end = max(tokens[owned[-1]].end_s, begin)
            cursor = end
        else:
            begin = end = cursor  # an untimed word collapses to the running boundary

        clips.append(WordClip(span=word, clip_begin_s=begin, clip_end_s=end))

    return clips
