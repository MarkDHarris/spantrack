"""Find out-of-vocabulary words so the user can curate lexicon overrides.

A fallback-disabled misaki G2P phonemizes what its gold/silver dictionaries know and
leaves every other word unphonemized (`phonemes is None`) instead of letting espeak
guess. A token with no phonemes and a letter in its text is therefore a word the user
should add to the lexicon. The `rating` attribute is unreliable in misaki 0.9.4, so
`phonemes is None` is the signal we key on.

The G2P is injected into `find_oov` so the collector is testable without loading misaki;
`build_g2p` is the production seam that constructs the real one.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol, cast

__all__ = ["build_g2p", "find_oov"]


class _Token(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def phonemes(self) -> str | None: ...


class _G2P(Protocol):
    def __call__(self, text: str) -> tuple[str, Sequence[_Token]]: ...


def find_oov(texts: Iterable[str], g2p: _G2P) -> list[str]:
    """Return the distinct words `g2p` cannot phonemize, in first-seen order.

    Args:
        texts: Normalized sentence texts to scan.
        g2p: A fallback-disabled grapheme-to-phoneme callable; a word it cannot look up
            carries `phonemes is None`.

    Returns:
        The out-of-vocabulary words, deduplicated case-insensitively (the first spelling
        kept), in the order first seen.
    """
    seen: dict[str, str] = {}
    for text in texts:
        _, tokens = g2p(text)
        for token in tokens:
            if token.phonemes is None and any(ch.isalpha() for ch in token.text):
                seen.setdefault(token.text.lower(), token.text)

    return list(seen.values())


def build_g2p(*, british: bool) -> _G2P:
    """Construct a fallback-disabled misaki English G2P for the OOV pass.

    With `fallback=None` an out-of-dictionary word is left unphonemized rather than
    guessed by espeak, which is exactly the signal the report needs.

    Args:
        british: Select the British English pronunciations instead of American.
    """
    from misaki import en  # pyright: ignore[reportMissingTypeStubs]

    return cast("_G2P", en.G2P(fallback=None, british=british))
