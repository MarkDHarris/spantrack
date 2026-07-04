"""Load a backend-neutral pronunciation lexicon from a TOML file.

The file's only top-level table is `words`, a set of `word = "phonemes"` string entries
matched case-insensitively. This module validates the file's shape and nothing more.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

__all__ = ["Lexicon", "LexiconError", "load"]


class LexiconError(Exception):
    """Raised when a lexicon file is unreadable, malformed, or violates the schema."""


@dataclass(frozen=True, slots=True)
class Lexicon:
    """A case-insensitive pronunciation lexicon: lowercased word -> phoneme string.

    Backend-neutral: phoneme strings are stored verbatim, never validated here.
    """

    entries: dict[str, str]

    def get(self, word: str) -> str | None:
        """Return the phoneme override for `word` (case-insensitive), or None."""
        return self.entries.get(word.lower())


def load(path: Path) -> Lexicon:
    """Load a pronunciation lexicon from a TOML file.

    Args:
        path: The `.toml` lexicon file. Its only top-level table is `words`, a set of
            `word = "phonemes"` entries; keys match case-insensitively. An empty or
            absent `words` table yields an empty lexicon.

    Returns:
        The parsed lexicon.

    Raises:
        LexiconError: If the file cannot be read, is not valid TOML, has an unexpected
            top-level key, or a `words` entry is not a string or collides on case.
    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        msg = f"cannot read lexicon {path}: {exc}"
        raise LexiconError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid TOML in lexicon {path}: {exc}"
        raise LexiconError(msg) from exc

    if unexpected := data.keys() - {"words"}:
        keys = ", ".join(sorted(unexpected))
        msg = f"unexpected top-level key(s) in lexicon: {keys}"
        raise LexiconError(msg)

    words = data.get("words", {})
    if not isinstance(words, dict):
        msg = "lexicon 'words' must be a table"
        raise LexiconError(msg)

    entries: dict[str, str] = {}
    for word, value in cast("dict[str, object]", words).items():
        if not isinstance(value, str):
            msg = f"lexicon entry {word!r} must be a string, not {type(value).__name__}"
            raise LexiconError(msg)

        key = word.lower()
        if key in entries:
            msg = f"duplicate case-insensitive lexicon key {key!r}"
            raise LexiconError(msg)

        entries[key] = value

    return Lexicon(entries=entries)
