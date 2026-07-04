"""The pluggable TTS backend boundary.

`TTSBackend` is the seam that keeps engine-specific behavior -- the misaki phoneme
inventory and the `[word](/phonemes/)` override syntax -- out of the rest of the
pipeline. The two lexicon hooks live here because a lexicon's phoneme strings are only
meaningful to a specific backend:

- `validate_lexicon` rejects a lexicon whose phonemes fall outside the backend's
  inventory. Kokoro silently drops unknown phonemes, so this must fail loudly at load.
- `apply_lexicon` rewrites a matched word to the backend's inline override form. For
  Kokoro that is `[word](/phonemes/)`, which bypasses misaki's gold/silver dictionary
  lookup -- the mechanism by which the user lexicon takes precedence, no merge logic.
"""

from typing import Protocol, runtime_checkable

from spantrack.lexicon import Lexicon

__all__ = ["TTSBackend"]


@runtime_checkable
class TTSBackend(Protocol):
    """A pluggable text-to-speech backend."""

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        """Raise if any lexicon phoneme string is outside the backend's inventory."""
        ...

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        """Rewrite `text`'s lexicon-matched words to the backend's override form."""
        ...
