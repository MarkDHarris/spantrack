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

`synthesize` returns audio in the backend-neutral canonical form (`Pcm`) plus optional
per-token timestamps; everything downstream (cache, assembly, SMIL) consumes those, so
the torch/misaki types never leak past this module.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from spantrack.lexicon import Lexicon

__all__ = ["SAMPLE_RATE", "Pcm", "TTSBackend", "TokenTimestamp"]

# The canonical audio form every backend emits and the cache/assembler consume: 24 kHz
# mono float32 samples in [-1, 1].
type Pcm = NDArray[np.float32]

# Kokoro synthesizes at 24 kHz; the whole pipeline stays at that rate, so `Pcm` sample
# counts convert to seconds through this one constant.
SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class TokenTimestamp:
    """One token's timing within a synthesized sentence.

    `start_s`/`end_s` are seconds from the start of the sentence's own audio (the
    assembler later shifts them by the chapter's cumulative offset). `whitespace` is
    the run that followed the token in the source, kept so word-level SMIL can
    reconstruct spacing without re-tokenizing.
    """

    text: str
    start_s: float
    end_s: float
    whitespace: str


@runtime_checkable
class TTSBackend(Protocol):
    """A pluggable text-to-speech backend."""

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        """Raise if any lexicon phoneme string is outside the backend's inventory."""
        ...

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        """Rewrite `text`'s lexicon-matched words to the backend's override form."""
        ...

    def synthesize(self, text: str) -> tuple[Pcm, tuple[TokenTimestamp, ...] | None]:
        """Synthesize `text` to 24 kHz mono float32 PCM and per-token timestamps.

        Returns:
            The PCM samples and per-token timestamps, or `None` timestamps for a
            backend that emits none.
        """
        ...
