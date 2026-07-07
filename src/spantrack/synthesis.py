"""Per-sentence synthesis: cache coordination plus backend-agnostic post-processing.

`synthesize_sentence` is where resume becomes emergent: it derives the content-addressed
key from the backend's identity and the text, returns a cache hit untouched, and only on
a miss drives the backend, trims the leading/trailing silence, shifts the token
timestamps to match, and stores the canonical result. Because the silence trim runs
before the cache write, cached audio is canonical regardless of backend, and the
assembler can insert its own exact gaps between sentences.
"""

from typing import Protocol

import numpy as np

from spantrack.sentence_cache import SentenceCache
from spantrack.tts.backend import (
    SAMPLE_RATE,
    Pcm,
    Synthesis,
    TokenTimestamp,
    shift_tokens,
)

__all__ = ["synthesize_sentence"]

# Samples with magnitude at or below this count as silence. Kokoro pads utterances with
# near-zero lead-in/out; trimming it lets the assembler control inter-sentence gaps.
_SILENCE_THRESHOLD = 1e-4


class _Backend(Protocol):
    """The backend surface the synthesis stage drives, plus its cache identity."""

    @property
    def model_version(self) -> str: ...

    @property
    def voice(self) -> str: ...

    @property
    def speed(self) -> float: ...

    def synthesize(self, text: str) -> Synthesis: ...


def synthesize_sentence(
    backend: _Backend,
    cache: SentenceCache,
    text: str,
) -> tuple[str, Pcm, tuple[TokenTimestamp, ...] | None]:
    """Return `text`'s cache key, canonical audio, and timestamps; synthesize on a miss.

    Args:
        backend: The TTS backend; its `model_version`/`voice`/`speed` key the cache.
        cache: The content-addressed sentence store.
        text: The exact synthesis input (already normalized and lexicon-applied), which
            is also the text component of the cache key.

    Returns:
        The content-addressed cache key, the silence-trimmed PCM, and the timestamps
        shifted to match (`None` for a backend that emits none). The key is returned so
        the caller reuses it instead of re-deriving the same hash from the same inputs.
    """
    key = cache.key(backend.model_version, backend.voice, backend.speed, text)
    cached = cache.get(key)
    if cached is not None:
        return key, *cached

    pcm, tokens = backend.synthesize(text)
    trimmed, shifted = _trim_silence(pcm, tokens)
    cache.put(key, trimmed, shifted)

    return key, trimmed, shifted


def _trim_silence(
    pcm: Pcm,
    tokens: tuple[TokenTimestamp, ...] | None,
) -> Synthesis:
    """Trim leading/trailing silence; shift token timestamps by the trimmed lead.

    Fully-silent audio yields empty PCM and no tokens: any timestamps would describe
    content the output no longer contains, so a timestamp-capable backend's tokens
    collapse to `()` while a token-less backend stays `None`.
    """
    loud = np.flatnonzero(np.abs(pcm) > _SILENCE_THRESHOLD)
    if loud.size == 0:
        return pcm[:0], None if tokens is None else ()

    lead = int(loud[0])
    trimmed = pcm[lead : int(loud[-1]) + 1]
    if lead == 0:
        return trimmed, tokens

    return trimmed, shift_tokens(tokens, -lead / SAMPLE_RATE)
