"""The Kokoro-82M TTS backend.

Kokoro builds its English G2P as `en.G2P(..., unk='')`, so any word misaki cannot look
up is phonemized to the empty string and silently deleted from the audio -- unless the
espeak-ng OOV fallback is present. Kokoro itself swallows a failed fallback (it logs a
warning and sets `fallback=None`), so this backend re-checks `pipeline.g2p.fallback` at
construction and fails loudly instead: a missing fallback drops not just rare names but
ordinary words like "acknowledgement".

`synthesize` delegates the >510-token chunking to Kokoro's own `en_tokenize` (which
splits preferentially at clause punctuation), then concatenates the per-chunk audio and
shifts each chunk's per-token timestamps by the running duration so the returned
timestamps are relative to the whole sentence.

The pipeline is injected so tests exercise every branch with a fake and never download
the model weights. `validate_lexicon` reads Kokoro's phoneme inventory -- the exact set
it drops from -- and `apply_lexicon` emits misaki's `[word](/phonemes/)` inline
override, by which a user lexicon bypasses the gold/silver dictionaries.
"""

import importlib.metadata
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from spantrack.lexicon import Lexicon
from spantrack.run_config import Device, RunConfig
from spantrack.tts.backend import SAMPLE_RATE, Pcm, TokenTimestamp

__all__ = ["KokoroBackend", "KokoroBackendError", "build"]

_REPO_ID = "hexgrad/Kokoro-82M"
_BACKEND_NAME = "kokoro"
_WORD = re.compile(r"[A-Za-z]+")

# Busts the sentence cache when the model changes: the pinned HF repo id plus the
# installed kokoro version is a stable proxy for the weights.
_MODEL_VERSION = f"{_REPO_ID}@{importlib.metadata.version('kokoro')}"


class KokoroBackendError(Exception):
    """Raised when the Kokoro backend cannot be built or a lexicon is invalid for it."""


class _Audio(Protocol):
    """A torch tensor's `.detach().cpu().numpy()` chain, narrowed to what we call."""

    def detach(self) -> "_Audio": ...

    def cpu(self) -> "_Audio": ...

    def numpy(self) -> NDArray[Any]: ...


class _Token(Protocol):
    """One of Kokoro's `MToken`s, as yielded by the pipeline.

    `start_ts`/`end_ts` mirror misaki's own field names (seconds); `_shift_tokens`
    reads them off the real token at runtime, so they must not be renamed to the `_s`
    suffix used by `TokenTimestamp`.
    """

    @property
    def text(self) -> str: ...

    @property
    def whitespace(self) -> str: ...

    @property
    def start_ts(self) -> float | None: ...

    @property
    def end_ts(self) -> float | None: ...


class _Result(Protocol):
    @property
    def audio(self) -> _Audio | None: ...

    @property
    def tokens(self) -> Sequence[_Token] | None: ...


class _G2P(Protocol):
    @property
    def fallback(self) -> object | None: ...


class _Model(Protocol):
    @property
    def vocab(self) -> Mapping[str, int]: ...


class _Pipeline(Protocol):
    """The slice of Kokoro's `KPipeline` this backend depends on."""

    @property
    def g2p(self) -> _G2P: ...

    @property
    def model(self) -> _Model: ...

    def __call__(self, text: str, *, voice: str, speed: float) -> Iterator[_Result]: ...


class KokoroBackend:
    """A Kokoro-82M text-to-speech backend driving an injected `KPipeline`."""

    def __init__(self, pipeline: _Pipeline, *, voice: str, speed: float) -> None:
        """Wrap a Kokoro pipeline, refusing one whose espeak OOV fallback is missing.

        Raises:
            KokoroBackendError: When the pipeline's espeak fallback failed to construct;
                Kokoro would then silently drop every out-of-dictionary word.
        """
        if pipeline.g2p.fallback is None:
            msg = (
                "Kokoro's espeak-ng OOV fallback is not available; without it every "
                "out-of-dictionary word is silently dropped from the audio. Check that "
                "espeakng-loader installed correctly."
            )
            raise KokoroBackendError(msg)

        self._pipeline = pipeline
        self._voice = voice
        self._speed = speed

    @property
    def name(self) -> str:
        """The backend's identifier, recorded in the chapter sidecar."""
        return _BACKEND_NAME

    @property
    def model_version(self) -> str:
        """A cache-key identifier for the acoustic model this backend drives."""
        return _MODEL_VERSION

    @property
    def voice(self) -> str:
        """The Kokoro voice this backend synthesizes with."""
        return self._voice

    @property
    def speed(self) -> float:
        """The speech-speed multiplier this backend synthesizes with."""
        return self._speed

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        """Reject any lexicon phoneme outside Kokoro's vocab, which it would drop.

        Raises:
            KokoroBackendError: When an entry contains a phoneme character absent from
                Kokoro's inventory.
        """
        inventory = frozenset(self._pipeline.model.vocab)
        for word, phonemes in lexicon.entries.items():
            if unknown := set(phonemes) - inventory:
                shown = "".join(sorted(unknown))

                msg = f"{word!r}: phonemes outside Kokoro's inventory: {shown!r}"
                raise KokoroBackendError(msg)

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        """Rewrite lexicon-matched words to Kokoro's `[word](/phonemes/)` override."""

        def override(match: re.Match[str]) -> str:
            phonemes = lexicon.get(match.group(0))
            if phonemes is None:
                return match.group(0)

            return f"[{match.group(0)}](/{phonemes}/)"

        return _WORD.sub(override, text)

    def synthesize(self, text: str) -> tuple[Pcm, tuple[TokenTimestamp, ...] | None]:
        """Synthesize `text` to 24 kHz float32 PCM with per-token timestamps.

        Kokoro chunks text longer than 510 phoneme tokens internally; this concatenates
        the chunk audio and offsets each chunk's timestamps by the running duration.
        """
        chunks: list[Pcm] = []
        tokens: list[TokenTimestamp] = []
        offset = 0.0
        for result in self._pipeline(text, voice=self._voice, speed=self._speed):
            audio = result.audio
            if audio is None:
                continue

            pcm = np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)
            tokens.extend(_shift_tokens(result.tokens or (), offset))
            chunks.append(pcm)
            offset += len(pcm) / SAMPLE_RATE

        merged = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        return merged, tuple(tokens)


def _shift_tokens(
    tokens: Sequence[_Token],
    offset: float,
) -> Iterator[TokenTimestamp]:
    """Yield the timed tokens shifted by `offset`, skipping untimed ones.

    Whitespace and punctuation tokens carry no timestamps (Kokoro leaves `start_ts`
    None), so they are dropped rather than emitted as zero-length spans.
    """
    for token in tokens:
        if token.start_ts is None or token.end_ts is None:
            continue

        yield TokenTimestamp(
            text=token.text,
            start_s=offset + token.start_ts,
            end_s=offset + token.end_ts,
            whitespace=token.whitespace,
        )


def build(config: RunConfig) -> KokoroBackend:
    """Construct a Kokoro backend backed by the real model weights.

    It wires the concrete `KPipeline`, which downloads the model on first use.
    """
    from kokoro import KPipeline  # pyright: ignore[reportMissingTypeStubs]

    device = None if config.device is Device.AUTO else config.device.value
    pipeline = KPipeline(lang_code=config.lang_code, repo_id=_REPO_ID, device=device)
    return KokoroBackend(
        cast("_Pipeline", pipeline),
        voice=config.voice,
        speed=config.speed,
    )
