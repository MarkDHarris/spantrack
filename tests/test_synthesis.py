"""Tests for the per-sentence synthesis coordinator (drive + DSP + cache)."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spantrack.sentence_cache import SentenceCache
from spantrack.synthesis import synthesize_sentence
from spantrack.tts.backend import Pcm, TokenTimestamp


class _FakeBackend:
    """A backend that records its call count and returns fixed audio + tokens."""

    def __init__(
        self,
        pcm: NDArray[np.float32],
        tokens: tuple[TokenTimestamp, ...] | None,
        *,
        model_version: str = "kokoro-test",
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> None:
        self._pcm = pcm
        self._tokens = tokens
        self._model_version = model_version
        self._voice = voice
        self._speed = speed
        self.calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def speed(self) -> float:
        return self._speed

    def synthesize(self, text: str) -> tuple[Pcm, tuple[TokenTimestamp, ...] | None]:
        assert text
        self.calls += 1
        return self._pcm, self._tokens


def _padded_signal() -> NDArray[np.float32]:
    # 0.5s of leading silence, 0.5s of signal, 0.25s of trailing silence.
    return np.concatenate(
        [
            np.zeros(12_000, dtype=np.float32),
            np.full(12_000, 0.5, dtype=np.float32),
            np.zeros(6_000, dtype=np.float32),
        ],
    )


def _leading_signal() -> NDArray[np.float32]:
    # 0.5s of signal from the first sample, then 0.25s of trailing silence (no lead).
    return np.concatenate(
        [
            np.full(12_000, 0.5, dtype=np.float32),
            np.zeros(6_000, dtype=np.float32),
        ],
    )


def test_cache_miss_synthesizes_trims_and_stores(tmp_path: Path):
    tokens = (TokenTimestamp("hi", 0.5, 1.0, ""),)
    backend = _FakeBackend(_padded_signal(), tokens)
    cache = SentenceCache(root=tmp_path)

    pcm, out_tokens = synthesize_sentence(backend, cache, "hi")

    assert backend.calls == 1
    assert len(pcm) == 12_000  # leading + trailing silence trimmed
    # Timestamps shift earlier by the trimmed lead (0.5 s).
    assert out_tokens == (TokenTimestamp("hi", 0.0, 0.5, ""),)
    key = SentenceCache.key("kokoro-test", "af_heart", 1.0, "hi")
    assert cache.get(key) is not None


def test_cache_hit_skips_backend(tmp_path: Path):
    backend = _FakeBackend(_padded_signal(), (TokenTimestamp("hi", 0.5, 1.0, ""),))
    cache = SentenceCache(root=tmp_path)

    synthesize_sentence(backend, cache, "hi")
    synthesize_sentence(backend, cache, "hi")

    assert backend.calls == 1


def test_resume_returns_byte_identical_cached_audio(tmp_path: Path):
    tokens = (TokenTimestamp("hi", 0.5, 1.0, ""),)
    first_pcm, _ = synthesize_sentence(
        _FakeBackend(_padded_signal(), tokens),
        SentenceCache(root=tmp_path),
        "hi",
    )

    # Simulate a restart: a fresh cache + a backend with the same identity but different
    # audio. A correct resume returns the stored audio and never calls the backend.
    resumed = _FakeBackend(np.full(24_000, 0.9, dtype=np.float32), tokens)
    second_pcm, _ = synthesize_sentence(resumed, SentenceCache(root=tmp_path), "hi")

    assert resumed.calls == 0
    assert np.array_equal(second_pcm, first_pcm)


def test_none_tokens_pass_through(tmp_path: Path):
    backend = _FakeBackend(_padded_signal(), None)
    _, tokens = synthesize_sentence(backend, SentenceCache(root=tmp_path), "hi")

    assert tokens is None


def test_all_silence_trims_to_empty(tmp_path: Path):
    backend = _FakeBackend(np.zeros(8_000, dtype=np.float32), None)
    pcm, _ = synthesize_sentence(backend, SentenceCache(root=tmp_path), "hi")

    assert len(pcm) == 0


def test_all_silence_drops_stale_timestamps(tmp_path: Path):
    # Fully-silent audio has no content for the timestamps to describe; a
    # timestamp-capable backend's tokens must not survive as stale spans into empty
    # audio (they would describe time ranges the output no longer contains).
    tokens = (TokenTimestamp("hi", 0.0, 0.5, ""),)
    backend = _FakeBackend(np.zeros(8_000, dtype=np.float32), tokens)
    pcm, out_tokens = synthesize_sentence(backend, SentenceCache(root=tmp_path), "hi")

    assert len(pcm) == 0
    assert out_tokens == ()


def test_no_leading_silence_keeps_timestamps_unshifted(tmp_path: Path):
    # With no leading silence to trim (lead == 0), tokens pass through verbatim while
    # the trailing silence is still removed. Every other signal-bearing test leads with
    # silence, so this is the sole cover for the lead == 0 branch in _trim_silence.
    tokens = (TokenTimestamp("hi", 0.0, 0.5, ""),)
    backend = _FakeBackend(_leading_signal(), tokens)

    pcm, out_tokens = synthesize_sentence(backend, SentenceCache(root=tmp_path), "hi")

    assert len(pcm) == 12_000  # trailing silence trimmed; nothing to trim at the lead
    assert out_tokens == tokens  # unshifted, since lead == 0
