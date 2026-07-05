"""Tests for the Kokoro backend: construction, lexicon hooks, espeak assertion, and
`synthesize` (chunk concatenation + cumulative timestamp mapping).

The unit tests inject a fake pipeline so no acoustic weights load; one integration test
builds a real English pipeline with `model=False` (no weights) to pin the `g2p.fallback`
contract against the actual misaki/kokoro release.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import kokoro  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.typing import NDArray

from spantrack.lexicon import Lexicon, load
from spantrack.tts.backend import SAMPLE_RATE, TokenTimestamp, TTSBackend
from spantrack.tts.kokoro_backend import KokoroBackend, KokoroBackendError

# A stand-in phoneme inventory: ascii letters and space. The real Kokoro vocab is IPA.
_VOCAB = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
_FALLBACK_PRESENT = object()


class _FakeG2P:
    def __init__(self, fallback: object | None) -> None:
        self.fallback = fallback


class _FakeModel:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab


class _FakeAudio:
    """Mimics a torch tensor's `.detach().cpu().numpy()` conversion chain."""

    def __init__(self, samples: NDArray[np.float32]) -> None:
        self._samples = samples

    def detach(self) -> "_FakeAudio":
        return self

    def cpu(self) -> "_FakeAudio":
        return self

    def numpy(self) -> NDArray[np.float32]:
        return self._samples


class _FakeToken:
    def __init__(
        self,
        text: str,
        whitespace: str,
        start_ts: float | None,
        end_ts: float | None,
    ) -> None:
        self.text = text
        self.whitespace = whitespace
        self.start_ts = start_ts
        self.end_ts = end_ts


class _FakeResult:
    def __init__(
        self,
        samples: NDArray[np.float32],
        tokens: Sequence[_FakeToken],
    ) -> None:
        self.audio = _FakeAudio(samples)
        self.tokens = tokens


class _FakePipeline:
    def __init__(
        self,
        *,
        fallback: object | None = _FALLBACK_PRESENT,
        vocab: dict[str, int] | None = None,
        results: Sequence[_FakeResult] = (),
    ) -> None:
        self.g2p = _FakeG2P(fallback)
        self.model = _FakeModel(_VOCAB if vocab is None else vocab)
        self._results = tuple(results)

    def __call__(self, text: str, *, voice: str, speed: float) -> Iterator[_FakeResult]:
        assert text
        assert voice
        assert speed > 0
        return iter(self._results)


def _backend(
    *,
    fallback: object | None = _FALLBACK_PRESENT,
    vocab: dict[str, int] | None = None,
    results: Sequence[_FakeResult] = (),
) -> KokoroBackend:
    pipeline = _FakePipeline(fallback=fallback, vocab=vocab, results=results)
    return KokoroBackend(pipeline, voice="af_heart", speed=1.0)


def _silence(seconds: float) -> NDArray[np.float32]:
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)


def _lex(tmp_path: Path, content: str) -> Lexicon:
    path = tmp_path / "lexicon.toml"
    path.write_text(content, encoding="utf-8")

    return load(path)


def test_construction_requires_espeak_fallback():
    with pytest.raises(KokoroBackendError, match="espeak"):
        _backend(fallback=None)


def test_construction_succeeds_with_fallback():
    assert isinstance(_backend(), KokoroBackend)


def test_backend_satisfies_tts_protocol():
    assert isinstance(_backend(), TTSBackend)


def test_exposes_cache_identity():
    backend = _backend()
    assert backend.voice == "af_heart"
    assert backend.speed == 1.0
    assert backend.model_version.startswith("hexgrad/Kokoro-82M@")


def test_validate_lexicon_accepts_in_inventory(tmp_path: Path):
    _backend().validate_lexicon(_lex(tmp_path, '[words]\ngif = "jif"\n'))


def test_validate_lexicon_rejects_out_of_inventory(tmp_path: Path):
    # '3' is not in the stand-in vocab, so Kokoro would silently drop it.
    lexicon = _lex(tmp_path, '[words]\ngif = "d3if"\n')
    with pytest.raises(KokoroBackendError, match="inventory"):
        _backend().validate_lexicon(lexicon)


def test_apply_lexicon_rewrites_case_insensitively(tmp_path: Path):
    lexicon = _lex(tmp_path, '[words]\ngif = "jif"\n')
    assert _backend().apply_lexicon("a GIF here", lexicon) == "a [GIF](/jif/) here"


def test_apply_lexicon_leaves_unmatched(tmp_path: Path):
    lexicon = _lex(tmp_path, '[words]\ngif = "jif"\n')
    assert _backend().apply_lexicon("nothing to see", lexicon) == "nothing to see"


def test_synthesize_single_chunk():
    result = _FakeResult(
        _silence(1.0),
        [_FakeToken("Hello", " ", 0.0, 0.5), _FakeToken("world", "", 0.5, 0.75)],
    )
    pcm, tokens = _backend(results=[result]).synthesize("Hello world")

    assert pcm.dtype == np.float32
    assert len(pcm) == SAMPLE_RATE
    assert tokens == (
        TokenTimestamp("Hello", 0.0, 0.5, " "),
        TokenTimestamp("world", 0.5, 0.75, ""),
    )


def test_synthesize_concatenates_and_shifts_timestamps():
    first = _FakeResult(
        _silence(1.0),
        [_FakeToken("Hello", " ", 0.0, 0.5), _FakeToken("world", "", 0.5, 0.75)],
    )
    second = _FakeResult(_silence(0.5), [_FakeToken("Bye", "", 0.0, 0.25)])
    pcm, tokens = _backend(results=[first, second]).synthesize("Hello world Bye")

    # Second chunk's timestamps shift by the first chunk's duration (1.0 s).
    assert len(pcm) == round(1.5 * SAMPLE_RATE)
    assert tokens == (
        TokenTimestamp("Hello", 0.0, 0.5, " "),
        TokenTimestamp("world", 0.5, 0.75, ""),
        TokenTimestamp("Bye", 1.0, 1.25, ""),
    )


def test_synthesize_skips_untimed_tokens():
    result = _FakeResult(
        _silence(0.5),
        [_FakeToken("word", "", 0.0, 0.5), _FakeToken(" ", " ", None, None)],
    )
    _, tokens = _backend(results=[result]).synthesize("word")

    assert tokens == (TokenTimestamp("word", 0.0, 0.5, ""),)


def test_synthesize_no_results_yields_empty():
    pcm, tokens = _backend(results=[]).synthesize("unspeakable")

    assert len(pcm) == 0
    assert pcm.dtype == np.float32
    assert tokens == ()


def test_real_pipeline_carries_espeak_fallback():
    # Offline integration: a real English pipeline (model=False, no weights) builds the
    # espeak fallback, so the backend accepts it -- guards the g2p.fallback contract
    # against a misaki/kokoro bump.
    pipeline = kokoro.KPipeline(
        lang_code="a", model=False, repo_id="hexgrad/Kokoro-82M"
    )
    # A real KPipeline's `.model` is `KModel | bool`, so it can't statically satisfy the
    # backend's protocol; the fallback assertion only reads `.g2p.fallback` at runtime.
    backend = KokoroBackend(cast("Any", pipeline), voice="af_heart", speed=1.0)
    assert isinstance(backend, KokoroBackend)
