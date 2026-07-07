"""Integration test for the per-chapter narration wiring (real ffmpeg encode)."""

import math
from pathlib import Path

import numpy as np
import pytest
from conftest import HAS_FFMPEG
from lxml import etree
from numpy.typing import NDArray
from xhtml_documents import parse_body as _parse

from spantrack import chapter_sidecar, pipeline
from spantrack.lexicon import Lexicon
from spantrack.run_config import AudioFormat, Granularity
from spantrack.sentence_cache import SentenceCache
from spantrack.tts.backend import TokenTimestamp

_SMIL = "http://www.w3.org/ns/SMIL"

pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


class _FakeBackend:
    """A backend returning a fixed 0.5s tone + one token per sentence, no weights."""

    name = "fake"
    model_version = "fake@1"
    voice = "v1"
    speed = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        _ = lexicon

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        _ = lexicon
        return text

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        self.calls += 1
        word = words[0] if (words := text.split()) else text
        pcm = np.full(12_000, 0.5, dtype=np.float32)  # 0.5s at 24 kHz, all voiced

        return pcm, (TokenTimestamp(word, 0.0, 0.5, ""),)


def _context(backend: _FakeBackend, cache_root: Path) -> pipeline.NarrationContext:
    return pipeline.NarrationContext(
        backend=backend,
        cache=SentenceCache(root=cache_root),
        lexicon=None,
        spantrack_version="0.1.0",
        bitrate=64_000,
        audio_format=AudioFormat.MP3,
    )


def _targets(tmp_path: Path, stem: str) -> pipeline.DocumentTargets:
    return pipeline.DocumentTargets(
        doc_href=f"{stem}.xhtml",
        audio_href=f"{stem}.mp3",
        audio_path=tmp_path / "audio" / f"{stem}.mp3",
        smil_path=tmp_path / f"{stem}.smil",
        sidecar_path=tmp_path / "sidecars" / f"{stem}.tokens.json",
    )


def test_narrate_document_writes_mp3_smil_and_sidecar(tmp_path: Path):
    root = _parse("<p>First sentence. Second sentence.</p>")
    context = _context(_FakeBackend(), tmp_path / "cache")
    targets = _targets(tmp_path, "chapter1")

    result = pipeline.narrate_document(context, root, targets)

    assert result.sentence_count == 2
    assert targets.audio_path.exists()  # real ffmpeg produced the mp3

    pars = etree.fromstring(targets.smil_path.read_bytes()).findall(
        f".//{{{_SMIL}}}par"
    )
    assert len(pars) == 2

    sidecar = chapter_sidecar.read(targets.sidecar_path)
    assert sidecar.backend.name == "fake"
    assert len(sidecar.sentences) == 2

    first, second = sidecar.sentences
    assert first.tokens is not None
    assert second.tokens is not None

    # Tokens are shifted to chapter-absolute time by their clip begin.
    assert math.isclose(first.tokens[0].start_s, 0.0)
    assert math.isclose(second.clip_begin_s, 0.75)  # 0.5s tone + 0.25s sentence gap
    assert math.isclose(second.tokens[0].start_s, 0.75)


def test_narrate_document_reuses_cached_audio(tmp_path: Path):
    backend = _FakeBackend()
    context = _context(backend, tmp_path / "cache")

    pipeline.narrate_document(
        context,
        _parse("<p>Same text here.</p>"),
        _targets(tmp_path, "a"),
    )
    after_first = backend.calls

    pipeline.narrate_document(
        context,
        _parse("<p>Same text here.</p>"),
        _targets(tmp_path, "b"),
    )

    assert backend.calls == after_first  # identical text hits the cache


class _WordFake(_FakeBackend):
    """Word-mode fake: a 0.5s tone + one evenly-timed token per source word."""

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        pcm = np.full(12_000, 0.5, dtype=np.float32)
        words = text.split() or [text]
        step = 0.5 / len(words)
        tokens = tuple(
            TokenTimestamp(word, index * step, (index + 1) * step, " ")
            for index, word in enumerate(words)
        )

        return pcm, tokens


class _SilentFake(_FakeBackend):
    """A fake whose audio is fully silent, so the DSP trims tokens to an empty tuple."""

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        return np.zeros(12_000, dtype=np.float32), (TokenTimestamp(text, 0.0, 0.5, ""),)


class _LastSilentFake(_FakeBackend):
    """Voice each sentence except one containing 'silent', trimmed to silence."""

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        if "silent" in text.lower():
            return np.zeros(12_000, dtype=np.float32), (
                TokenTimestamp(text, 0.0, 0.5, ""),
            )

        return super().synthesize(text)


def _word_context(backend: _FakeBackend, cache_root: Path) -> pipeline.NarrationContext:
    return pipeline.NarrationContext(
        backend=backend,
        cache=SentenceCache(root=cache_root),
        lexicon=None,
        spantrack_version="0.1.0",
        bitrate=64_000,
        audio_format=AudioFormat.MP3,
        granularity=Granularity.WORD,
    )


def test_word_mode_writes_word_level_pars(tmp_path: Path):
    root = _parse("<p>First second. Third fourth.</p>")
    targets = _targets(tmp_path, "chapter1")

    result = pipeline.narrate_document(
        _word_context(_WordFake(), tmp_path / "cache"), root, targets
    )

    assert result.sentence_count == 2
    assert targets.audio_path.exists()

    smil_text = targets.smil_path.read_text(encoding="utf-8")
    assert "#s00001-w0" in smil_text
    assert "#s00001-w1" in smil_text

    sidecar = chapter_sidecar.read(targets.sidecar_path)
    assert sidecar.sentences[0].source_ranges is not None


def test_word_mode_skips_a_wordless_chapter(tmp_path: Path):
    targets = _targets(tmp_path, "divider")

    result = pipeline.narrate_document(
        _word_context(_WordFake(), tmp_path / "cache"), _parse("<p>* * *</p>"), targets
    )

    assert result.sentence_count == 0
    assert not targets.audio_path.exists()  # nothing encoded, nothing orphaned


def test_word_mode_skips_a_silent_chapter_without_crashing(tmp_path: Path):
    targets = _targets(tmp_path, "silent")

    result = pipeline.narrate_document(
        _word_context(_SilentFake(), tmp_path / "cache"),
        _parse("<p>Hello world.</p>"),
        targets,
    )

    assert result.sentence_count == 0  # empty-tuple tokens -> no par, chapter skipped
    assert not targets.audio_path.exists()


def test_sentence_mode_skips_a_fully_silent_chapter(tmp_path: Path):
    targets = _targets(tmp_path, "silent")

    result = pipeline.narrate_document(
        _context(_SilentFake(), tmp_path / "cache"),
        _parse("<p>Hello world.</p>"),  # a single sentence that trims to silence
        targets,
    )

    # Mirrors word mode: a chapter with nothing highlightable is skipped, never written
    # as a SMIL whose only <par> has clipBegin == clipEnd (invalid, epubcheck MED-009).
    assert result.sentence_count == 0
    assert not targets.audio_path.exists()
    assert not targets.smil_path.exists()


def test_sentence_mode_drops_the_zero_length_par_of_a_trailing_silent_sentence(
    tmp_path: Path,
):
    root = _parse("<p>First sentence here. This one is silent.</p>")
    targets = _targets(tmp_path, "chapter1")

    result = pipeline.narrate_document(
        _context(_LastSilentFake(), tmp_path / "cache"),
        root,
        targets,
    )

    # The voiced sentence is highlighted; the trailing silent sentence's clip is
    # zero-length, so it emits no <par> yet is still recorded in the sidecar.
    pars = etree.fromstring(targets.smil_path.read_bytes()).findall(
        f".//{{{_SMIL}}}par"
    )
    assert len(pars) == 1
    assert result.sentence_count == 2
    assert len(chapter_sidecar.read(targets.sidecar_path).sentences) == 2
