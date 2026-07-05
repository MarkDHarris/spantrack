"""Integration test for the per-chapter narration wiring (real ffmpeg encode)."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from lxml import etree
from numpy.typing import NDArray

from spantrack import chapter_sidecar, pipeline
from spantrack.lexicon import Lexicon
from spantrack.lxml_reading import XmlElement, hardened_parser
from spantrack.sentence_cache import SentenceCache
from spantrack.tts.backend import TokenTimestamp

_XHTML = "http://www.w3.org/1999/xhtml"
_SMIL = "http://www.w3.org/ns/SMIL"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed",
)


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


def _parse(body: str) -> XmlElement:
    doc = (
        f'<html xmlns="{_XHTML}"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )
    return etree.fromstring(doc.encode(), hardened_parser())


def _context(backend: _FakeBackend, cache_root: Path) -> pipeline.NarrationContext:
    return pipeline.NarrationContext(
        backend=backend,
        cache=SentenceCache(root=cache_root),
        lexicon=None,
        spantrack_version="0.1.0",
        bitrate=64_000,
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
