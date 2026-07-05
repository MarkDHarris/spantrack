"""Tests for the versioned per-chapter token-timestamp sidecar."""

import json
from pathlib import Path

import pytest

from spantrack import chapter_sidecar
from spantrack.chapter_sidecar import (
    BackendIdentity,
    ChapterAudio,
    ChapterSidecar,
    SidecarSentence,
)
from spantrack.tts.backend import SAMPLE_RATE, TokenTimestamp


def _sidecar() -> ChapterSidecar:
    return ChapterSidecar(
        spantrack_version="0.1.0",
        backend=BackendIdentity(
            name="kokoro",
            model_version="hexgrad/Kokoro-82M@0.9.4",
            voice="af_heart",
            speed=1.0,
        ),
        doc_href="OEBPS/chapter1.xhtml",
        audio=ChapterAudio(
            file="audio/chapter1.mp3",
            sample_rate=SAMPLE_RATE,
            duration_s=12.5,
        ),
        sentences=(
            SidecarSentence(
                span_id="s00001",
                clip_begin_s=0.0,
                clip_end_s=1.25,
                display_text="Call me Ishmael.",
                normalized_text="Call me Ishmael.",
                cache_key="abc123",
                tokens=(
                    TokenTimestamp("Call", 0.0, 0.3, " "),
                    TokenTimestamp("me", 0.3, 0.5, " "),
                    TokenTimestamp("Ishmael", 0.5, 1.1, ""),
                ),
            ),
            SidecarSentence(
                span_id="s00002",
                clip_begin_s=1.25,
                clip_end_s=2.5,
                display_text="Some years ago.",
                normalized_text="Some years ago.",
                cache_key="def456",
                tokens=None,
            ),
        ),
    )


def test_round_trip_preserves_the_sidecar(tmp_path: Path):
    path = tmp_path / "chapter1.tokens.json"
    original = _sidecar()

    chapter_sidecar.write(path, original)

    assert chapter_sidecar.read(path) == original


def test_written_json_carries_schema_version(tmp_path: Path):
    path = tmp_path / "c.tokens.json"

    chapter_sidecar.write(path, _sidecar())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_none_tokens_round_trip(tmp_path: Path):
    path = tmp_path / "c.tokens.json"

    chapter_sidecar.write(path, _sidecar())

    # A timestamp-less backend stores `null`, which must come back as None, not ().
    assert chapter_sidecar.read(path).sentences[1].tokens is None


def test_rewrite_is_byte_identical(tmp_path: Path):
    path = tmp_path / "c.tokens.json"
    chapter_sidecar.write(path, _sidecar())
    first = path.read_bytes()

    chapter_sidecar.write(path, chapter_sidecar.read(path))

    assert path.read_bytes() == first


def test_unknown_schema_version_raises(tmp_path: Path):
    path = tmp_path / "c.tokens.json"
    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    with pytest.raises(chapter_sidecar.SidecarError, match="schema_version"):
        chapter_sidecar.read(path)


def test_malformed_json_raises(tmp_path: Path):
    path = tmp_path / "c.tokens.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(chapter_sidecar.SidecarError):
        chapter_sidecar.read(path)


def test_missing_required_key_raises(tmp_path: Path):
    path = tmp_path / "c.tokens.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(chapter_sidecar.SidecarError):
        chapter_sidecar.read(path)
