"""Tests for chapter audio assembly (gap insertion, edge fades, clip offsets)."""

import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from spantrack.audio_assembly import (
    BreakAfter,
    CompletedCommand,
    FfmpegEncodeError,
    FfmpegNotFoundError,
    assemble,
    encode_chapter,
)
from spantrack.tts.backend import SAMPLE_RATE


def _tone(seconds: float) -> np.ndarray:
    """A full-amplitude constant signal of the given duration (24 kHz float32)."""
    return np.full(round(seconds * SAMPLE_RATE), 0.5, dtype=np.float32)


def test_sentence_break_inserts_gap_and_meets_gapless():
    u0 = _tone(0.1)  # 2400 samples
    u1 = _tone(0.2)  # 4800 samples

    chapter = assemble([(u0, BreakAfter.SENTENCE), (u1, BreakAfter.SENTENCE)])
    # 2400 + 250ms gap (6000) + 4800, with no trailing gap after the last unit.
    assert len(chapter.pcm) == 2400 + 6000 + 4800
    assert np.all(
        chapter.pcm[2400 : 2400 + 6000] == 0.0
    )  # the inter-unit gap is silence

    first, second = chapter.clips
    assert math.isclose(first.clip_begin_s, 0.0)
    assert math.isclose(first.clip_end_s, second.clip_begin_s)  # gapless
    assert math.isclose(second.clip_end_s, chapter.duration_s)


@pytest.mark.parametrize(
    ("break_after", "gap_samples"),
    [
        (BreakAfter.SENTENCE, 6_000),
        (BreakAfter.PARAGRAPH, 14_400),
        (BreakAfter.HEADING, 21_600),
    ],
)
def test_break_type_sets_gap_duration(break_after: BreakAfter, gap_samples: int):
    u = _tone(0.1)

    chapter = assemble([(u, break_after), (u, break_after)])
    assert len(chapter.pcm) == 2400 + gap_samples + 2400


def test_last_unit_has_no_trailing_gap():
    u = _tone(0.1)

    chapter = assemble([(u, BreakAfter.HEADING)])
    # A single unit's heading break must not append 900ms of trailing silence.
    assert len(chapter.pcm) == 2400
    assert math.isclose(chapter.clips[0].clip_end_s, chapter.duration_s)
    assert math.isclose(chapter.duration_s, 0.1)


def test_edge_fade_attenuates_unit_boundaries():
    chapter = assemble([(_tone(0.1), BreakAfter.SENTENCE)])

    assert chapter.pcm[0] == 0.0  # fade-in starts from silence
    assert chapter.pcm[-1] == 0.0  # fade-out ends in silence
    assert math.isclose(float(chapter.pcm[1200]), 0.5)  # interior is untouched


def test_single_unit_produces_one_clip_and_no_gaps():
    chapter = assemble([(_tone(0.1), BreakAfter.PARAGRAPH)])

    assert len(chapter.clips) == 1
    assert math.isclose(chapter.clips[0].clip_begin_s, 0.0)
    assert math.isclose(chapter.clips[0].clip_end_s, 0.1)


def test_empty_input_yields_empty_chapter():
    chapter = assemble([])

    assert len(chapter.pcm) == 0
    assert chapter.clips == ()
    assert chapter.duration_s == 0.0


def test_unit_shorter_than_fade_is_attenuated_without_overlap():
    # Three samples is far below the fade window (360 each side); the fade must scale
    # down rather than overrun the array or amplify anything.
    chapter = assemble([(np.full(3, 0.5, dtype=np.float32), BreakAfter.SENTENCE)])

    assert len(chapter.pcm) == 3
    assert np.all(np.abs(chapter.pcm) <= 0.5)


def test_input_pcm_is_not_mutated():
    u = _tone(0.1)
    original = u.copy()

    assemble([(u, BreakAfter.SENTENCE)])

    assert np.array_equal(u, original)  # the fade must copy, not edit the cached array


class _RecordingRunner:
    """A fake ffmpeg runner capturing its argv + stdin and returning a fixed outcome."""

    def __init__(self, *, returncode: int = 0, stderr: str = "") -> None:
        self.commands: list[list[str]] = []
        self.stdins: list[bytes] = []
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, command: Sequence[str], stdin: bytes) -> CompletedCommand:
        self.commands.append(list(command))
        self.stdins.append(stdin)
        return CompletedCommand(
            returncode=self._returncode,
            stdout="",
            stderr=self._stderr,
        )


def test_encode_builds_command_and_streams_pcm(tmp_path: Path):
    pcm = _tone(0.05)
    runner = _RecordingRunner()
    out = tmp_path / "chapter.mp3"

    encode_chapter(
        pcm,
        out,
        bitrate=64_000,
        locate=lambda _: "/usr/bin/ffmpeg",
        runner=runner,
    )

    command = runner.commands[0]
    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-f") + 1] == "f32le"
    assert command[command.index("-ar") + 1] == str(SAMPLE_RATE)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-c:a") + 1] == "libmp3lame"
    assert command[command.index("-b:a") + 1] == "64000"
    assert command[-1] == str(out)

    # PCM reaches ffmpeg as raw little-endian float32 on stdin (no intermediate WAV).
    assert runner.stdins[0] == np.ascontiguousarray(pcm, dtype="<f4").tobytes()


def test_encode_raises_when_ffmpeg_missing(tmp_path: Path):
    with pytest.raises(FfmpegNotFoundError, match="ffmpeg"):
        encode_chapter(
            _tone(0.05),
            tmp_path / "c.mp3",
            bitrate=64_000,
            locate=lambda _: None,
        )


def test_encode_raises_on_nonzero_exit(tmp_path: Path):
    runner = _RecordingRunner(returncode=1, stderr="boom")

    with pytest.raises(FfmpegEncodeError, match="boom"):
        encode_chapter(
            _tone(0.05),
            tmp_path / "c.mp3",
            bitrate=64_000,
            locate=lambda _: "ffmpeg",
            runner=runner,
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_produces_a_playable_mp3(tmp_path: Path):
    out = tmp_path / "chapter.mp3"

    encode_chapter(_tone(0.2), out, bitrate=64_000)

    data = out.read_bytes()
    assert len(data) > 0
    # ffmpeg's mp3 muxer opens with an ID3 tag or a raw frame sync.
    assert data[:3] == b"ID3" or data[0] == 0xFF
