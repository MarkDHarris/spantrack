"""Chapter audio assembly: concatenate sentence PCM with gaps, fades, and offsets.

`assemble` turns an ordered run of per-sentence PCM into one chapter waveform plus the
clip boundaries the SMIL stage emits verbatim. Two rules shape it:

1. Gaps fold into the preceding clip. A `break_after` inserts silence between two
   sentences (250ms sentence / 600ms paragraph / 900ms heading), and each clip runs up
   to where the next sentence's audio begins (`clip_end_n == clip_begin_{n+1}`). Gapless
   intervals keep the highlight from flickering in the reader and keep epubcheck from
   flagging overlay gaps -- the pause belongs to the sentence it follows.
2. Edges fade. Silence-trimmed sentences start and end mid-waveform, so a raw
   concatenation clicks at every join. A short linear fade on each unit's edges removes
   the discontinuity without shifting any timestamp.

The assembled PCM is deterministic given the same sentence PCM (gaps and fades are
fixed); the encoded chapter file `encode_chapter` produces (MP3 or Ogg Opus) is not
guaranteed byte-identical across ffmpeg versions -- the sentence cache, not the chapter
file, is the determinism anchor.
"""

import enum
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spantrack.run_config import AudioFormat
from spantrack.tts.backend import SAMPLE_RATE, Pcm

__all__ = [
    "AssembledChapter",
    "BreakAfter",
    "ClipInterval",
    "CompletedCommand",
    "EncodeSettings",
    "FfmpegEncodeError",
    "FfmpegNotFoundError",
    "assemble",
    "encode_chapter",
]


class BreakAfter(enum.StrEnum):
    """The break after a sentence; it sets the silence gap before the next one."""

    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HEADING = "heading"


# Inter-sentence pause per break type, folded into the preceding clip.
_GAP_MS: dict[BreakAfter, int] = {
    BreakAfter.SENTENCE: 250,
    BreakAfter.PARAGRAPH: 600,
    BreakAfter.HEADING: 900,
}

# Linear fade applied to each unit's leading and trailing edge to suppress join clicks.
_FADE_MS = 15
_FADE_SAMPLES = round(_FADE_MS / 1000 * SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class ClipInterval:
    """One sentence's gapless span in the chapter timeline, in seconds."""

    clip_begin_s: float
    clip_end_s: float


@dataclass(frozen=True, slots=True)
class AssembledChapter:
    """A concatenated chapter waveform with its per-sentence clip boundaries."""

    pcm: Pcm
    clips: tuple[ClipInterval, ...]
    duration_s: float


def assemble(units: Sequence[tuple[Pcm, BreakAfter]]) -> AssembledChapter:
    """Concatenate per-sentence PCM into a chapter with gaps, fades, and clip offsets.

    Args:
        units: Ordered `(pcm, break_after)` pairs, one per sentence; `break_after` sets
            the silence gap inserted before the following sentence (the last unit's
            break adds no trailing silence).

    Returns:
        The concatenated 24 kHz float32 PCM, one gapless `ClipInterval` per unit
        (`clip_end_n == clip_begin_{n+1}`, the gap folded into the preceding clip), and
        the total duration in seconds.
    """
    segments: list[Pcm] = []
    clips: list[ClipInterval] = []
    cursor = 0  # running sample offset into the assembled waveform
    last = len(units) - 1
    for index, (pcm, break_after) in enumerate(units):
        begin = cursor
        segments.append(_edge_fade(pcm))
        cursor += len(pcm)

        # The gap belongs to this clip (gapless), so extend the clip past it; the final
        # unit gets none, so the chapter never ends on trailing silence.
        if index != last:
            gap = round(_GAP_MS[break_after] / 1000 * SAMPLE_RATE)
            segments.append(np.zeros(gap, dtype=np.float32))
            cursor += gap

        clips.append(ClipInterval(begin / SAMPLE_RATE, cursor / SAMPLE_RATE))

    pcm = (
        np.concatenate(segments).astype(np.float32, copy=False)  # pyright: ignore[reportUnknownMemberType]
        if segments
        else np.zeros(0, dtype=np.float32)
    )
    return AssembledChapter(
        pcm=pcm, clips=tuple(clips), duration_s=cursor / SAMPLE_RATE
    )


def _edge_fade(pcm: Pcm) -> Pcm:
    """Fade `pcm`'s leading and trailing edge on a copy; too-short units pass through.

    The fade shrinks to half the unit when it is shorter than two fade windows, so the
    ramps never overlap. A unit too short to fade (`fade == 0`, i.e. 0-1 samples) is
    returned unchanged; otherwise the fade is applied to a copy, since the cached
    sentence audio must never be mutated in place.
    """
    fade = min(_FADE_SAMPLES, len(pcm) // 2)
    if fade == 0:
        return pcm

    faded = pcm.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]

    return faded


DEFAULT_EXECUTABLE = "ffmpeg"

type Locator = Callable[[str], str | None]
type FfmpegRunner = Callable[[Sequence[str], bytes], "CompletedCommand"]


class FfmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg executable is not on PATH."""


class FfmpegEncodeError(RuntimeError):
    """Raised when ffmpeg exits non-zero while encoding a chapter."""


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    """Outcome of running the ffmpeg encode subprocess."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class EncodeSettings:
    """How to encode a chapter: the target audio format and its bitrate."""

    audio_format: AudioFormat
    bitrate: int


# ffmpeg audio codec per format; the container (MP3 / Ogg Opus) follows the output file
# extension the pipeline derives from `AudioFormat`.
_CODECS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "libmp3lame",
    AudioFormat.OPUS: "libopus",
}


def encode_chapter(
    pcm: Pcm,
    out_path: Path,
    *,
    settings: EncodeSettings,
    locate: Locator = shutil.which,
    runner: FfmpegRunner | None = None,
) -> None:
    """Encode assembled chapter PCM to a mono audio file at `out_path` via ffmpeg.

    The float32 PCM is streamed to ffmpeg as raw little-endian `f32le` on stdin, so no
    intermediate WAV is written. `settings.audio_format` selects the codec (MP3 via
    `libmp3lame`, Opus via `libopus`); the container follows `out_path`'s extension.
    `locate`/`runner` are injected to keep the stage testable without the binary,
    mirroring the epubcheck gate.

    Args:
        pcm: The assembled 24 kHz mono float32 chapter waveform.
        out_path: Destination audio path; its extension sets the container (`.mp3` /
            `.opus`) and is overwritten if it already exists.
        settings: The target audio format and bitrate (bits per second).
        locate: Resolves the ffmpeg executable to a full path, or None if absent.
        runner: Runs the resolved command with stdin bytes; defaults to a subprocess.

    Raises:
        FfmpegNotFoundError: If the executable cannot be resolved on PATH.
        FfmpegEncodeError: If ffmpeg exits with a non-zero status.
    """
    resolved = locate(DEFAULT_EXECUTABLE)
    if resolved is None:
        msg = (
            f"{DEFAULT_EXECUTABLE!r} not found on PATH; install ffmpeg to encode audio"
        )
        raise FfmpegNotFoundError(msg)

    # ffmpeg wants input options before `-i` and output options after; raw f32le/24 kHz
    # mono matches the `Pcm` bytes piped on stdin (`-i -`).
    command = [
        resolved,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "-",
        "-c:a",
        _CODECS[settings.audio_format],
        "-b:a",
        str(settings.bitrate),
        str(out_path),
    ]
    run = runner if runner is not None else _run_ffmpeg
    completed = run(command, np.ascontiguousarray(pcm, dtype="<f4").tobytes())
    if completed.returncode != 0:
        msg = f"ffmpeg exited {completed.returncode}: {completed.stderr.strip()}"
        raise FfmpegEncodeError(msg)


def _run_ffmpeg(command: Sequence[str], stdin: bytes) -> CompletedCommand:
    completed = subprocess.run(  # noqa: S603
        command,
        input=stdin,
        capture_output=True,
        check=False,
    )
    return CompletedCommand(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", "replace"),
        stderr=completed.stderr.decode("utf-8", "replace"),
    )
