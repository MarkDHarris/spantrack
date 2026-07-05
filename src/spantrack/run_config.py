"""The validated configuration for a single spantrack run."""

import enum
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AudioFormat",
    "Device",
    "Granularity",
    "Language",
    "RunConfig",
    "parse_bitrate",
]


class Language(enum.StrEnum):
    """Narration language; mapped to a Kokoro pipeline code by `RunConfig.lang_code`."""

    EN_US = "en-us"
    EN_GB = "en-gb"


class Granularity(enum.StrEnum):
    """Highlight granularity for the media overlay."""

    SENTENCE = "sentence"
    WORD = "word"


class AudioFormat(enum.StrEnum):
    """Chapter audio container/codec."""

    MP3 = "mp3"
    OPUS = "opus"


class Device(enum.StrEnum):
    """Synthesis device selection passed to Kokoro."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


_LANG_CODES: dict[Language, str] = {Language.EN_US: "a", Language.EN_GB: "b"}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A fully resolved set of options for one conversion run."""

    input: Path
    output: Path
    work_dir: Path
    voice: str
    speed: float
    lang: Language
    granularity: Granularity
    audio_format: AudioFormat
    bitrate: str
    lexicon: Path | None
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    pick: tuple[str, ...]
    include_tables: bool
    include_code: bool
    device: Device
    dry_run: bool
    oov_report: bool
    clean: bool
    validate: bool

    @property
    def lang_code(self) -> str:
        """Kokoro KPipeline language code for the selected language."""
        return _LANG_CODES[self.lang]


# Audio bitrate as accepted on the command line: a number with an optional k/M suffix.
_BITRATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([km]?)\s*$", re.IGNORECASE)
_BITRATE_SUFFIXES = {"": 1, "k": 1_000, "m": 1_000_000}


def parse_bitrate(bitrate: str) -> int:
    """Parse an encoder bitrate like `64k`, `128000`, or `1.5M` into bits per second.

    Lives here so the CLI can reject a bad rate at parse time and the projection can
    reuse the same grammar rather than each owning a copy.

    Raises:
        ValueError: When `bitrate` is not a recognizable rate.
    """
    match = _BITRATE.match(bitrate)
    if match is None:
        msg = f"invalid bitrate: {bitrate!r}"
        raise ValueError(msg)

    value, suffix = match.groups()

    return int(float(value) * _BITRATE_SUFFIXES[suffix.lower()])
