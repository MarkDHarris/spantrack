"""The validated configuration for a single spantrack run."""

import enum
from dataclasses import dataclass
from pathlib import Path


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
    clean: bool
    validate: bool

    @property
    def lang_code(self) -> str:
        """Kokoro KPipeline language code for the selected language."""
        return _LANG_CODES[self.lang]
