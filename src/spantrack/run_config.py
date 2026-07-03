"""The validated configuration for a single spantrack run."""

from dataclasses import dataclass
from pathlib import Path

LANG_CODES = {"en-us": "a", "en-gb": "b"}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A fully resolved set of options for one conversion run."""

    input: Path
    output: Path
    work_dir: Path
    voice: str
    speed: float
    lang: str
    granularity: str
    audio_format: str
    bitrate: str
    lexicon: Path | None
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    pick: tuple[str, ...]
    include_tables: bool
    include_code: bool
    device: str
    dry_run: bool
    clean: bool
    validate: bool

    @property
    def lang_code(self) -> str:
        """Kokoro KPipeline language code for the selected language."""
        return LANG_CODES[self.lang]
