"""Human-readable progress reporting to a text stream."""

import sys
from collections.abc import Sequence
from typing import TextIO

from spantrack.narration_projection import SelectionProjection
from spantrack.narration_set import Selection


class ProgressReporter:
    """Write plain progress lines to a text stream (stderr by default)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def info(self, message: str) -> None:
        """Write one progress line."""
        print(message, file=self._stream)

    def selection(self, selection: Selection) -> None:
        """Print the narration set: chosen documents, skips with reasons, dud globs."""
        self.info(f"narration set: {len(selection.documents)} document(s)")
        for document in selection.documents:
            self.info(f"  narrate  {document.href}")

        for excluded in selection.excluded:
            self.info(
                f"  skip     {excluded.href or excluded.item_id} ({excluded.reason})"
            )

        for selector in selection.unmatched:
            self.info(f"  warning  override matched no document: {selector}")

    def projection(self, projection: SelectionProjection) -> None:
        """Print the projected character total, audio duration, and output size."""
        self.info(
            f"projected: {projection.total_characters:,} characters, "
            f"~{_format_duration(projection.duration_seconds)} audio, "
            f"~{_format_size(projection.size_bytes)}",
        )

    def oov(self, words: Sequence[str]) -> None:
        """Print the OOV review list as a curatable lexicon TOML skeleton."""
        if not words:
            self.info("no out-of-vocabulary words found")
            return

        self.info(f"out-of-vocabulary words: {len(words)}")
        self.info("[words]")
        for word in words:
            self.info(f'# {word} = ""')


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"

    if minutes:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


_BYTES_PER_UNIT = 1024


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    if size < _BYTES_PER_UNIT:
        return f"{size:.0f} B"

    for unit in ("KB", "MB"):
        size /= _BYTES_PER_UNIT
        if size < _BYTES_PER_UNIT:
            return f"{size:.1f} {unit}"

    return f"{size / _BYTES_PER_UNIT:.1f} GB"
