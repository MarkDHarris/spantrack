"""Human-readable progress reporting to a text stream.

Per-chapter narration progress renders as an optional tqdm bar (`progress_bar=True`) or,
by default, as one plain line per chapter. tqdm is an optional dependency, imported only
when a bar is actually requested, so the base install never pulls it in.
"""

import importlib.util
import sys
from collections.abc import Callable, Sequence
from typing import Protocol, Self, TextIO, cast

from spantrack.narration_projection import SelectionProjection
from spantrack.narration_set import Selection


class ProgressBarUnavailableError(RuntimeError):
    """Raised when a progress bar is requested but the optional tqdm dep is absent."""


_TQDM_HINT = (
    "the --progress bar needs tqdm; install it with "
    "`pip install 'spantrack[progress]'` (or `pip install tqdm`)"
)


def ensure_tqdm_available() -> None:
    """Raise `ProgressBarUnavailableError` if the optional tqdm dependency is absent."""
    if importlib.util.find_spec("tqdm") is None:
        raise ProgressBarUnavailableError(_TQDM_HINT)


class _ProgressBar(Protocol):
    """The slice of tqdm's API the chapter tracker drives (tqdm ships no type stubs)."""

    def update(self, n: float = ...) -> object: ...

    def close(self) -> None: ...


class ChapterProgress:
    """Per-chapter narration progress: a tqdm bar when enabled, else one line each."""

    def __init__(self, stream: TextIO, total: int, *, use_bar: bool) -> None:
        self._stream = stream
        self._bar = _build_bar(stream, total) if use_bar else None

    def narrated(self, href: str) -> None:
        """Record one narrated chapter (advance the bar, or print a plain line)."""
        if self._bar is None:
            print(f"narrated {href}", file=self._stream)
        else:
            self._bar.update(1)

    def close(self) -> None:
        """Finish the progress bar, if one is active."""
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> Self:
        """Enter the progress context, returning this tracker."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the bar on exit, so an error mid-narration still tidies up."""
        self.close()


def _build_bar(stream: TextIO, total: int) -> _ProgressBar:
    # Import lazily via importlib so the optional dep is neither a hard import nor a
    # static `from tqdm import ...` (tqdm has no stubs; the cast quarantines its types).
    try:
        tqdm_module = importlib.import_module("tqdm")
    except ImportError as exc:
        raise ProgressBarUnavailableError(_TQDM_HINT) from exc

    factory = cast("Callable[..., _ProgressBar]", tqdm_module.tqdm)
    return factory(total=total, desc="narrating", unit="chapter", file=stream)


class ProgressReporter:
    """Write plain progress lines to a text stream (stderr by default)."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        progress_bar: bool = False,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._progress_bar = progress_bar

    def info(self, message: str) -> None:
        """Write one progress line."""
        print(message, file=self._stream)

    def narrating(self, total: int) -> ChapterProgress:
        """Open per-chapter progress over `total` chapters.

        Returns a tqdm bar when this reporter was built with `progress_bar=True`,
        otherwise a tracker that prints one `narrated <href>` line per chapter.
        """
        return ChapterProgress(self._stream, total, use_bar=self._progress_bar)

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
