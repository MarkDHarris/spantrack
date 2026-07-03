"""Human-readable progress reporting to a text stream."""

import sys
from typing import TextIO


class ProgressReporter:
    """Write plain progress lines to a text stream (stderr by default)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def info(self, message: str) -> None:
        """Write one progress line."""
        print(message, file=self._stream)
