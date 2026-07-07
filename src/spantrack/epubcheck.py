"""Opt-in EPUB validation gate wrapping the external `epubcheck` tool."""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CompletedCommand",
    "EpubcheckNotFoundError",
    "EpubcheckResult",
    "validate",
]

DEFAULT_EXECUTABLE = "epubcheck"
_MESSAGE_PREFIXES = ("FATAL", "ERROR", "WARNING")


class EpubcheckNotFoundError(RuntimeError):
    """Raised when the `epubcheck` executable is not on PATH."""


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    """Outcome of running an external command."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class EpubcheckResult:
    """Result of validating an EPUB with epubcheck."""

    ok: bool
    returncode: int
    messages: tuple[str, ...]


type CommandRunner = Callable[[Sequence[str]], CompletedCommand]
type Locator = Callable[[str], str | None]


def validate(
    epub_path: Path,
    *,
    executable: str = DEFAULT_EXECUTABLE,
    locate: Locator = shutil.which,
    runner: CommandRunner | None = None,
) -> EpubcheckResult:
    """Validate an EPUB by running epubcheck, located on the system PATH.

    Args:
        epub_path: The .epub file to validate.
        executable: Name (or path) of the epubcheck command to resolve.
        locate: Resolves an executable name to a full path, or None if absent.
        runner: Runs the resolved command; defaults to a subprocess call.

    Returns:
        The validation outcome. `ok` follows epubcheck's exit code (warnings alone do
        not fail); `messages` collects any FATAL/ERROR/WARNING lines.

    Raises:
        EpubcheckNotFoundError: If the executable cannot be resolved on PATH.
    """
    resolved = locate(executable)
    if resolved is None:
        err_msg = (
            f"{executable!r} not found on PATH; install epubcheck or drop --validate"
        )
        raise EpubcheckNotFoundError(err_msg)

    run = runner if runner is not None else _run_subprocess
    completed = run([resolved, str(epub_path)])

    return EpubcheckResult(
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        messages=_extract_messages(completed),
    )


def _run_subprocess(command: Sequence[str]) -> CompletedCommand:
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return CompletedCommand(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _extract_messages(completed: CompletedCommand) -> tuple[str, ...]:
    # epubcheck splits diagnostics across stdout and stderr depending on version, so
    # scan the two together; a line counts only when it opens with a severity token.
    lines = f"{completed.stdout}\n{completed.stderr}".splitlines()
    stripped = (line.strip() for line in lines)

    return tuple(line for line in stripped if line.startswith(_MESSAGE_PREFIXES))
