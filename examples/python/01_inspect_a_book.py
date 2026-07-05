"""Inspect what spantrack would narrate, without synthesizing any audio.

A dry run is the fast, free way to see two things before committing to a full (and slow)
conversion:

- the *narration set*: which content documents spantrack chose to narrate, and why it
  skipped the rest (a cover, a table of contents, front matter, a non-linear page);
- a *projection*: a rough estimate of the resulting audio duration and file size.

Run it with `python examples/python/01_inspect_a_book.py`.

Nothing is written to your checkout: each book is unpacked into a throwaway temporary
directory, and a dry run never produces an output EPUB at all.
"""

import sys
import tempfile
from pathlib import Path

from spantrack import cli, pipeline
from spantrack.progress import ProgressReporter

# This file lives in examples/python/, so examples/ is two directories up.
EXAMPLES = Path(__file__).resolve().parent.parent


def inspect(epub: Path) -> None:
    """Print the narration set and projection for one EPUB."""
    # Every run unpacks the source EPUB into a work directory. A dry run keeps nothing,
    # so point it at a fresh temp dir we can forget about.
    work_dir = Path(tempfile.mkdtemp(prefix="spantrack-inspect-"))

    # `cli.parse_args` turns the flags you would type on the command line into a
    # validated RunConfig -- the single object every pipeline stage reads from.
    config = cli.parse_args([str(epub), "--dry-run", "--work-dir", str(work_dir)])

    # Progress normally goes to stderr; send it to stdout so it is easy to read here.
    print(f"# {epub.name}")
    pipeline.run(config, reporter=ProgressReporter(sys.stdout))
    print()


def main() -> None:
    """Inspect each of the sample books in turn."""
    for name in (
        "the-clockmakers-apprentice",
        "field-notes-on-comets",
        "the-pocket-anthology",
    ):
        inspect(EXAMPLES / f"{name}.epub")


if __name__ == "__main__":
    main()
