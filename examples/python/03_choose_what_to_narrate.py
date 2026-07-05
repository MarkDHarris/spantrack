"""Steer the narration set with --include, --exclude, and --pick.

By default spantrack narrates the body of a book and skips covers, tables of contents,
front matter, and non-linear pages. The three override flags adjust that set without
editing the book:

- `--exclude GLOB` drops matching documents from the default set;
- `--include GLOB` adds back a document the default rules skipped as *soft* (front
  matter or a non-linear page). It cannot resurrect a cover or the nav document, which
  never carry prose;
- `--pick GLOB` narrates exactly the matching documents and nothing else.

Globs match a document's container-relative path or its bare file name.

Run it with `python examples/python/03_choose_what_to_narrate.py`. Every variation is a
dry run, so nothing is synthesized or written.
"""

import sys
import tempfile
from pathlib import Path

from spantrack import cli, pipeline
from spantrack.progress import ProgressReporter

EXAMPLES = Path(__file__).resolve().parent.parent
ANTHOLOGY = EXAMPLES / "the-pocket-anthology.epub"


def show(title: str, *flags: str) -> None:
    """Dry-run the anthology with the given override flags and print the result."""
    work_dir = Path(tempfile.mkdtemp(prefix="spantrack-select-"))
    config = cli.parse_args(
        [str(ANTHOLOGY), "--dry-run", "--work-dir", str(work_dir), *flags],
    )
    print(f"## {title}")
    print(f"   flags: {' '.join(flags) or '(none)'}")
    pipeline.run(config, reporter=ProgressReporter(sys.stdout))
    print()


def main() -> None:
    """Compare the default narration set against a few override combinations."""
    show("Default: three stories; front matter and colophon skipped")
    show("Add the foreword back", "--include", "foreword.xhtml")
    show("Add the non-linear colophon back", "--include", "colophon.xhtml")
    show("Drop one story", "--exclude", "story-tide.xhtml")
    show("Narrate a single story only", "--pick", "story-lamp.xhtml")


if __name__ == "__main__":
    main()
