"""Produce a word-level read-along, where each word highlights as it is spoken.

spantrack's default granularity is one sentence at a time. In experimental *word* mode
it also wraps every word in its own `<span>` and points the Media Overlay at those, so
the highlight advances word by word. The per-word timings come from the Kokoro voice's
own per-token timestamps.

Same real-voice conversion as example 02, so it needs ffmpeg and (on the first run) the
one-time Kokoro weight download.

Run it with `python examples/python/05_word_by_word_highlighting.py`.

Word mode is experimental: Kokoro's per-token timings drift, and the CLI prints a
warning. Prefer sentence mode (the default) for anything you intend to ship.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

from spantrack import cli, pipeline
from spantrack.progress import ProgressReporter

EXAMPLES = Path(__file__).resolve().parent.parent


def report_word_spans(narrated_epub: Path) -> None:
    """Confirm the output wraps words and that the overlay points at word spans."""
    with zipfile.ZipFile(narrated_epub) as archive:
        chapters = [n for n in archive.namelist() if n.endswith(".xhtml")]
        overlays = [n for n in archive.namelist() if n.endswith(".smil")]

        # A word span id looks like s00001-w0 (sentence id + "-w" + word index). If the
        # first word span is present, the document was wrapped word by word.
        wrapped = sum('-w0"' in archive.read(name).decode("utf-8") for name in chapters)
        overlay = archive.read(overlays[0]).decode("utf-8") if overlays else ""

    print(f"\ncontent documents wrapped word-by-word: {wrapped} of {len(chapters)}")
    print(f"first overlay targets word spans: {'-w0' in overlay}")


def main() -> None:
    """Convert the clockmaker sample with Kokoro in experimental word mode."""
    epub = EXAMPLES / "the-clockmakers-apprentice.epub"
    out_dir = Path(tempfile.mkdtemp(prefix="spantrack-word-"))
    config = cli.parse_args(
        [
            str(epub),
            "--granularity",
            "word",
            "--output",
            str(out_dir / "the-clockmakers-apprentice.word.narrated.epub"),
            "--work-dir",
            str(out_dir / "work"),
        ],
    )
    exit_code = pipeline.run(config, reporter=ProgressReporter(sys.stdout))
    assert exit_code == 0, "conversion failed"

    report_word_spans(config.output)


if __name__ == "__main__":
    main()
