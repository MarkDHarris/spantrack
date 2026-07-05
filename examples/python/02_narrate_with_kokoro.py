"""Convert a book into a read-along EPUB with the real Kokoro voice.

This runs the whole spantrack pipeline end to end -- unpack, pick chapters, synthesize
each sentence with the local Kokoro-82M neural voice, assemble per-chapter audio, write
the SMIL Media Overlay, repack -- producing a genuine, epubcheck-clean read-along EPUB
that a reader like Thorium will play and highlight.

Needs ffmpeg on your PATH (spantrack shells out to it to encode the audio). The first
run downloads the Kokoro weights once; after that it is fully offline. The clockmaker
sample is tiny, so a run takes well under a minute on CPU.

Run it with `python examples/python/02_narrate_with_kokoro.py`.

What is a read-along EPUB? spantrack wraps each sentence in a `<span>` with a stable id,
records where that sentence's audio starts and ends, and writes a SMIL Media Overlay
tying the two together. As the audio plays, the reader highlights whichever sentence's
clip is currently sounding.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

from spantrack import cli, pipeline
from spantrack.progress import ProgressReporter

EXAMPLES = Path(__file__).resolve().parent.parent


def summarize(narrated_epub: Path) -> None:
    """Print the read-along artifacts spantrack added inside the output EPUB."""
    with zipfile.ZipFile(narrated_epub) as archive:
        names = archive.namelist()

    audio = sorted(name for name in names if name.endswith(".mp3"))
    overlays = sorted(name for name in names if name.endswith(".smil"))
    # spantrack adds one stylesheet of its own (the highlight rule); the book keeps its
    # existing style.css untouched, so filter to the one spantrack wrote.
    highlight = sorted(name for name in names if name.endswith("highlight.css"))

    print(f"\nwrote {narrated_epub.name} ({len(names)} files inside)")
    print(f"  chapter audio  : {audio}")
    print(f"  media overlays : {overlays}")
    print(f"  highlight CSS  : {highlight}")


def main() -> None:
    """Narrate the clockmaker sample with Kokoro and summarize the read-along output."""
    epub = EXAMPLES / "the-clockmakers-apprentice.epub"

    # Everything the run produces goes to a throwaway directory, so the example never
    # touches your checkout.
    out_dir = Path(tempfile.mkdtemp(prefix="spantrack-narrate-"))
    config = cli.parse_args(
        [
            str(epub),
            "--output",
            str(out_dir / "the-clockmakers-apprentice.narrated.epub"),
            "--work-dir",
            str(out_dir / "work"),
        ],
    )

    # With no backend_factory, pipeline.run builds the default Kokoro backend. The first
    # call loads the model (and downloads its weights the very first time you run it).
    exit_code = pipeline.run(config, reporter=ProgressReporter(sys.stdout))
    assert exit_code == 0, "conversion failed"

    summarize(config.output)


if __name__ == "__main__":
    main()
