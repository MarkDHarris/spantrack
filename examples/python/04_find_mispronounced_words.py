"""List the words spantrack's voice might mispronounce, as a lexicon starter.

Before a long conversion it is worth checking which words fall outside the TTS engine's
pronunciation dictionary: names, coined terms, unusual units. `--oov-report`
runs the same normalization the real narration would, feeds it through the
grapheme-to-phoneme model, and prints every out-of-vocabulary ("OOV") word as a
commented-out TOML lexicon entry you can uncomment and fill in.

You would save those lines to a file and pass it back with `--lexicon words.toml` on a
real run, so the voice pronounces them your way.

Run it with `python examples/python/04_find_mispronounced_words.py`. It loads the G2P
model, so the first run takes a few seconds; no audio is synthesized.
"""

import sys
import tempfile
from pathlib import Path

from spantrack import cli, pipeline
from spantrack.progress import ProgressReporter

EXAMPLES = Path(__file__).resolve().parent.parent


def main() -> None:
    """Print the out-of-vocabulary review list for the comet field-notes book."""
    epub = EXAMPLES / "field-notes-on-comets.epub"
    work_dir = Path(tempfile.mkdtemp(prefix="spantrack-oov-"))
    config = cli.parse_args([str(epub), "--oov-report", "--work-dir", str(work_dir)])
    pipeline.run(config, reporter=ProgressReporter(sys.stdout))


if __name__ == "__main__":
    main()
