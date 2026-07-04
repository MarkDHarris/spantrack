"""Console entry point and CLI flag surface for spantrack."""

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

from spantrack import pipeline
from spantrack.run_config import (
    AudioFormat,
    Device,
    Granularity,
    Language,
    RunConfig,
    parse_bitrate,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `spantrack` command."""
    parser = argparse.ArgumentParser(
        prog="spantrack",
        description="Convert a text EPUB into an EPUB 3 with Media Overlays.",
    )

    parser.add_argument("input", type=Path, help="source EPUB file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output EPUB path (default: <stem>.narrated.epub)",
    )
    parser.add_argument("--voice", default="af_heart", help="TTS voice")
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="Speech speed multiplier",
    )
    parser.add_argument(
        "--lang",
        choices=tuple(Language),
        type=Language,
        default=Language.EN_US,
        help="Narration language",
    )
    parser.add_argument(
        "--granularity",
        choices=tuple(Granularity),
        type=Granularity,
        default=Granularity.SENTENCE,
        help="Highlight granularity (word is experimental)",
    )
    parser.add_argument(
        "--audio-format",
        choices=tuple(AudioFormat),
        type=AudioFormat,
        default=AudioFormat.MP3,
        dest="audio_format",
        help="Chapter audio codec",
    )
    parser.add_argument(
        "--bitrate",
        default="64k",
        type=_valid_bitrate,
        help="Audio bitrate",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="Pronunciation lexicon",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob of content documents to add to the narration set (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob of content documents to drop from the narration set (repeatable)",
    )
    parser.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="DOC[,DOC...]",
        help="Explicit content documents to narrate (repeatable, comma-separated)",
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        dest="include_tables",
        help="Narrate table content (skipped by default)",
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        dest="include_code",
        help="Narrate pre/code content (skipped by default)",
    )
    parser.add_argument(
        "--device",
        choices=tuple(Device),
        type=Device,
        default=Device.AUTO,
        help="Synthesis device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the narration set and projections, then exit",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Discard any existing work directory",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        dest="work_dir",
        help="Override the work directory (default: <input>.spantrack)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run epubcheck on the output",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse command-line arguments into a validated RunConfig.

    Args:
        argv: Arguments to parse, defaulting to `sys.argv[1:]`.

    Returns:
        The resolved run configuration.

    Raises:
        SystemExit: On invalid arguments (argparse exits with code 2).
    """
    args = build_parser().parse_args(argv)
    if args.granularity is Granularity.WORD:
        warnings.warn(
            "word granularity is experimental; A/B a single chapter in Thorium "
            "before committing to a full-book run",
            stacklevel=2,
        )

    return _to_config(args)


def _to_config(args: argparse.Namespace) -> RunConfig:
    input_path: Path = args.input
    output = args.output or input_path.with_name(f"{input_path.stem}.narrated.epub")
    work_dir = args.work_dir or input_path.with_name(f"{input_path.name}.spantrack")

    return RunConfig(
        input=input_path,
        output=output,
        work_dir=work_dir,
        voice=args.voice,
        speed=args.speed,
        lang=args.lang,
        granularity=args.granularity,
        audio_format=args.audio_format,
        bitrate=args.bitrate,
        lexicon=args.lexicon,
        include=tuple(args.include),
        exclude=tuple(args.exclude),
        pick=tuple(part for value in args.pick for part in value.split(",") if part),
        include_tables=args.include_tables,
        include_code=args.include_code,
        device=args.device,
        dry_run=args.dry_run,
        clean=args.clean,
        validate=args.validate,
    )


def _valid_bitrate(raw: str) -> str:
    # Validate at parse time so a bad rate exits with argparse's code 2.
    try:
        parse_bitrate(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    return raw


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        err_msg = "must be greater than 0"
        raise argparse.ArgumentTypeError(err_msg)

    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the spantrack CLI.

    Args:
        argv: Command-line arguments, defaulting to `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    config = parse_args(argv)

    return pipeline.run(config)
