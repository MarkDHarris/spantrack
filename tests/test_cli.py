from collections.abc import Sequence
from pathlib import Path

import pytest

from spantrack import cli
from spantrack.run_config import RunConfig


def test_defaults_are_resolved():
    config = cli.parse_args(["book.epub"])

    assert isinstance(config, RunConfig)
    assert config.input == Path("book.epub")
    assert config.output == Path("book.narrated.epub")
    assert config.work_dir == Path("book.epub.spantrack")
    assert config.voice == "af_heart"
    assert config.speed == 1.0
    assert config.lang == "en-us"
    assert config.lang_code == "a"
    assert config.granularity == "sentence"
    assert config.audio_format == "mp3"
    assert config.bitrate == "64k"
    assert config.lexicon is None
    assert config.include == ()
    assert config.exclude == ()
    assert config.pick == ()
    assert config.include_tables is False
    assert config.include_code is False
    assert config.device == "auto"
    assert config.dry_run is False
    assert config.clean is False
    assert config.validate is False


def test_output_and_work_dir_overrides():
    config = cli.parse_args(["dir/book.epub", "-o", "out.epub", "--work-dir", "wd"])

    assert config.output == Path("out.epub")
    assert config.work_dir == Path("wd")


def test_default_paths_respect_input_directory():
    config = cli.parse_args(["dir/book.epub"])

    assert config.output == Path("dir/book.narrated.epub")
    assert config.work_dir == Path("dir/book.epub.spantrack")


def test_selection_flags_collect_into_tuples():
    config = cli.parse_args(
        [
            "book.epub",
            "--include",
            "ch*.xhtml",
            "--include",
            "app*.xhtml",
            "--exclude",
            "cover.xhtml",
            "--pick",
            "a.xhtml,b.xhtml",
            "--pick",
            "c.xhtml",
        ]
    )

    assert config.include == ("ch*.xhtml", "app*.xhtml")
    assert config.exclude == ("cover.xhtml",)
    assert config.pick == ("a.xhtml", "b.xhtml", "c.xhtml")


def test_lang_en_gb_maps_to_code_b():
    assert cli.parse_args(["book.epub", "--lang", "en-gb"]).lang_code == "b"


def test_boolean_flags_set():
    config = cli.parse_args(
        [
            "book.epub",
            "--include-tables",
            "--include-code",
            "--dry-run",
            "--clean",
            "--validate",
        ]
    )

    assert config.include_tables
    assert config.include_code
    assert config.dry_run
    assert config.clean
    assert config.validate


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["book.epub", "--lang", "fr"],
        ["book.epub", "--granularity", "paragraph"],
        ["book.epub", "--audio-format", "flac"],
        ["book.epub", "--device", "tpu"],
        ["book.epub", "--speed", "0"],
        ["book.epub", "--speed", "notanumber"],
        ["book.epub", "--bitrate", "fast"],
    ],
)
def test_invalid_arguments_exit_code_2(argv: Sequence[str] | None):
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(argv)

    assert excinfo.value.code == 2


def test_word_granularity_emits_experimental_warning():
    with pytest.warns(UserWarning, match="experimental"):
        config = cli.parse_args(["book.epub", "--granularity", "word"])

    assert config.granularity == "word"
