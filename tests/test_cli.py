from pathlib import Path

import pytest

from spantrack import cli, progress
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
    assert config.device == "auto"
    assert config.dry_run is False
    assert config.oov_report is False
    assert config.clean is False
    assert config.validate is False
    assert config.allow_untimed_fallback is False
    assert config.progress is False


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
            "--dry-run",
            "--clean",
            "--validate",
            "--allow-untimed-fallback",
        ]
    )

    assert config.dry_run
    assert config.clean
    assert config.validate
    assert config.allow_untimed_fallback


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
def test_invalid_arguments_exit_code_2(argv: list[str]):
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(argv)

    assert excinfo.value.code == 2


def test_oov_report_flag_sets_config():
    assert cli.parse_args(["book.epub", "--oov-report"]).oov_report is True


def test_word_granularity_emits_experimental_warning():
    with pytest.warns(UserWarning, match="experimental"):
        config = cli.parse_args(["book.epub", "--granularity", "word"])

    assert config.granularity == "word"


def test_allow_untimed_fallback_pairs_with_word_granularity():
    with pytest.warns(UserWarning, match="experimental"):
        config = cli.parse_args(
            ["book.epub", "--granularity", "word", "--allow-untimed-fallback"],
        )

    assert config.allow_untimed_fallback is True


def test_progress_flag_sets_config():
    assert cli.parse_args(["book.epub", "--progress"]).progress is True


def test_progress_without_tqdm_exits_with_a_hint(monkeypatch: pytest.MonkeyPatch):
    def boom() -> None:
        raise progress.ProgressBarUnavailableError("tqdm not installed")

    # --progress fails fast at parse time when the optional dependency is missing.
    # (cli imports this same module object, so patching it here patches cli's view.)
    monkeypatch.setattr(progress, "ensure_tqdm_available", boom)
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args(["book.epub", "--progress"])

    assert excinfo.value.code == 2
