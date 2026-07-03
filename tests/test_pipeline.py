from pathlib import Path

import pytest

from spantrack import cli, epub_container, narration_projection, narration_set, pipeline
from spantrack.epubcheck import EpubcheckResult
from spantrack.progress import ProgressReporter


class _RecordingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.selections: list[narration_set.Selection] = []
        self.projections: list[narration_projection.SelectionProjection] = []

    def info(self, message: str) -> None:
        pass

    def selection(self, selection: narration_set.Selection) -> None:
        self.selections.append(selection)

    def projection(self, projection: narration_projection.SelectionProjection) -> None:
        self.projections.append(projection)


def _read_tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _config(epub: Path, tmp_path: Path, *extra: str):
    return cli.parse_args(
        [
            str(epub),
            "-o",
            str(tmp_path / "out.epub"),
            "--work-dir",
            str(tmp_path / "work"),
            *extra,
        ],
    )


def test_run_produces_round_trippable_output(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path)
    code = pipeline.run(config)

    assert code == 0
    assert config.output.exists()

    source = tmp_path / "source"
    result = tmp_path / "result"

    epub_container.unzip(minimal_epub_path, source)
    epub_container.unzip(config.output, result)

    assert _read_tree(result) == _read_tree(source)


def test_dry_run_writes_no_output(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--dry-run")

    assert pipeline.run(config) == 0
    assert not config.output.exists()


def test_dry_run_reports_selection_and_projection(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    reporter = _RecordingReporter()
    config = _config(minimal_epub2_path, tmp_path, "--dry-run")

    assert pipeline.run(config, reporter=reporter) == 0
    assert not config.output.exists()

    selection = reporter.selections[0]
    assert [doc.href for doc in selection.documents] == ["OEBPS/chapter1.xhtml"]
    assert reporter.projections[0].total_characters > 0


def test_pick_override_limits_narration_set(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    reporter = _RecordingReporter()
    config = _config(
        minimal_epub2_path,
        tmp_path,
        "--dry-run",
        "--pick",
        "cover.xhtml",
    )

    pipeline.run(config, reporter=reporter)

    assert [doc.href for doc in reporter.selections[0].documents] == [
        "OEBPS/cover.xhtml"
    ]


def test_epub2_output_is_upgraded_to_epub3(minimal_epub2_path: Path, tmp_path: Path):
    config = _config(minimal_epub2_path, tmp_path)
    assert pipeline.run(config) == 0

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)
    assert 'version="3.0"' in (result / "OEBPS" / "content.opf").read_text()
    assert (result / "OEBPS" / "nav.xhtml").is_file()


def test_dry_run_prints_selection_to_stream(
    minimal_epub2_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _config(minimal_epub2_path, tmp_path, "--dry-run")
    pipeline.run(config)

    err = capsys.readouterr().err
    assert "OEBPS/chapter1.xhtml" in err
    assert "projected" in err


def test_clean_rebuilds_work_dir(minimal_epub_path: Path, tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()

    stale = work / "stale.txt"
    stale.write_text("old")

    config = _config(minimal_epub_path, tmp_path, "--clean")

    assert pipeline.run(config) == 0
    assert not stale.exists()
    assert config.output.exists()


def test_validate_pass_returns_zero(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--validate")
    code = pipeline.run(
        config,
        validator=lambda _: EpubcheckResult(ok=True, returncode=0, messages=[]),
    )

    assert code == 0


def test_validate_failure_returns_nonzero(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--validate")
    code = pipeline.run(
        config,
        validator=lambda _: EpubcheckResult(
            ok=False,
            returncode=1,
            messages=["ERROR(RSC-005): broken"],
        ),
    )

    assert code == 1


def test_validate_invoked_with_output_path(minimal_epub_path: Path, tmp_path: Path):
    seen: dict[str, Path] = {}

    def validator(path: Path) -> EpubcheckResult:
        seen["path"] = path
        return EpubcheckResult(ok=True, returncode=0, messages=[])

    config = _config(minimal_epub_path, tmp_path, "--validate")
    pipeline.run(config, validator=validator)

    assert seen["path"] == config.output


def test_no_validation_by_default(minimal_epub_path: Path, tmp_path: Path):
    def validator(path: Path) -> EpubcheckResult:  # noqa: ARG001
        raise AssertionError("validator must not run without --validate")

    config = _config(minimal_epub_path, tmp_path)

    assert pipeline.run(config, validator=validator) == 0


def test_main_runs_end_to_end(minimal_epub_path: Path, tmp_path: Path):
    out = tmp_path / "out.epub"
    code = cli.main(
        [
            str(minimal_epub_path),
            "-o",
            str(out),
            "--work-dir",
            str(tmp_path / "w"),
        ],
    )

    assert code == 0
    assert out.exists()
