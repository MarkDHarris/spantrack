from collections.abc import Callable
from pathlib import Path

import pytest

from spantrack import epubcheck


def _recording_runner(
    returncode: int,
    stdout: str = "",
    stderr: str = "",
) -> Callable[..., epubcheck.CompletedCommand]:
    def run(command: list[str]):
        run.command = command  # pyright: ignore[reportFunctionMemberAccess]
        return epubcheck.CompletedCommand(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def test_validate_ok_on_exit_zero(tmp_path: Path):
    run = _recording_runner(0, stdout="No errors or warnings detected.\n")
    result = epubcheck.validate(
        tmp_path / "book.epub",
        locate=lambda _: "epubcheck",
        runner=run,
    )

    assert result.ok is True
    assert result.returncode == 0


def test_validate_reports_errors_on_nonzero(tmp_path: Path):
    stderr = "ERROR(RSC-005): content.opf(2,10): missing required attribute\n"
    run = _recording_runner(1, stderr=stderr)
    result = epubcheck.validate(
        tmp_path / "book.epub",
        locate=lambda _: "epubcheck",
        runner=run,
    )

    assert result.ok is False
    assert result.returncode == 1
    assert any("RSC-005" in message for message in result.messages)


def test_validate_invokes_epubcheck_with_resolved_path(tmp_path: Path):
    run = _recording_runner(0)
    epub = tmp_path / "book.epub"

    epubcheck.validate(epub, locate=lambda _: "/usr/bin/epubcheck", runner=run)

    assert run.command == ["/usr/bin/epubcheck", str(epub)]  # pyright: ignore[reportFunctionMemberAccess]


def test_validate_raises_when_binary_absent(tmp_path: Path):
    with pytest.raises(epubcheck.EpubcheckNotFoundError, match="epubcheck"):
        epubcheck.validate(tmp_path / "book.epub", executable="spantrack-no-such-tool")


def test_collects_severity_lines_from_both_streams_and_drops_noise(tmp_path: Path):
    run = _recording_runner(
        1,
        stdout="Validating using EPUB version 3.3 rules.\nWARNING(OPF-096): dated\n",
        stderr="ERROR(RSC-005): missing attr\nCheck finished with 2 messages\n",
    )
    result = epubcheck.validate(
        tmp_path / "book.epub",
        locate=lambda _: "epubcheck",
        runner=run,
    )

    # stdout severity lines precede stderr; the non-severity progress lines are dropped.
    assert result.messages == (
        "WARNING(OPF-096): dated",
        "ERROR(RSC-005): missing attr",
    )
