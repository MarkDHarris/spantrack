"""Sequence the conversion stages and hand artifacts between them."""

import shutil
from collections.abc import Callable
from pathlib import Path

from spantrack import epub_container, epubcheck
from spantrack.progress import ProgressReporter
from spantrack.run_config import RunConfig

type Validator = Callable[[Path], epubcheck.EpubcheckResult]

BOOK_SUBDIR = "book"


def run(
    config: RunConfig,
    *,
    reporter: ProgressReporter | None = None,
    validator: Validator = epubcheck.validate,
) -> int:
    """Convert one EPUB end to end and return the process exit code.

    The pipeline unpacks the container and repacks it unchanged; narration stages are
    inserted between those steps.

    Args:
        config: The resolved run configuration.
        reporter: Progress sink; defaults to stderr reporting.
        validator: EPUB validator invoked only when `config.validate` is set.

    Returns:
        0 on success, nonzero if validation fails.
    """
    reporter = reporter or ProgressReporter()

    if config.dry_run:
        reporter.info(f"[dry-run] would narrate {config.input} -> {config.output}")
        return 0

    if config.clean and config.work_dir.exists():
        shutil.rmtree(config.work_dir)

    book_dir = config.work_dir / BOOK_SUBDIR
    if book_dir.exists():
        shutil.rmtree(book_dir)

    names = epub_container.unzip(config.input, book_dir)

    epub_container.repack(book_dir, config.output, order=names)
    reporter.info(f"wrote {config.output}")

    if config.validate:
        return _report_validation(validator(config.output), reporter)

    return 0


def _report_validation(
    result: epubcheck.EpubcheckResult,
    reporter: ProgressReporter,
) -> int:
    if result.ok:
        reporter.info("epubcheck: no errors")
        return 0

    for message in result.messages:
        reporter.info(message)

    reporter.info(f"epubcheck failed with {len(result.messages)} message(s)")
    return 1
