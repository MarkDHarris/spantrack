"""Sequence the conversion stages and hand artifacts between them."""

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from spantrack import (
    content_namespace,
    epub2_upgrade,
    epub_container,
    epubcheck,
    narration_projection,
    narration_set,
)
from spantrack.progress import ProgressReporter
from spantrack.run_config import RunConfig

__all__ = ["run"]

type Validator = Callable[[Path], epubcheck.EpubcheckResult]

BOOK_SUBDIR = "book"


def run(
    config: RunConfig,
    *,
    reporter: ProgressReporter | None = None,
    validator: Validator = epubcheck.validate,
) -> int:
    """Convert one EPUB end to end and return the process exit code.

    Unpacks the container, upgrades EPUB 2 inputs to EPUB 3, resolves the narration set,
    refuses any non-XHTML content document, and projects the narration scale.

    A `--dry-run` returns after reporting the projection; a full run then repacks the
    book into the output EPUB (still a passthrough of the upgraded copy). Sentence-span
    injection and synthesis slot in before the repack in a later phase.

    Args:
        config: The resolved run configuration.
        reporter: Progress sink; defaults to stderr reporting.
        validator: EPUB validator invoked only when `config.validate` is set.

    Returns:
        0 on success, nonzero if validation fails.
    """
    reporter = reporter or ProgressReporter()

    if config.clean and config.work_dir.exists():
        shutil.rmtree(config.work_dir)

    # Rebuild book/ from scratch every run: unzip writes members without clearing the
    # directory first, so a stale tree from an earlier (crashed or resumed) run would
    # otherwise bleed into this one. --clean additionally discards sibling work state.
    book_dir = config.work_dir / BOOK_SUBDIR
    if book_dir.exists():
        shutil.rmtree(book_dir)

    archive_order = epub_container.unzip(config.input, book_dir)
    package = epub2_upgrade.upgrade(book_dir, modified=_source_modified(config.input))

    landmarks = narration_set.read_landmarks(book_dir, package)
    selection = narration_set.apply_overrides(
        narration_set.select(package, landmarks=landmarks),
        include=config.include,
        exclude=config.exclude,
        pick=config.pick,
    )

    # Refuse a malformed content document whose root is not XHTML-namespaced before it
    # is silently narrated as empty; runs before projection so --dry-run fails too.
    content_namespace.require_xhtml_namespace(book_dir, selection.documents)
    projection = narration_projection.project(
        book_dir,
        selection.documents,
        bitrate=config.bitrate,
        speed=config.speed,
    )
    reporter.selection(selection)
    reporter.projection(projection)

    if config.dry_run:
        return 0

    epub_container.repack(book_dir, config.output, order=archive_order)
    reporter.info(f"wrote {config.output}")

    if config.validate:
        return _report_validation(validator(config.output), reporter)

    return 0


def _source_modified(path: Path) -> str:
    # Derive dcterms:modified from the input's mtime so an EPUB 2 upgrade is
    # reproducible across runs of the same file — never datetime.now(), which would
    # break the determinism guarantee.
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

    return modified.strftime("%Y-%m-%dT%H:%M:%SZ")


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
