import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

type TreeReader = Callable[[Path], dict[str, bytes]]

# Single availability probe for the optional external tools some suites gate on, so the
# ffmpeg check does not drift into four slightly different forms across the test tree.
HAS_FFMPEG = shutil.which("ffmpeg") is not None

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_EPUB_SRC = FIXTURES / "minimal_epub"
MINIMAL_EPUB2_SRC = FIXTURES / "minimal_epub2"


def _source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _zip_epub(src_root: Path, epub_path: Path) -> Path:
    """Zip a readable EPUB source tree into a valid .epub.

    Built with raw zipfile (mimetype first, STORED) so container tests do not depend on
    the production repack under test.
    """
    mimetype = src_root / "mimetype"
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in _source_files(src_root):
            if path == mimetype:
                continue

            zf.write(path, path.relative_to(src_root).as_posix())

    return epub_path


@pytest.fixture
def minimal_epub_src() -> Path:
    return MINIMAL_EPUB_SRC


@pytest.fixture
def minimal_epub2_src() -> Path:
    return MINIMAL_EPUB2_SRC


@pytest.fixture
def real_epub_path() -> Path:
    """A real-world EPUB from .epubs/ for integration checks; skipped when absent.

    The books there are copyrighted and gitignored, so this never runs in CI.
    """
    root = Path(__file__).resolve().parents[1]

    matches = sorted((root / ".epubs").glob("*.epub"))
    if not matches:
        pytest.skip("no real EPUB present in .epubs/")

    return matches[0]


@pytest.fixture
def minimal_epub_path(tmp_path: Path, minimal_epub_src: Path) -> Path:
    return _zip_epub(minimal_epub_src, tmp_path / "minimal.epub")


@pytest.fixture
def minimal_epub2_path(tmp_path: Path, minimal_epub2_src: Path) -> Path:
    return _zip_epub(minimal_epub2_src, tmp_path / "minimal2.epub")


@pytest.fixture
def read_file_tree() -> TreeReader:
    """Return a helper that snapshots a directory as {posix-relative-path: bytes}."""

    def read(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    return read


@pytest.fixture
def zip_epub_tree() -> Callable[[Path, Path], Path]:
    """Return a helper that zips an EPUB source tree into a valid .epub."""
    return _zip_epub
